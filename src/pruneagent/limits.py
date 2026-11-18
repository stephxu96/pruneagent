"""Codex usage limits (task 2.12, FR15): read the limit status after every call, estimate whether a run fits,
and say when to pause.

Where the readings come from: `codex exec --json` prints none. Codex writes them into the saved session of each
call (`$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<time>-<thread id>.jsonl`), in its `token_count` events:

    "rate_limits": {"limit_id": "codex", "plan_type": "plus", "rate_limit_reached_type": null,
                    "primary":   {"used_percent": 7.0, "window_minutes": 300,   "resets_at": 1794983552},
                    "secondary": {"used_percent": 1.0, "window_minutes": 10080, "resets_at": 1795384569},
                    "credits": {"has_credits": true, "unlimited": false, "balance": "..."}, ...}

`used_percent` is a whole number; `resets_at` is Unix time. The 300-minute window is the 5-hour limit and the
10080-minute one the weekly limit; a free plan shows one 30-day window instead, and some events carry no window
at all (`limit_id: "premium"`), so windows are told apart by their length, and events without one are skipped.
The limits belong to the ChatGPT account, so every reading, from any Codex session, shows the same state.

Pausing: at `PAUSE_AT` (90%) of any window, or when Codex says a limit is reached. A window whose reset time
has passed counts as 0% used.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PAUSE_AT = 90.0  # % of any window
FIVE_HOUR, WEEKLY = 300, 10080  # window lengths in minutes

# Before enough calls have been measured: % of each window one call uses (measured 2026-11-16 on the Plus plan:
# 5 calls used ~2% of the 5-hour window and < 1% of the weekly one; rounded up).
DEFAULT_PCT_PER_CALL = {FIVE_HOUR: 0.4, WEEKLY: 0.2}
FALLBACK_PCT_PER_CALL = 0.4  # any other window
# Tokens per call before any is measured: Codex's fixed ~3.1k plus a prompt of ~3k.
DEFAULT_TOKENS_PER_CALL = 6100
# Readings are whole percents, so the % per token is learned only once enough tokens have been seen.
MIN_TOKENS_TO_LEARN = 100_000
MIN_PCT_TO_LEARN = 2.0  # and the window must have risen by at least this much
RECENT_CALLS = 20  # calls averaged for the tokens per call


def window_name(minutes: int) -> str:
    return {FIVE_HOUR: "5-hour", WEEKLY: "weekly"}.get(minutes) or (
        f"{minutes // 1440}-day" if minutes % 1440 == 0 else f"{minutes}-minute")


def when(t: datetime) -> str:
    """A reset time for people: local time with its zone."""
    return t.astimezone().strftime("%Y-%m-%d %H:%M %Z")


@dataclass
class Window:
    minutes: int
    used_percent: float
    resets_at: datetime | None
    slot: str | None = None  # "primary" or "secondary", as Codex names it (and names a reached limit)

    @property
    def name(self) -> str:
        return window_name(self.minutes)

    def used_now(self, now: datetime | None = None) -> float:
        """% used now: 0 once the reset time has passed."""
        now = now or datetime.now(UTC)
        return 0.0 if self.resets_at is not None and now >= self.resets_at else self.used_percent


@dataclass
class LimitReading:
    """The limit status after one call. `reached` is Codex's own flag (which limit was hit), None when none."""

    time: datetime
    windows: list[Window] = field(default_factory=list)
    reached: str | None = None
    plan_type: str | None = None
    has_credits: bool | None = None
    credits_unlimited: bool | None = None

    # ------------------------------------------------------------------------------------- pause or not

    def pause_reason(self, now: datetime | None = None, pause_at: float = PAUSE_AT) -> str | None:
        """Why a run must pause now, with the reset time; None when it can go on."""
        now = now or datetime.now(UTC)
        hit = self.blocking(now, pause_at)
        if hit is None:
            if self.reached and not self.windows and (now - self.time).total_seconds() < FIVE_HOUR * 60:
                return f"Codex usage limit reached ({self.reached}); reset time not reported"
            return None
        if self.reached and hit.used_now(now) < pause_at:
            what = f"Codex usage limit reached ({self.reached})"
        else:
            what = f"Codex {hit.name} limit at {hit.used_now(now):.0f}% used (pause at {pause_at:.0f}%)"
        return what + (f"; resets at {when(hit.resets_at)}" if hit.resets_at else "")

    def blocking(self, now: datetime | None = None, pause_at: float = PAUSE_AT) -> Window | None:
        """The window that stops the run (the one resetting last when several do), or None."""
        now = now or datetime.now(UTC)
        live = [w for w in self.windows if w.resets_at is None or now < w.resets_at]
        over = [w for w in live if w.used_now(now) >= pause_at]
        if not over and self.reached and self.windows:
            # Codex says a limit is reached: the window it names, else the fullest one, unless it has reset since
            hit = next((w for w in self.windows if w.slot == self.reached), None) or max(
                self.windows, key=lambda w: w.used_percent)
            over = [hit] if hit in live else []
        if not over:
            return None
        return max(over, key=lambda w: w.resets_at or now)

    def resets_at(self, now: datetime | None = None, pause_at: float = PAUSE_AT) -> datetime | None:
        hit = self.blocking(now, pause_at)
        return hit.resets_at if hit else None

    def summary(self, now: datetime | None = None) -> str:
        now = now or datetime.now(UTC)
        parts = [f"{w.name} {w.used_now(now):.0f}% used" + (f" (resets {when(w.resets_at)})" if w.resets_at else "")
                 for w in self.windows]
        if self.reached:
            parts.append(f"limit reached: {self.reached}")
        return "; ".join(parts) or "no limit windows"

    # ------------------------------------------------------------------------------------------- storage

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time"] = self.time.isoformat()
        for w in d["windows"]:
            w["resets_at"] = w["resets_at"].isoformat(timespec="seconds") if w["resets_at"] else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LimitReading:
        wins = [Window(minutes=w["minutes"], used_percent=w["used_percent"],
                       resets_at=datetime.fromisoformat(w["resets_at"]) if w.get("resets_at") else None,
                       slot=w.get("slot"))
                for w in d.get("windows", [])]
        return cls(time=datetime.fromisoformat(d["time"]), windows=wins, reached=d.get("reached"),
                   plan_type=d.get("plan_type"), has_credits=d.get("has_credits"),
                   credits_unlimited=d.get("credits_unlimited"))


# ----------------------------------------------------------------------------------------------- reading


def from_rate_limits(rl: dict | None, time: datetime) -> LimitReading | None:
    """A reading from one `rate_limits` object; None when it has no window and no reached flag."""
    if not isinstance(rl, dict):
        return None
    wins = []
    for key in ("primary", "secondary"):
        w = rl.get(key)
        if isinstance(w, dict) and w.get("used_percent") is not None and w.get("window_minutes"):
            ts = w.get("resets_at")
            wins.append(Window(minutes=int(w["window_minutes"]), used_percent=float(w["used_percent"]),
                               resets_at=datetime.fromtimestamp(ts, UTC) if ts else None, slot=key))
    reached = rl.get("rate_limit_reached_type") or (
        "spend control" if rl.get("spend_control_reached") else None)
    if not wins and not reached:
        return None
    credits = rl.get("credits") if isinstance(rl.get("credits"), dict) else {}
    return LimitReading(time=time, windows=wins, reached=str(reached) if reached else None,
                        plan_type=rl.get("plan_type"), has_credits=credits.get("has_credits"),
                        credits_unlimited=credits.get("unlimited"))


def _time(s: str | None) -> datetime:
    try:
        return datetime.fromisoformat(s) if s else datetime.now(UTC)
    except ValueError:
        return datetime.now(UTC)


def read_session(path: Path | str) -> LimitReading | None:
    """The last usable limit reading in one saved Codex session, or None."""
    last = None
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if '"rate_limits"' not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else ev
        reading = from_rate_limits(payload.get("rate_limits"), _time(ev.get("timestamp")))
        if reading is not None:
            last = reading
    return last


def find_session(codex_home: Path | str, thread_id: str) -> Path | None:
    """The saved session of one call, by the thread id `codex exec --json` prints first."""
    found = sorted((Path(codex_home) / "sessions").glob(f"*/*/*/rollout-*{thread_id}.jsonl"))
    return found[-1] if found else None


def latest_session_reading(*codex_homes: Path | str, newest: int = 5) -> LimitReading | None:
    """The newest reading in the newest saved sessions of any of these Codex homes (only the limit status is read)."""
    files: list[Path] = []
    for home in codex_homes:
        files += (Path(home) / "sessions").glob("*/*/*/rollout-*.jsonl")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    readings = [r for f in files[:newest] if (r := read_session(f)) is not None]
    return max(readings, key=lambda r: r.time, default=None)


# ---------------------------------------------------------------------------------------------- estimate


@dataclass
class WindowEstimate:
    name: str
    used: float
    resets_at: datetime | None
    pct_per_call: float
    basis: str  # "measured over N calls" or "default"
    calls_fit: int


@dataclass
class RunEstimate:
    """Whether `calls` more agent calls fit before any window reaches the pause point."""

    calls: int
    tokens_per_call: int
    tokens_basis: str
    windows: list[WindowEstimate]
    reading_time: datetime | None
    reached: str | None = None
    pause_at: float = PAUSE_AT

    @property
    def calls_fit(self) -> int | None:
        if self.reached:
            return 0
        return min((w.calls_fit for w in self.windows), default=None)

    @property
    def fits(self) -> bool | None:
        """True or False; None when there is no reading to go by."""
        n = self.calls_fit
        return None if n is None else n >= self.calls

    def message(self) -> str:
        need = (f"up to {self.calls} agent calls (~{self.tokens_per_call / 1000:.1f}k tokens each, "
                f"~{self.calls * self.tokens_per_call / 1000:.0f}k in all)")
        if self.reading_time is None:
            return (f"Usage limits: this run needs {need}. No Codex limit reading yet, so whether it fits is not "
                    "known; the first call measures it, and the run pauses at "
                    f"{self.pause_at:.0f}% of a limit.")
        if self.reached:
            return f"Warning: Codex reports a usage limit reached ({self.reached}). " + self._options()
        tight = min(self.windows, key=lambda w: w.calls_fit)
        detail = "; ".join(f"{w.name} {w.used:.0f}% used, ~{w.pct_per_call:.2g}% per call ({w.basis})"
                           + (f", resets {when(w.resets_at)}" if w.resets_at else "") for w in self.windows)
        if self.fits:
            return f"Usage limits: this run needs {need}; about {tight.calls_fit} fit. {detail}."
        return (f"Warning: this run may need {need}, but only about {tight.calls_fit} fit before the Codex "
                f"{tight.name} limit reaches {self.pause_at:.0f}% ({detail}). " + self._options())

    @staticmethod
    def _options() -> str:
        return ("Options: start anyway (the run pauses at the limit and `pruneagent resume` continues after the "
                "reset), wait for the reset, or use the API backend.")


def learn_pct_per_token(history: list[tuple[int, LimitReading]]) -> dict[int, tuple[float, int, int]]:
    """% of each window per token, from consecutive readings in the same window period: {minutes: (% per token,
    calls, tokens)}, only for windows with at least `MIN_TOKENS_TO_LEARN` tokens seen and a rise of at least
    `MIN_PCT_TO_LEARN`. `history` is
    (tokens of the call, reading after it), oldest first."""
    pct: dict[int, float] = {}
    tok: dict[int, int] = {}
    calls: dict[int, int] = {}
    for (_, prev), (tokens, cur) in itertools.pairwise(history):
        before = {(w.minutes, w.resets_at): w.used_percent for w in prev.windows}
        for w in cur.windows:
            b = before.get((w.minutes, w.resets_at))
            if b is None or w.used_percent < b or not tokens:
                continue
            pct[w.minutes] = pct.get(w.minutes, 0.0) + (w.used_percent - b)
            tok[w.minutes] = tok.get(w.minutes, 0) + tokens
            calls[w.minutes] = calls.get(w.minutes, 0) + 1
    return {m: (pct[m] / tok[m], calls[m], tok[m]) for m in pct
            if tok[m] >= MIN_TOKENS_TO_LEARN and pct[m] >= MIN_PCT_TO_LEARN}


def estimate_run(reading: LimitReading | None, calls: int, history: list[tuple[int, LimitReading]] = (),
                 recent_tokens: list[int] = (), now: datetime | None = None,
                 pause_at: float = PAUSE_AT) -> RunEstimate:
    """`calls` agent calls times the tokens per call, against what is left of each window below `pause_at`."""
    now = now or datetime.now(UTC)
    recent = [t for t in recent_tokens if t][-RECENT_CALLS:]
    if recent:
        tokens, tokens_basis = round(sum(recent) / len(recent)), f"average of the last {len(recent)} calls"
    else:
        tokens, tokens_basis = DEFAULT_TOKENS_PER_CALL, "default"
    if reading is None:
        return RunEstimate(calls, tokens, tokens_basis, [], None, pause_at=pause_at)
    learned = learn_pct_per_token(list(history))
    wins = []
    for w in reading.windows:
        if w.minutes in learned:
            per_tok, n, _ = learned[w.minutes]
            per_call, basis = per_tok * tokens, f"measured over {n} calls"
        else:
            per_call, basis = DEFAULT_PCT_PER_CALL.get(w.minutes, FALLBACK_PCT_PER_CALL), "default"
        used = w.used_now(now)
        room = max(0.0, pause_at - used)
        fit = math.floor(room / per_call + 1e-9) if per_call > 0 else 10**6
        wins.append(WindowEstimate(w.name, used, w.resets_at if used else None, per_call, basis, fit))
    reached = reading.reached if reading.blocking(now, pause_at) is not None and reading.reached else None
    return RunEstimate(calls, tokens, tokens_basis, wins, reading.time, reached=reached, pause_at=pause_at)
