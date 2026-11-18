"""Agent client (task 2.3): one interface, two backends, a shared cache and a usage ledger.

    client = AgentClient(CodexBackend())            # or AgentClient(ApiBackend(...))
    reply = client.propose(AgentRequest(prompt, schema))
    reply.data                                       # the parsed JSON answer

- `CodexBackend` (default): `codex exec` on the ChatGPT plan, from an empty temporary directory, read-only,
  pinned model and reasoning effort, and a trimmed context (task 2.10; see `CODEX_TRIM` and README "Agent
  backend"). Sessions are saved, in a Codex home of their own (`DEFAULT_CODEX_HOME`), because only a saved
  session holds the usage-limit readings (task 2.12, `limits`); the login is shared with ~/.codex.
- `ApiBackend`: an LLM API with structured output (OpenAI Responses API format). The HTTP transport is
  injectable; tests use a fake one. The key comes from the environment or `.env` and is never logged.
- Cache: one JSON file per request, keyed by a hash of the prompt, the schema and the backend settings, so
  an identical request is answered from disk without a call.
- Ledger: one JSONL line per call attempt and per cache hit: time, run, backend, model, effort, tokens,
  seconds, cost (API), outcome.
- One retry on a transient failure (timeout, dropped connection, server error); a usage limit, a login
  problem or a malformed answer is not retried.
- Usage limits (task 2.12): the limit reading of every Codex call goes into the ledger. A reply whose reading is
  at 90% of a limit carries `pause_reason` (the graph pauses after that round); a call is refused before it is
  sent while the last reading says a limit is still at 90% or reached (`LimitPause`); `estimate_run` says
  before a run whether its calls fit. The API backend's cost cap pauses a run too (`CostCapReached`).

`AgentRequest` / `AgentReply` are deliberately minimal so they can be swapped for task 2.1's state types.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pruneagent import limits as L

# ------------------------------------------------------------------------------------------- pinned settings

CODEX_MODEL = "gpt-5.6-terra"
CODEX_REASONING_EFFORT = "high"
INSTRUCTIONS_FILE = Path(__file__).with_name("codex_instructions.md")

DEFAULT_CACHE_DIR = Path("runs/agent_cache")
DEFAULT_LEDGER = Path("runs/agent_usage.jsonl")
# Codex home for the agent's calls: its saved sessions stay out of Steph's own Codex history. Its auth.json is a
# link to the login in ~/.codex (or $CODEX_HOME). Override with PRUNEAGENT_CODEX_HOME.
DEFAULT_CODEX_HOME = Path.home() / ".codex-pruneagent"
MAX_CALLS_PER_RUN = 60
MAX_COST_PER_RUN_USD = 5.0  # API backend only

# Settings that remove Codex's own instructions, skills, plugins, apps and tools from every call
# (measured 2026-11-17, one-word reply: 19,426 input tokens before, 3,135 after). Passed as `-c` / `--disable`
# on top of `--ignore-user-config`, so Steph's own ~/.codex/config.toml (plugins, MCP servers) is not loaded;
# the login in ~/.codex is still used.
CODEX_TRIM: dict[str, Any] = {
    "project_doc_max_bytes": 0,                      # no AGENTS.md
    "include_permissions_instructions": False,
    "include_apps_instructions": False,
    "include_collaboration_mode_instructions": False,
    "include_environment_context": False,
    "skills.include_instructions": False,            # no skill list (~5k tokens)
    "skills.bundled.enabled": False,
    "web_search": "disabled",
}
CODEX_DISABLED_FEATURES = (
    "plugins", "apps", "multi_agent", "goals", "image_generation", "browser_use", "computer_use", "in_app_browser",
    "personality", "hooks", "tool_suggest", "skill_search", "shell_tool", "unified_exec", "shell_snapshot",
    "code_mode_host", "view_image", "sleep_tool", "workspace_dependencies",
)


# -------------------------------------------------------------------------------------------------- types


@dataclass(frozen=True)
class AgentRequest:
    prompt: str
    schema: dict | None = None   # JSON Schema of the answer; checked by the backend where it can
    tag: str = ""                # free label for the ledger, e.g. "round 3"
    # keys a reply must have to be read; None: the schema's required keys. Not part of the cache key.
    required: tuple[str, ...] | None = None


@dataclass
class Usage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass
class BackendResult:
    text: str
    usage: Usage
    model: str
    limits: L.LimitReading | None = None  # Codex only: the limit status after this call


@dataclass
class AgentReply:
    data: dict
    text: str
    backend: str
    model: str
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0
    cached: bool = False
    key: str = ""
    limits: L.LimitReading | None = None
    pause_reason: str | None = None  # set when this call's reading reached the pause point: pause after this round


class AgentError(Exception):
    """A call that failed for a reason a retry will not fix (login, bad request)."""


class TransientError(AgentError):
    """Timeout, dropped connection or server error: retried once."""


class UsageLimitReached(AgentError):
    """Codex plan limit or API quota reached: not retried; the run pauses (task 2.12). `resets_at` when known."""

    def __init__(self, message: str = "", resets_at: datetime | None = None,
                 reading: L.LimitReading | None = None):
        super().__init__(message)
        self.resets_at = resets_at
        self.reading = reading


class LimitPause(UsageLimitReached):
    """Not sent: the last reading says a limit is at the pause point (90%) or reached, and has not reset."""


class MalformedReply(AgentError):
    """The answer has no JSON object of the expected shape."""


class CapReached(AgentError):
    """The run's call cap or cost cap would be exceeded."""


class CostCapReached(CapReached, UsageLimitReached):
    """The API backend's cost cap: the run pauses (resume on the Codex backend or with a higher cap)."""


class Backend(Protocol):
    name: str
    model: str

    def identity(self) -> dict: ...

    def complete(self, prompt: str, schema: dict | None, timeout: float) -> BackendResult: ...


# --------------------------------------------------------------------------------------------- parsing


def extract_json(text: str, required: list[str] | tuple[str, ...] = ()) -> dict:
    """The first JSON object in `text` that has every key in `required`: the whole reply, a fenced block,
    or an object surrounded by prose. Raises MalformedReply."""
    dec = json.JSONDecoder()
    stripped = text.strip()
    candidates = [stripped] + re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    for chunk in candidates:
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(k in obj for k in required):
            return obj
    seen: list[str] = []
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if all(k in obj for k in required):
                return obj
            seen = seen or sorted(obj)
    need = f" with keys {list(required)}" if required else ""
    found = f" (found an object with keys {seen})" if seen else ""
    raise MalformedReply(f"no JSON object{need} in the reply{found}: {text[:200]!r}")


def thread_id(stdout: str) -> str | None:
    """The session id `codex exec --json` prints first (`thread.started`)."""
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "thread.started":
            return ev.get("thread_id")
    return None


def tokens_used(text: str) -> int | None:
    """The "tokens used" figure Codex prints at the end of a plain (non --json) run."""
    m = re.search(r"tokens used\s*:?\s*\n?\s*([\d,]+)", text, flags=re.IGNORECASE)
    return int(m.group(1).replace(",", "")) if m else None


_LIMIT = re.compile(r"usage limit|rate limit|quota|too many requests|\b429\b|purchase more credits", re.IGNORECASE)
_AUTH = re.compile(r"not logged in|log ?in again|codex login|unauthori[sz]ed|\b401\b|\b403\b|api key", re.IGNORECASE)


def classify_failure(message: str) -> type[AgentError]:
    if _LIMIT.search(message):
        return UsageLimitReached
    if _AUTH.search(message):
        return AgentError
    return TransientError


# ---------------------------------------------------------------------------------------------- Codex


Runner = Callable[..., subprocess.CompletedProcess]


def _toml(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return str(v)
    return json.dumps(str(v))


def _login_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


class CodexBackend:
    """`codex exec` with the trimmed, pinned setup. `run` is injectable (tests pass a fake).

    Sessions are saved (not `--ephemeral`) so each call's usage-limit reading can be read back, under
    `codex_home` (default `DEFAULT_CODEX_HOME`, or PRUNEAGENT_CODEX_HOME), not in Steph's own Codex history.
    Its `auth.json` is a link to the login in `login_home` (~/.codex); if Codex ever replaces the link with a
    file (a refreshed login), the newer file is copied back there and the link restored, so both stay logged in.
    `ephemeral=True` saves nothing and uses the normal Codex home (no limit readings)."""

    name = "codex"

    def __init__(self, model: str = CODEX_MODEL, reasoning_effort: str = CODEX_REASONING_EFFORT,
                 instructions_file: Path = INSTRUCTIONS_FILE, trim: bool = True, ephemeral: bool = False,
                 codex_home: Path | str | None = None, login_home: Path | str | None = None,
                 executable: str = "codex", run: Runner = subprocess.run):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.instructions_file = Path(instructions_file).resolve()
        self.trim = trim
        self.ephemeral = ephemeral
        self.codex_home = Path(codex_home or os.environ.get("PRUNEAGENT_CODEX_HOME") or DEFAULT_CODEX_HOME)
        self.login_home = Path(login_home) if login_home else _login_home()
        self.executable = executable
        self._run = run

    # ------------------------------------------------------------------------------- own Codex home

    def prepare_home(self) -> None:
        """Create `codex_home` with `auth.json` linked to the login in `login_home`; copy a refreshed login back."""
        self.codex_home.mkdir(parents=True, exist_ok=True)
        link, src = self.codex_home / "auth.json", self.login_home / "auth.json"
        if link.is_symlink() or not src.is_file() or link.resolve() == src.resolve():
            return
        if link.exists():  # Codex replaced the link with a file when it refreshed the login
            if link.stat().st_mtime > src.stat().st_mtime:
                tmp = src.with_name(".auth.json.pruneagent-tmp")
                shutil.copy2(link, tmp)
                os.chmod(tmp, 0o600)
                tmp.replace(src)
            link.unlink()
        link.symlink_to(src)

    def limits_after(self, thread_id: str | None) -> L.LimitReading | None:
        """The limit reading saved with one call's session, or None (no thread id, no session, no reading)."""
        if self.ephemeral or not thread_id:
            return None
        path = L.find_session(self.codex_home, thread_id)
        return L.read_session(path) if path else None

    def identity(self) -> dict:
        instr = self.instructions_file.read_text() if self.trim else ""
        return {"backend": self.name, "model": self.model, "reasoning_effort": self.reasoning_effort,
                "trim": self.trim, "instructions_sha": hashlib.sha256(instr.encode()).hexdigest()[:16]}

    def command(self, schema_path: Path | None = None) -> list[str]:
        cmd = [self.executable, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--json",
               "--color", "never", "--model", self.model, "-c", f"model_reasoning_effort={_toml(self.reasoning_effort)}"]
        if self.ephemeral:
            cmd.append("--ephemeral")
        if self.trim:
            cmd += ["--ignore-user-config", "--ignore-rules",
                    "-c", f"model_instructions_file={_toml(str(self.instructions_file))}"]
            for k, v in CODEX_TRIM.items():
                cmd += ["-c", f"{k}={_toml(v)}"]
            for f in CODEX_DISABLED_FEATURES:
                cmd += ["--disable", f]
        if schema_path is not None:
            cmd += ["--output-schema", str(schema_path)]
        return cmd + ["-"]  # prompt on stdin (no argument-length limit)

    def complete(self, prompt: str, schema: dict | None, timeout: float) -> BackendResult:
        with tempfile.TemporaryDirectory(prefix="pruneagent-codex-") as tmp:
            work = Path(tmp) / "empty"
            work.mkdir()
            schema_path = None
            if schema is not None:
                schema_path = Path(tmp) / "schema.json"
                schema_path.write_text(json.dumps(schema))
            env = None
            if not self.ephemeral:
                self.prepare_home()
                env = {**os.environ, "CODEX_HOME": str(self.codex_home)}
            try:
                proc = self._run(self.command(schema_path), cwd=work, input=prompt, capture_output=True, text=True,
                                 timeout=timeout, check=False, env=env)
            except subprocess.TimeoutExpired as e:
                raise TransientError(f"codex exec timed out after {timeout:.0f} s") from e
            finally:
                if not self.ephemeral:
                    self.prepare_home()
        stdout = proc.stdout or ""
        reading = self.limits_after(thread_id(stdout))
        try:
            res = self.parse_output(stdout, proc.stderr or "", proc.returncode)
        except UsageLimitReached as e:
            e.reading = reading
            e.resets_at = reading.resets_at() if reading else None
            raise
        except AgentError as e:  # a limit Codex reports only in its session
            if reading is not None and reading.reached:
                raise UsageLimitReached(str(e), reading.resets_at(), reading) from e
            raise
        res.limits = reading
        return res

    def parse_output(self, stdout: str, stderr: str, returncode: int) -> BackendResult:
        """Final message and token usage from `codex exec --json` events (falls back to "tokens used")."""
        messages: list[str] = []
        errors: list[str] = []
        failed = False
        usage = Usage()
        for line in stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            item = ev.get("item") or {}
            if kind == "item.completed" and item.get("type") == "agent_message":
                messages.append(item.get("text", ""))
            elif kind == "turn.completed":
                u = ev.get("usage") or {}
                usage = Usage(input_tokens=u.get("input_tokens"), cached_input_tokens=u.get("cached_input_tokens"),
                              output_tokens=u.get("output_tokens"), reasoning_tokens=u.get("reasoning_output_tokens"))
            elif kind == "turn.failed":
                failed = True
                errors.append(str((ev.get("error") or {}).get("message", ev)))
            elif kind == "error":
                errors.append(str(ev.get("message", ev)))
        if usage.input_tokens is None and (n := tokens_used(stderr + "\n" + stdout)) is not None:
            usage = Usage(input_tokens=n)  # plain-text total only
        if returncode != 0 or failed or not messages:
            msg = "; ".join(errors) or stderr.strip()[-500:] or f"codex exec exited with {returncode}"
            raise classify_failure(msg)(f"codex exec failed (exit {returncode}): {msg}")
        return BackendResult(text=messages[-1], usage=usage, model=self.model)


# ------------------------------------------------------------------------------------------------ API


Transport = Callable[[str, dict, dict, float], tuple[int, dict]]


def _urllib_transport(url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            body = {}
        return e.code, body
    except (urllib.error.URLError, TimeoutError) as e:
        raise TransientError(f"API request failed: {e}") from e


def read_api_key(var: str = "OPENAI_API_KEY", env_file: Path | str = ".env") -> str | None:
    """The key from the environment, else from a `.env` file (ignored by git). Never printed or logged."""
    if os.environ.get(var):
        return os.environ[var]
    path = Path(env_file)
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k.strip() == var:
            return v.strip().strip("'\"") or None
    return None


class ApiBackend:
    """LLM API backend (OpenAI Responses API with a JSON-schema answer). Not used for real calls yet: the
    transport is injectable, and tests use a fake one. Cost needs prices per million tokens."""

    name = "api"

    def __init__(self, model: str = CODEX_MODEL, reasoning_effort: str | None = CODEX_REASONING_EFFORT,
                 provider: str = "openai", url: str = "https://api.openai.com/v1/responses",
                 api_key: str | None = None, key_var: str = "OPENAI_API_KEY", env_file: Path | str = ".env",
                 price_in_per_mtok: float | None = None, price_out_per_mtok: float | None = None,
                 max_output_tokens: int = 2000, instructions_file: Path = INSTRUCTIONS_FILE,
                 transport: Transport = _urllib_transport):
        self.model, self.reasoning_effort, self.provider, self.url = model, reasoning_effort, provider, url
        self._api_key, self.key_var, self.env_file = api_key, key_var, env_file
        self.price_in, self.price_out = price_in_per_mtok, price_out_per_mtok
        self.max_output_tokens = max_output_tokens
        self.instructions = Path(instructions_file).read_text().strip()
        self._transport = transport

    def identity(self) -> dict:
        return {"backend": self.name, "provider": self.provider, "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "instructions_sha": hashlib.sha256(self.instructions.encode()).hexdigest()[:16]}

    def payload(self, prompt: str, schema: dict | None) -> dict:
        fmt = ({"type": "json_schema", "name": "agent_reply", "schema": schema, "strict": True} if schema
               else {"type": "json_object"})
        body: dict[str, Any] = {"model": self.model, "instructions": self.instructions, "input": prompt,
                                "text": {"format": fmt}, "max_output_tokens": self.max_output_tokens}
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def complete(self, prompt: str, schema: dict | None, timeout: float) -> BackendResult:
        key = self._api_key or read_api_key(self.key_var, self.env_file)
        if not key:
            raise AgentError(f"no API key: set {self.key_var} in the environment or in .env")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        status, body = self._transport(self.url, headers, self.payload(prompt, schema), timeout)
        if status != 200:
            err = body.get("error") or {}
            msg = f"API returned {status}: {err.get('code') or ''} {err.get('message') or ''}".strip()
            if status == 429:
                kind = UsageLimitReached if err.get("code") == "insufficient_quota" else TransientError
            elif status >= 500 or status == 408:
                kind = TransientError
            else:
                kind = AgentError
            raise kind(msg)
        return BackendResult(text=self._text(body), usage=self._usage(body.get("usage") or {}),
                             model=body.get("model", self.model))

    @staticmethod
    def _text(body: dict) -> str:
        if isinstance(body.get("output_text"), str):
            return body["output_text"]
        parts = [c.get("text", "") for item in body.get("output", []) if item.get("type") == "message"
                 for c in item.get("content", []) if c.get("type") == "output_text"]
        return "".join(parts)

    def _usage(self, u: dict) -> Usage:
        n_in, n_out = u.get("input_tokens"), u.get("output_tokens")
        cost = None
        if self.price_in is not None and self.price_out is not None and n_in is not None and n_out is not None:
            cost = round((n_in * self.price_in + n_out * self.price_out) / 1e6, 6)
        return Usage(input_tokens=n_in, cached_input_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens"),
                     output_tokens=n_out, reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                     cost_usd=cost)


# --------------------------------------------------------------------------------------- cache and ledger


class Cache:
    """One JSON file per request under `root`, shared by both backends (the key includes the backend)."""

    def __init__(self, root: Path | str = DEFAULT_CACHE_DIR):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def put(self, key: str, record: dict) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(p)


class Ledger:
    """Append-only JSONL of calls: one line per attempt and per cache hit."""

    def __init__(self, path: Path | str = DEFAULT_LEDGER):
        self.path = Path(path)

    def append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def entries(self, run_id: str | None = None) -> list[dict]:
        if not self.path.is_file():
            return []
        out = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return [e for e in out if run_id is None or e.get("run_id") == run_id]

    def totals(self, run_id: str | None = None) -> dict:
        """Calls sent (cache hits excluded), tokens and cost for one run (or all)."""
        sent = [e for e in self.entries(run_id) if not e.get("cached")]
        return {"calls": len(sent),
                "tokens": sum(e.get("total_tokens") or 0 for e in sent),
                "cost_usd": round(sum(e.get("cost_usd") or 0.0 for e in sent), 6),
                "cache_hits": sum(1 for e in self.entries(run_id) if e.get("cached"))}

    def last_limits(self) -> L.LimitReading | None:
        """The newest usage-limit reading of any run (the limits belong to the account, not to a run)."""
        found = [L.LimitReading.from_dict(e["limits"]) for e in self.entries() if e.get("limits")]
        newest = max(range(len(found)), key=lambda i: (found[i].time, i), default=None)  # ties: the later line
        return None if newest is None else found[newest]

    def limit_history(self) -> list[tuple[int, L.LimitReading]]:
        """(tokens of the call, reading after it) for every answered call with a reading, oldest first."""
        return [(e.get("total_tokens") or 0, L.LimitReading.from_dict(e["limits"])) for e in self.entries()
                if e.get("limits") and not e.get("cached") and e.get("ok")]

    def recent_tokens(self, backend: str | None = None) -> list[int]:
        return [e["total_tokens"] for e in self.entries() if not e.get("cached") and e.get("ok")
                and e.get("total_tokens") and (backend is None or e.get("backend") == backend)]


def cache_key(request: AgentRequest, backend: Backend) -> str:
    blob = json.dumps({"prompt": request.prompt, "schema": request.schema, "backend": backend.identity()},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# -------------------------------------------------------------------------------------------------- client


class AgentClient:
    """`propose(request) -> AgentReply`: cache, caps, one retry on a transient failure, JSON extraction, ledger."""

    def __init__(self, backend: Backend | None = None, cache: Cache | None = None, ledger: Ledger | None = None,
                 run_id: str | None = None, max_calls: int = MAX_CALLS_PER_RUN,
                 max_cost_usd: float | None = MAX_COST_PER_RUN_USD, timeout: float = 180.0, retries: int = 1,
                 retry_wait: float = 5.0, sleep: Callable[[float], None] = time.sleep,
                 pause_at: float = L.PAUSE_AT, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.backend = backend if backend is not None else CodexBackend()
        self.cache = cache if cache is not None else Cache()
        self.ledger = ledger if ledger is not None else Ledger()
        self.run_id = run_id
        self.max_calls, self.max_cost_usd = max_calls, max_cost_usd
        self.timeout, self.retries, self.retry_wait, self._sleep = timeout, retries, retry_wait, sleep
        prior = self.ledger.totals(run_id) if run_id else {"calls": 0, "cost_usd": 0.0}
        self.calls, self.cost_usd = prior["calls"], prior["cost_usd"]
        self.pause_at, self._now = pause_at, now
        self.limits = self.ledger.last_limits() if self.backend.name == "codex" else None

    # ---------------------------------------------------------------------------------------- usage limits

    def limit_reading(self) -> L.LimitReading | None:
        """The last reading this client knows (from its calls or the ledger), else the newest saved session in
        the backend's Codex home or the login's (the limits belong to the account, so any session will do)."""
        if self.limits is not None or self.backend.name != "codex":
            return self.limits
        homes = [h for h in (getattr(self.backend, "codex_home", None), getattr(self.backend, "login_home", None))
                 if h is not None]
        return L.latest_session_reading(*homes) if homes else None

    def estimate_run(self, calls: int) -> L.RunEstimate | None:
        """Whether `calls` more calls fit the remaining Codex limits (None on a backend without plan limits)."""
        if self.backend.name != "codex":
            return None
        return L.estimate_run(self.limit_reading(), calls, self.ledger.limit_history(),
                              self.ledger.recent_tokens(self.backend.name), now=self._now(),
                              pause_at=self.pause_at)

    def _check_limits(self) -> None:
        """Refuse to send while the last reading is at the pause point and its window has not reset."""
        if self.limits is None:
            return
        reason = self.limits.pause_reason(self._now(), self.pause_at)
        if reason:
            raise LimitPause(reason, self.limits.resets_at(self._now(), self.pause_at), self.limits)

    def _log(self, request: AgentRequest, key: str, **fields) -> None:
        self.ledger.append({"time": datetime.now(UTC).isoformat(timespec="seconds"), "run_id": self.run_id,
                            "tag": request.tag, "key": key[:16], **self.backend.identity(), **fields})

    def _check_caps(self) -> None:
        if self.calls >= self.max_calls:
            raise CapReached(f"call cap reached: {self.calls} of {self.max_calls} calls used")
        if self.backend.name == "api" and self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            raise CostCapReached(f"API cost cap reached: ${self.cost_usd:.2f} of ${self.max_cost_usd:.2f}")

    def propose(self, request: AgentRequest) -> AgentReply:
        key = cache_key(request, self.backend)
        required = (request.required if request.required is not None
                    else tuple((request.schema or {}).get("required", ())))
        if (hit := self.cache.get(key)) is not None:
            self._log(request, key, cached=True, ok=True, seconds=0.0, total_tokens=0)
            return AgentReply(data=hit["data"], text=hit["text"], backend=hit["backend"], model=hit["model"],
                              usage=Usage(**hit.get("usage", {})), seconds=0.0, cached=True, key=key)
        attempt = 0
        while True:
            self._check_caps()
            self._check_limits()
            attempt += 1
            self.calls += 1
            start = time.perf_counter()
            try:
                res = self.backend.complete(request.prompt, request.schema, self.timeout)
            except TransientError as e:
                self._log(request, key, cached=False, ok=False, attempt=attempt, error=str(e)[:500],
                          seconds=round(time.perf_counter() - start, 2))
                if attempt <= self.retries:
                    self._sleep(self.retry_wait)
                    continue
                raise
            except AgentError as e:
                extra = {}
                if isinstance(e, UsageLimitReached) and e.reading is not None:
                    self.limits = e.reading
                    extra["limits"] = e.reading.to_dict()
                self._log(request, key, cached=False, ok=False, attempt=attempt, error=str(e)[:500],
                          error_kind=type(e).__name__, seconds=round(time.perf_counter() - start, 2), **extra)
                raise
            seconds = round(time.perf_counter() - start, 2)
            self.cost_usd += res.usage.cost_usd or 0.0
            usage = {**asdict(res.usage), "total_tokens": res.usage.total_tokens}
            if res.limits is not None:
                self.limits = res.limits
                usage["limits"] = res.limits.to_dict()
            try:
                data = extract_json(res.text, required)
            except MalformedReply as e:
                self._log(request, key, cached=False, ok=False, attempt=attempt, error=str(e)[:500],
                          error_kind="MalformedReply", seconds=seconds, **usage)
                raise
            self._log(request, key, cached=False, ok=True, attempt=attempt, seconds=seconds, **usage)
            self.cache.put(key, {"key": key, "data": data, "text": res.text, "backend": self.backend.name,
                                 "model": res.model, "usage": asdict(res.usage), "identity": self.backend.identity(),
                                 "created": datetime.now(UTC).isoformat(timespec="seconds")})
            pause = res.limits.pause_reason(self._now(), self.pause_at) if res.limits else None
            return AgentReply(data=data, text=res.text, backend=self.backend.name, model=res.model, usage=res.usage,
                              seconds=seconds, cached=False, key=key, limits=res.limits, pause_reason=pause)
