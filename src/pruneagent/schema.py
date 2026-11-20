"""State types (ARCHITECTURE.md §4.3 and §7, task 2.1): everything a run reads, writes, logs or shows the agent.

All types are pydantic models that reject unknown fields, so a typo or an extra field in a saved file or an agent
reply is an error rather than silently ignored. Each one round-trips through JSON (`model_dump_json` /
`model_validate_json`); plan layer keys become strings in JSON and come back as ints.

Existing types are reused, not copied: `plan.LayerPlan` (and `Plan`), `estimate.DeviceTarget` (memory limit
required) and `estimate.Estimate` are validated as they are.

Rules that need the model's shape (layer indices, MLP widths, step sizes, the memory budget) are checked by
`validator`, not here: these types check only what can be checked without the model.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from pruneagent.estimate import REPORT_CONTEXT, DeviceTarget, Estimate, layer_cost_s, speed_machine
from pruneagent.plan import STRATEGIES, LayerPlan
from pruneagent.search import SearchName
from pruneagent.search import get as search_strategy

Strategy = Literal["narrow_mlp", "remove_attention_group", "drop_layer", "wanda_sparsity"]
assert set(Strategy.__args__) == set(STRATEGIES)  # one menu (plan.STRATEGIES)

Plan = dict[int, LayerPlan]
Action = Literal["probe", "apply", "start_from_flap", "rollback", "stop"]
V1_ACTIONS: tuple[str, ...] = ("apply", "rollback")  # what the agent may do in v1 (task 2.9); see RunConfig


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------------------------ run settings


class Calibration(_Model):
    """The one calibration set every method uses (agent scores and quick tests, Baseline: FLAP, any backup)."""

    dataset: str = "c4"
    n_samples: PositiveInt = 128
    seqlen: PositiveInt = 2048
    seed: int = 0


class Rules(_Model):
    """The fixed rules of §4.4. The validator and the gate read them from here."""

    max_same_steps: PositiveInt = 2  # same strategy on one layer, per candidate
    max_layers_per_candidate: PositiveInt | None = None  # None: any number (search strategies, search.py)
    max_drops_per_candidate: NonNegativeInt = 1
    max_drops_per_run: NonNegativeInt = 6
    min_mlp_fraction: float = Field(0.25, ge=0, le=1)  # of the layer's original MLP width
    min_groups: PositiveInt = 1  # key/value groups per remaining layer
    max_wanda_sparsity: float = Field(0.70, ge=0, le=1)
    max_candidates: PositiveInt = 3  # per round
    step_limit: float = Field(0.15, gt=0)  # reject a candidate whose perplexity rises more than this per step
    max_rounds: PositiveInt = 40
    max_rounds_without_accept: PositiveInt = 5
    max_invalid_in_row: PositiveInt = 3


class RunConfig(_Model):
    """One run: model, device target (memory limit required), rules, agent backend and limits."""

    model_id: str
    baseline_method: Literal["flap"] = "flap"
    device: DeviceTarget
    calibration: Calibration = Calibration()
    rules: Rules = Rules()
    # how much one candidate may change (search.py, ADR-009); sets its rules unless `rules` sets them explicitly
    search: SearchName = "multi_layer"
    gate_windows: PositiveInt = 32  # validation windows used by the gate
    # per kept layer, added to each token's estimated time when ranking plans and checking a minimum speed (G1);
    # None: the value fitted for the M4 or the T4 (`estimate.LAYER_COST_S`), 0 for other devices
    layer_cost_s: NonNegativeFloat | None = None
    agent_backend: Literal["codex", "api"] = "codex"
    agent_provider: str | None = None
    agent_model: str | None = None
    agent_reasoning_effort: str | None = None
    # v1 (task 2.9): the agent proposes candidates or rolls back. True also allows probe, start_from_flap and stop
    # (the code is kept; the v1 instructions do not offer them).
    agent_extra_actions: bool = False
    # "shared": an identical prompt reuses a saved answer from any earlier run (cheap reruns). "run": answers are saved
    # per run only, so a repeat run with the same instructions asks the agent afresh (task 3.1 repeats); a resume of
    # the same run still reuses its own answers.
    agent_cache_scope: Literal["shared", "run"] = "shared"
    max_agent_calls: PositiveInt = 60
    max_cost_usd: NonNegativeFloat | None = None  # API backend only; defaults to $5 there
    seed: int = 0
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    gpu: str | None = None

    @model_validator(mode="after")
    def _search_rules(self) -> RunConfig:
        given = self.rules.model_fields_set
        extra = {k: v for k, v in search_strategy(self.search).rules.items() if k not in given}
        if extra:
            self.rules = self.rules.model_copy(update=extra)
        return self

    @model_validator(mode="after")
    def _cost_cap(self) -> RunConfig:
        if self.agent_backend == "api" and self.max_cost_usd is None:
            self.max_cost_usd = 5.0
        if self.agent_backend == "codex" and self.max_cost_usd is not None:
            raise ValueError("max_cost_usd applies to the API backend only (Codex has no per-call charge)")
        return self

    @model_validator(mode="after")
    def _speed_where_checked(self) -> RunConfig:
        if self.device.min_decode_tok_s is not None and speed_machine(self.device) is None:
            raise ValueError(f"a minimum speed can be checked only on the M4 or the T4, where the speed estimate "
                             f"was checked against measurements; {self.device.name!r} gets memory fit only")
        return self

    @property
    def layer_cost(self) -> float:
        """The per-layer cost in use: `layer_cost_s` if given, else the fitted value for the device."""
        return layer_cost_s(self.device) if self.layer_cost_s is None else self.layer_cost_s

    @property
    def limit_bytes(self) -> int:
        """The memory limit in bytes (GB = 1e9 bytes, as in `estimate`)."""
        return int(self.device.memory_limit_gb * 1e9)


# ------------------------------------------------------------------------------------------------ baseline


class Baseline(_Model):
    """Baseline: FLAP cut to the memory limit (the smallest FLAP ratio whose weights fit). FLAP allocates any MLP
    width, so its plan need not use the agent's 64-neuron steps. Test-split numbers are filled in at the end."""

    method: Literal["flap"] = "flap"
    calibration: Calibration = Calibration()
    memory_limit_gb: float = Field(gt=0)
    ratio: float = Field(ge=0, le=1)
    plan: Plan
    weight_bytes: PositiveInt
    val_ppl_subset: float | None = Field(None, gt=0)  # the gate's 32 windows: shown to the agent
    val_ppl: float | None = Field(None, gt=0)  # full validation split
    test_ppl: float | None = Field(None, gt=0)
    multiple_choice_mean: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def _fits(self) -> Baseline:
        if self.weight_bytes > self.memory_limit_gb * 1e9:
            raise ValueError(f"Baseline: FLAP weighs {self.weight_bytes:,} bytes, over its memory limit "
                             f"({self.memory_limit_gb} GB)")
        return self


# ------------------------------------------------------------------------------------------- sensitivity table


class QuickTest(_Model):
    """One quick test (layer-local probe): the layer's output error after `steps` steps of a strategy alone."""

    model_config = ConfigDict(extra="ignore")  # probe files also carry each test's `setting`
    strategy: Strategy
    steps: PositiveInt
    local_error: float | None = Field(None, ge=0)  # None when the steps are impossible


class SensitivityRow(_Model):
    """One layer of the sensitivity table. Quick tests compare strategies; the measured damage (whole-model
    validation perplexity with one step of a strategy on this layer alone, measured once before the search)
    compares layers (G1)."""

    layer: NonNegativeInt
    block_influence: float
    quick_tests: list[QuickTest] = []
    measured_ppl: dict[Strategy, float] = {}

    @model_validator(mode="before")
    @classmethod
    def _from_probe(cls, data: Any) -> Any:
        """Accept a `probe.LayerProbe` dict, whose quick tests are under `candidates`."""
        if isinstance(data, dict) and "candidates" in data and "quick_tests" not in data:
            data = {**data, "quick_tests": data["candidates"]}
            del data["candidates"]
        return data


class SensitivityTable(_Model):
    dense_ppl: float | None = Field(None, gt=0)  # the unpruned model on the windows the damage was measured on
    measured_windows: PositiveInt | None = None
    rows: dict[int, SensitivityRow]

    @model_validator(mode="after")
    def _keys(self) -> SensitivityTable:
        for k, row in self.rows.items():
            if k != row.layer:
                raise ValueError(f"row key {k} holds layer {row.layer}")
        return self

    @classmethod
    def from_probe_results(cls, results: dict) -> SensitivityTable:
        """From a `pruneagent probe` results file: `table` (quick tests) and `ground_truth` ("layer/strategy" ->
        {"ppl": ...}, plus "dense")."""
        truth = results.get("ground_truth") or {}
        rows = {}
        for key, probe in results["table"].items():
            measured = {k.split("/", 1)[1]: v["ppl"] for k, v in truth.items() if k.split("/", 1)[0] == str(key)}
            rows[int(key)] = SensitivityRow.model_validate({**probe, "measured_ppl": measured})
        dense = truth.get("dense", {})
        return cls(dense_ppl=dense.get("ppl"), measured_windows=dense.get("n_windows"), rows=rows)

    def with_measured_damage(self, damage: dict) -> SensitivityTable:
        """This table with its measured damage replaced by a damage-table file (`scripts/damage_table.py`):
        `dense` {"ppl"}, `windows`, and `entries` ("layer/strategy" -> {"layer", "strategy", "ppl", ...}).
        Every measured layer must already have a row (the rows carry the quick tests and Block Influence)."""
        measured: dict[int, dict[str, float]] = {i: {} for i in self.rows}
        for key, e in damage["entries"].items():
            if e["layer"] not in measured:
                raise ValueError(f"damage entry {key}: layer {e['layer']} has no row in the table")
            measured[e["layer"]][e["strategy"]] = e["ppl"]
        rows = {i: SensitivityRow.model_validate({**r.model_dump(), "measured_ppl": measured[i]})
                for i, r in self.rows.items()}
        return SensitivityTable(dense_ppl=damage["dense"]["ppl"], measured_windows=damage["windows"], rows=rows)


# -------------------------------------------------------------------------------------------- agent in and out
#
# The agent's instructions (task 2.9) call a strategy a "method" and a rationale a "reason", and name layers
# "layer 12"; older replies and the decision log use "strategy", "rationale" and integers. Both are accepted and
# normalised to the internal names (`normalize_reply`), so the log always reads the same.

_LAYER = re.compile(r"^\s*(?:layer\s*)?(\d+)\s*$", re.IGNORECASE)


def parse_layer(value: Any) -> Any:
    """12, "12" or "layer 12" -> 12; anything else is returned unchanged (the validator explains why)."""
    if isinstance(value, str) and (m := _LAYER.match(value)):
        return int(m.group(1))
    return value


def _rename(d: dict, alias: str, name: str) -> dict:
    if alias in d and name not in d:
        d = dict(d)
        d[name] = d.pop(alias)
    return d


def normalize_choice(raw: Any) -> Any:
    """{"layer": "layer 12", "method": m} -> {"layer": 12, "strategy": m}; non-objects unchanged."""
    if not isinstance(raw, dict):
        return raw
    d = _rename(raw, "method", "strategy")
    return {**d, "layer": parse_layer(d["layer"])} if "layer" in d else d


def normalize_candidate(raw: Any) -> Any:
    """A candidate as {"choices", "rationale"} (from "reason"), or a bare list of choices."""
    if isinstance(raw, list):
        return [normalize_choice(c) for c in raw]
    if not isinstance(raw, dict):
        return raw
    d = _rename(raw, "reason", "rationale")
    if "rationale" in d and d["rationale"] is None:  # a sweep reply's empty top-level reason: as if absent
        d = {k: v for k, v in d.items() if k != "rationale"}
    if isinstance(d.get("choices"), list):
        d = {**d, "choices": [normalize_choice(c) for c in d["choices"]]}
    return d


def normalize_reply(raw: Any) -> Any:
    """An agent reply in the internal form. A v1 reply has no "action": {"candidates": [...]} is "apply" and
    {"rollback_to": n} (with no or empty candidates) is "rollback"; its rationale comes from the candidates'
    reasons. A reply that gives both, or neither, is left without an action for the validator to explain. Aliases
    are renamed only when the internal name is absent, so a reply that gives both keeps both (and is rejected for
    the extra field). Anything that is not an object is returned unchanged."""
    if not isinstance(raw, dict):
        return raw
    d = _rename(raw, "reason", "rationale")
    if isinstance(d.get("candidates"), list):
        d = {**d, "candidates": [normalize_candidate(c) for c in d["candidates"]]}
    if isinstance(d.get("probe"), list):
        d = {**d, "probe": [normalize_choice(c) for c in d["probe"]]}
    if "action" not in d:
        has_candidates = d.get("candidates") not in (None, [])
        has_target = d.get("rollback_to") is not None
        if has_target and not has_candidates:
            d = {k: v for k, v in d.items() if k != "candidates"}
            d["action"] = "rollback"
            d.setdefault("rationale", f"return to kept plan {d['rollback_to']}")
        elif has_candidates and not has_target:
            d = {k: v for k, v in d.items() if k != "rollback_to"}
            d["action"] = "apply"
            if "rationale" not in d:
                reasons = [c.get("rationale") for c in d["candidates"] if isinstance(c, dict)]
                d["rationale"] = " | ".join(r for r in reasons if isinstance(r, str) and r.strip()) or "(no reason)"
    return d


class Choice(_Model):
    """One choice: a strategy on a layer. Code applies one fixed step of it; there is no amount. Also read from
    {"layer": "layer 12", "method": ...}."""

    layer: NonNegativeInt
    strategy: Strategy

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        return normalize_choice(data)


class Candidate(_Model):
    """One candidate plan change: choices applied together, with the agent's reason ("rationale" or "reason")."""

    choices: list[Choice]
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        return normalize_candidate(data) if isinstance(data, dict) else data


class AgentReply(_Model):
    """What the agent answers (§4.5): an action, a rationale and, for apply, 1-3 candidates; for probe, the
    (layer, strategy) pairs to test; for rollback, which accepted plan to restore (default: the one before the
    current). No amounts anywhere. A candidate may also be a bare list of choices (§4.5's example); it is read as
    a candidate with an empty rationale. How many candidates, and whether they are allowed, is the validator's job,
    so that the agent gets a readable reason."""

    action: Action
    rationale: str
    candidates: list[Candidate] = []
    probe: list[Choice] = []
    rollback_to: NonNegativeInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _v1_form(cls, data: Any) -> Any:
        return normalize_reply(data)

    @field_validator("candidates", mode="before")
    @classmethod
    def _bare_lists(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [{"choices": c} if isinstance(c, list) else c for c in v]
        return v


class StepSummary(_Model):
    """One earlier round as the agent sees it."""

    round: NonNegativeInt
    action: Action | None  # None when the reply could not be read
    rationale: str
    outcome: str  # e.g. "ACCEPT candidate 2, perplexity 15.10 (+3.2%)" or "INVALID: ..."


class PastCandidate(_Model):
    """One evaluated candidate of an earlier round, as the agent sees it."""

    candidate: PositiveInt  # numbered from 1, as in the agent's reply
    choices: list[Choice]
    val_ppl: float = Field(gt=0)
    step_rise: float
    freed_bytes: int
    rise_per_mb: float | None = None
    result: str  # e.g. "kept: plan 1", "rejected: over +15%", "not kept: candidate 2 chosen"


class PastRound(_Model):
    """One earlier round, as the agent sees it: its candidates, or a note (an unusable reply, a rollback)."""

    round: PositiveInt
    candidates: list[PastCandidate] = []
    note: str = ""


class AgentRequest(_Model):
    """What the agent sees (§4.5). Memory in bytes, speed in tokens/s."""

    objective: str
    device: DeviceTarget
    strategy_menu: dict[Strategy, str]  # neutral, equal-detail description of one step of each
    rules: Rules
    search: SearchName = "multi_layer"  # search strategy (search.py): what one candidate may change
    layer_at: NonNegativeInt | None = None  # sweep strategies: the one layer this round decides
    sensitivity: SensitivityTable
    plan: Plan
    dense_weight_bytes: PositiveInt
    current_weight_bytes: PositiveInt
    limit_bytes: PositiveInt
    est_decode_tok_s: float | None = None
    dense_est_decode_tok_s: float | None = None
    current_ppl: float = Field(gt=0)
    last_accepted_ppl: float = Field(gt=0)
    baseline_ppl: float = Field(gt=0)  # Baseline: FLAP on the same gate windows
    recent: list[StepSummary] = []
    round: NonNegativeInt = 0
    # for the v1 instructions (task 2.9)
    n_layers: PositiveInt | None = None
    n_groups: PositiveInt | None = None  # key/value (attention) groups per layer in the unpruned model
    mlp_width: PositiveInt | None = None
    step_bytes: dict[Strategy, int] = {}  # weight bytes one step of each method frees on an unpruned layer
    dense_ppl: float | None = Field(None, gt=0)  # the unpruned model on the gate's windows
    rounds_left: NonNegativeInt | None = None  # including this one, within both the round and the call caps
    max_agent_calls: PositiveInt = 60
    since_kept: NonNegativeInt = 0  # rounds in a row without a kept step
    kept: list[AcceptedPlan] = []  # rollback targets, by index (0 is the unpruned model)
    past: list[PastRound] = []  # every earlier round


# ------------------------------------------------------------------------------------------- checks and results


class CandidateCheck(_Model):
    """The validator's verdict on one candidate; `plan` and `weight_bytes` only when it is valid."""

    index: NonNegativeInt
    ok: bool
    reasons: list[str] = []
    plan: Plan | None = None
    weight_bytes: PositiveInt | None = None


class CheckResult(_Model):
    """The validator's verdict on one reply. `reasons` are written for the agent and fed back to it."""

    ok: bool
    reasons: list[str] = []
    candidates: list[CandidateCheck] = []

    @model_validator(mode="after")
    def _consistent(self) -> CheckResult:
        if self.ok and (self.reasons or any(not c.ok for c in self.candidates)):
            raise ValueError("a passing check has no reasons and no failing candidate")
        if not self.ok and not self.reasons:
            raise ValueError("a failing check must give its reasons")
        return self

    def feedback(self) -> str:
        """The reasons as one message for the agent."""
        if self.ok:
            return "All candidates are valid."
        return "Your proposal was rejected:\n" + "\n".join(f"- {r}" for r in self.reasons)


class CandidateResult(_Model):
    """One evaluated candidate."""

    candidate_id: NonNegativeInt
    choices: list[Choice]
    rationale: str = ""
    plan: Plan
    estimate: Estimate
    rank_decode_tok_s: float | None = None  # estimate with the per-layer cost added: breaks perplexity ties
    val_ppl: float = Field(gt=0)
    step_rise: float  # val_ppl / last accepted val_ppl - 1
    freed_bytes: int = 0  # weight memory freed against the last accepted plan (0 or less: none)
    rise_per_mb: float | None = None  # step_rise per MB (1e6 bytes) freed; None when nothing is freed (can't win)


class GateResult(_Model):
    decision: Literal["ACCEPT", "ROLLBACK"]
    chosen_candidate: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _chosen(self) -> GateResult:
        if (self.decision == "ACCEPT") != (self.chosen_candidate is not None):
            raise ValueError("ACCEPT names the chosen candidate; ROLLBACK names none")
        return self


class AcceptedPlan(_Model):
    round: NonNegativeInt
    plan: Plan
    val_ppl: float = Field(gt=0)
    weight_bytes: PositiveInt


class Usage(_Model):
    calls: NonNegativeInt = 0
    tokens: NonNegativeInt = 0
    seconds: NonNegativeFloat = 0.0
    cost_usd: NonNegativeFloat = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(calls=self.calls + other.calls, tokens=self.tokens + other.tokens,
                     seconds=self.seconds + other.seconds, cost_usd=self.cost_usd + other.cost_usd)


class StepRecord(_Model):
    """One line of the decision log (JSONL): what the agent chose and why, and what code made of it."""

    round: NonNegativeInt
    action: Action | None  # None when the reply could not be read (logged as INVALID)
    rationale: str
    reply: AgentReply | None = None
    check: CheckResult | None = None
    candidates: list[CandidateResult] = []
    gate: GateResult | None = None
    # LEFT: a sweep round where the agent left its layer as it is (no candidates)
    outcome: Literal["ACCEPT", "ROLLBACK", "INVALID", "LEFT"] | None = None  # None for probe, rollback and stop
    strategies_used: dict[Strategy, NonNegativeInt] = {}
    llm_tokens: NonNegativeInt = 0
    llm_seconds: NonNegativeFloat = 0.0
    llm_cost_usd: NonNegativeFloat | None = None  # API backend only
    gpu_seconds: NonNegativeFloat = 0.0


class DeviceWarning(_Model):
    kind: Literal["out_of_reach", "already_met", "below_flap"]
    message: str
    closest_plan: Plan | None = None
    memory_gap_gb: NonNegativeFloat = 0.0
    speed_gap_tok_s: NonNegativeFloat = 0.0


# ---------------------------------------------------------------------------------------------------- report


class MemoryReport(_Model):
    machine: Literal["m4", "t4"]
    params: PositiveInt
    file_size_mb: float = Field(gt=0)
    peak_memory_mb: float = Field(gt=0)
    prefill_tok_s: float = Field(gt=0)
    decode_tok_s: float = Field(gt=0)


class FinalScores(_Model):
    ppl_test: float = Field(gt=0)
    tasks: dict[str, float] = {}  # accuracy per multiple-choice question set
    mmlu: float | None = None
    memory: list[MemoryReport] = []

    @property
    def multiple_choice_mean(self) -> float | None:
        return sum(self.tasks.values()) / len(self.tasks) if self.tasks else None


class MethodResult(_Model):
    """One compared model in the report: the agent's, Baseline: FLAP, the unpruned reference or a backup."""

    name: str
    plan: Plan | None = None
    weight_bytes: PositiveInt
    scores: FinalScores | None = None


class DeviceReport(_Model):
    """Fits or not (weights against the memory limit), the key/value cache beside it at a stated text length (not
    part of the limit), the closest plan when the limit is not met, and, for a device whose speed was checked
    against measurements (the M4 and the T4), the estimated speed with the per-layer cost. For other devices (a
    phone) the report gives memory fit only (G1, A8), so the speed fields must stay empty."""

    target: DeviceTarget
    fits_memory: bool
    memory_gap_gb: NonNegativeFloat = 0.0  # GB of weights over the limit
    weight_bytes: PositiveInt | None = None
    kv_cache_bytes: NonNegativeInt | None = None  # at kv_context tokens, batch 1; not part of the limit
    kv_context: PositiveInt = REPORT_CONTEXT
    closest_plan: Plan | None = None  # the plan reported when the limit is not met
    speed_reported: bool
    est_decode_tok_s: float | None = None
    speed_gap_tok_s: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def _memory_fit_only(self) -> DeviceReport:
        if self.speed_reported and speed_machine(self.target) is None:
            raise ValueError(f"speed is reported only for the M4 and the T4, not {self.target.name!r}: give memory "
                             "fit only")
        if not self.speed_reported and (self.est_decode_tok_s is not None or self.speed_gap_tok_s is not None):
            raise ValueError("speed is not reported for this device: give memory fit only")
        return self


class Report(_Model):
    run: RunConfig
    baseline: Baseline
    final_plan: Plan
    methods: list[MethodResult]  # the agent's model, Baseline: FLAP, the unpruned model, any backups
    device: DeviceReport
    warnings: list[DeviceWarning] = []
    strategy_counts: dict[Strategy, dict[Literal["proposed", "accepted"], NonNegativeInt]] = {}
    log: list[StepRecord] = []
    usage: Usage = Usage()

    @model_validator(mode="after")
    def _fair_test(self) -> Report:
        if self.baseline.calibration != self.run.calibration:
            raise ValueError("Baseline: FLAP must use the run's calibration set (one set for every method)")
        if abs(self.baseline.memory_limit_gb - self.run.device.memory_limit_gb) > 1e-12:
            raise ValueError("Baseline: FLAP must be cut to the run's memory limit")
        return self


# ------------------------------------------------------------------------------------------- graph state (§4.3)


def merge_sensitivity(a: SensitivityTable | None, b: SensitivityTable | None) -> SensitivityTable | None:
    """Reducer for parallel quick tests: rows from `b` replace or add to those in `a`."""
    if a is None or b is None:
        return a or b
    return a.model_copy(update={"rows": {**a.rows, **b.rows}})


def add_usage(a: Usage | None, b: Usage | None) -> Usage:
    return (a or Usage()) + (b or Usage())


class PruneState(TypedDict, total=False):
    """LangGraph state (§4.3). `plan` is the only thing pruning depends on."""

    run: RunConfig
    baseline: Baseline
    sensitivity: Annotated[SensitivityTable, merge_sensitivity]
    plan: Plan
    accepted: list[AcceptedPlan]
    candidates: Annotated[list[CandidateResult], operator.add]
    action: AgentReply | None
    check: CheckResult | None
    gate: GateResult | None
    history: Annotated[list[StepRecord], operator.add]
    warnings: Annotated[list[DeviceWarning], operator.add]
    device_report: DeviceReport | None
    usage: Annotated[Usage, add_usage]
