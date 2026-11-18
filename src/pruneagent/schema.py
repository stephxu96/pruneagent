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

from pruneagent.estimate import DeviceTarget, Estimate
from pruneagent.plan import STRATEGIES, LayerPlan

Strategy = Literal["narrow_mlp", "remove_attention_group", "drop_layer", "wanda_sparsity"]
assert set(Strategy.__args__) == set(STRATEGIES)  # one menu (plan.STRATEGIES)

Plan = dict[int, LayerPlan]
Action = Literal["probe", "apply", "start_from_flap", "rollback", "stop"]


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
    gate_windows: PositiveInt = 32  # validation windows used by the gate
    layer_cost_s: NonNegativeFloat = 0.0  # per kept layer, added to the per-token time when ranking plans (G1)
    agent_backend: Literal["codex", "api"] = "codex"
    agent_provider: str | None = None
    agent_model: str | None = None
    agent_reasoning_effort: str | None = None
    max_agent_calls: PositiveInt = 60
    max_cost_usd: NonNegativeFloat | None = None  # API backend only; defaults to $5 there
    seed: int = 0
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    gpu: str | None = None

    @model_validator(mode="after")
    def _cost_cap(self) -> RunConfig:
        if self.agent_backend == "api" and self.max_cost_usd is None:
            self.max_cost_usd = 5.0
        if self.agent_backend == "codex" and self.max_cost_usd is not None:
            raise ValueError("max_cost_usd applies to the API backend only (Codex has no per-call charge)")
        return self

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


# -------------------------------------------------------------------------------------------- agent in and out


class Choice(_Model):
    """One choice: a strategy on a layer. Code applies one fixed step of it; there is no amount."""

    layer: NonNegativeInt
    strategy: Strategy


class Candidate(_Model):
    """One candidate plan change: choices applied together, with the agent's reason."""

    choices: list[Choice]
    rationale: str = ""


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

    @field_validator("candidates", mode="before")
    @classmethod
    def _bare_lists(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [{"choices": c} if isinstance(c, list) else c for c in v]
        return v


class StepSummary(_Model):
    """One earlier round as the agent sees it."""

    round: NonNegativeInt
    action: Action
    rationale: str
    outcome: str  # e.g. "ACCEPT candidate 2, perplexity 15.10 (+3.2%)" or "INVALID: ..."


class AgentRequest(_Model):
    """What the agent sees (§4.5). Memory in bytes, speed in tokens/s."""

    objective: str
    device: DeviceTarget
    strategy_menu: dict[Strategy, str]  # neutral, equal-detail description of one step of each
    rules: Rules
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
    action: Action
    rationale: str
    reply: AgentReply | None = None
    check: CheckResult | None = None
    candidates: list[CandidateResult] = []
    gate: GateResult | None = None
    outcome: Literal["ACCEPT", "ROLLBACK", "INVALID"] | None = None  # None for probe, rollback and stop
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
    """Fits or not, and, for a device whose speed was checked against measurements (the M4 and the T4), the
    estimated speed. For other devices (a phone) the report gives memory fit only (G1, A8), so the speed fields
    must stay empty."""

    target: DeviceTarget
    fits_memory: bool
    memory_gap_gb: NonNegativeFloat = 0.0
    speed_reported: bool
    est_decode_tok_s: float | None = None
    speed_gap_tok_s: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def _memory_fit_only(self) -> DeviceReport:
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
    usage: Annotated[Usage, add_usage]
