"""Agent graph (ARCHITECTURE.md §4.1, task 2.4): the LangGraph loop from scores to the final evaluation.

    load -> feasibility -> baseline -> plan -> validate -> evaluate (<= 3) -> gate
         -> accept | rollback -> stop_check -> plan ... -> final

- load: the sensitivity table (quick tests, plus the measured per-layer damage when it exists) and the dense plan.
- feasibility: early warning, before any agent call, when the memory limit is out of reach (below the smallest
  model the rules allow) or already met by the unpruned model; either way the run goes straight to `final`
  (`device.feasibility`, task 2.11).
- baseline: once per run, Baseline: FLAP cut to the memory limit and its perplexity on the gate's windows (shown
  to the agent), and the unpruned model's perplexity on the same windows (the first accepted plan).
- plan: one agent call. The prompt comes from `Deps.build_prompt` (a placeholder until the instructions of task
  2.9); after an invalid reply, the validator's reasons go back to the agent in the next prompt.
- validate: `validator.check_reply`; invalid replies count towards `Rules.max_invalid_in_row`.
- evaluate: one task (`Send`) per candidate; perplexity from the injected `Deps.evaluate`. The real evaluator
  (`candidate_eval.CandidateEvaluator`, task 2.5) also gets the last accepted plan, skips the layers before the
  candidate's first change, queues the tasks so they run one after another on one GPU, and reports its own GPU
  seconds; a plain function plan -> perplexity (the fakes in tests) is timed here instead.
- gate (`choose`, task 2.6): the per-step limit is the only bar (`Rules.step_limit`, 15%: perplexity at most 15%
  above the last accepted plan's; no allowance against Baseline: FLAP). Among candidates within it, the one with
  the smallest perplexity rise per MB of weight memory freed wins (a candidate that frees nothing cannot win); an
  exact tie goes to the faster plan by the speed estimate with the per-layer cost, then to the earlier candidate.
  No winner is a ROLLBACK: the plan stays the last accepted one.
- accept / rollback: adopt the chosen candidate, or keep (gate) or restore (agent's rollback action) an
  accepted plan.
- stop_check: memory limit met; `max_invalid_in_row` invalid replies in a row; `max_rounds_without_accept`
  rounds without an accepted step; `max_rounds` rounds; `RunConfig.max_agent_calls` calls.
- final: final warnings (limit not met, worse than Baseline: FLAP on the gate's windows, or short of a minimum
  speed), the device report (`device.final_warnings`, `device.report`) and the optional final evaluation hook.

A round is one agent reply, whatever its outcome; every round writes one `StepRecord` to the decision log.
Memory is weight memory only (`estimate.weight_bytes`), the measure Baseline: FLAP is cut to.

Everything that needs the model or the GPU is injected (`Deps`), so the graph runs in tests with a fake agent
backend and a fake evaluator. `flap_baseline` and `ppl_evaluator` (the candidate evaluator of task 2.5) wrap the
real functions for a real run.
Checkpoints, accepted plans (rollback targets) and the decision log go to `Deps.store`: `runstore.RunStore` on
disk for a real run (tasks 2.7, 2.8), or `MemoryStore` in tests. The state is checkpointed at the end of every
round; `resume` continues a run from its last checkpoint, replaying an unfinished round (its agent call comes
from the agent cache).

Usage limits (task 2.12, FR15): before a run (and a resume), `Deps.agent.estimate_run` says whether the calls
left fit the remaining Codex limits, and `Deps.notify` shows it (a warning when they do not). A reply whose
limit reading reached 90% finishes its round; the run then pauses right after that round's checkpoint. A call
refused or failed on a limit (or the API cost cap) pauses the run at once; the unfinished round is replayed on
resume. A paused run exits cleanly: `run` and `resume` return {"paused": reason, "resets_at": time, ...}, the
store says `paused: <reason>; resets at <time>`, and `pruneagent resume` continues after the reset.
"""

from __future__ import annotations

import importlib
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import ValidationError

from pruneagent import agent_client as AC
from pruneagent import agent_round, runstore
from pruneagent import device as D
from pruneagent import estimate as E
from pruneagent import plan as P
from pruneagent import validator as V
from pruneagent.schema import (
    AcceptedPlan,
    AgentReply,
    AgentRequest,
    Baseline,
    CandidateResult,
    CheckResult,
    Choice,
    GateResult,
    Plan,
    PruneState,
    RunConfig,
    SensitivityTable,
    StepRecord,
    StepSummary,
    Usage,
)

if TYPE_CHECKING:  # the evaluator needs torch; the graph itself does not
    from pruneagent.candidate_eval import Evaluation

GB = 1e9
RECENT_STEPS = 5  # earlier rounds shown to the agent

# The reply's JSON schema, in the strict form structured output needs (every field required, nullable where
# unused). The validator, not the schema, gives the agent readable reasons.
_CHOICE = {"type": "object", "additionalProperties": False, "required": ["layer", "strategy"],
           "properties": {"layer": {"type": "integer"}, "strategy": {"type": "string", "enum": list(P.STRATEGIES)}}}
REPLY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "rationale", "candidates", "probe", "rollback_to"],
    "properties": {
        "action": {"type": "string", "enum": list(V.ACTIONS)},
        "rationale": {"type": "string"},
        "candidates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["choices", "rationale"],
            "properties": {"choices": {"type": "array", "items": _CHOICE}, "rationale": {"type": "string"}}}},
        "probe": {"type": "array", "items": _CHOICE},
        "rollback_to": {"type": ["integer", "null"]},
    },
}


# ---------------------------------------------------------------------------------------------------- state


def _concat(a: list | None, b: list | None) -> list:
    return (a or []) + (b or [])


class GraphState(PruneState, total=False):
    """`PruneState` (§4.3) plus the loop's bookkeeping."""

    round: int  # agent replies so far
    invalid_in_row: int
    since_accept: int  # rounds since the last accepted step
    feedback: str | None  # validator reasons for the next prompt
    raw_reply: dict | None  # the reply as parsed from JSON, before validation
    read_error: str | None  # why the reply could not be read
    call: Usage  # the last agent call, for its StepRecord
    round_start: int  # index in `candidates` where this round's results begin
    eval_seconds: Annotated[list[float], _concat]  # evaluation time, one entry per evaluated candidate
    job: dict  # one candidate to evaluate (the payload of a `Send`)
    stop_reason: str | None
    final: dict | None  # from `Deps.final_eval`
    resume: bool  # set by `resume`: start at the next round instead of `load`
    pause: tuple | None  # (reason, reset time): this round's call reached the pause point; pause after the round


class RunPaused(Exception):
    """Raised inside the graph to stop a run at a usage limit; `run` and `resume` turn it into a clean return."""

    def __init__(self, reason: str, resets_at=None):
        super().__init__(reason)
        self.reason, self.resets_at = reason, resets_at


# ------------------------------------------------------------------------------------------- injected parts


class Agent(Protocol):
    """`agent_client.AgentClient`, or a fake with the same `propose`."""

    def propose(self, request: AC.AgentRequest) -> AC.AgentReply: ...


def placeholder_prompt(req: AgentRequest, feedback: str | None = None) -> str:
    """PLACEHOLDER until the agent instructions of task 2.9: the request as JSON and the reply format. Swap it
    through `Deps.build_prompt`; any function (AgentRequest, feedback) -> str will do."""
    parts = [
        "You plan structured pruning of a language model. Answer with JSON only; do not run commands or read files.",
        f"Objective: {req.objective}",
        "",
        "Input (memory in bytes, speed in tokens/s; perplexity on the gate's validation windows):",
        req.model_dump_json(),
        "",
        ("Reply with one JSON object: {\"action\": \"probe\" | \"apply\" | \"start_from_flap\" | \"rollback\" | "
        "\"stop\", \"rationale\": \"...\", \"candidates\": [{\"choices\": [{\"layer\": 0, \"strategy\": "
        "\"narrow_mlp\"}], \"rationale\": \"...\"}], \"probe\": [{\"layer\": 0, \"strategy\": \"...\"}], "
        "\"rollback_to\": null}. Leave unused fields empty ([] or null)."),
    ]
    if feedback:
        parts += ["", "Your previous reply was not accepted:", feedback]
    return "\n".join(parts)


class Store(Protocol):
    """Where a run keeps its decision log, checkpoints and accepted plans: `MemoryStore` or `runstore.RunStore`."""

    def append_log(self, record: StepRecord) -> None: ...
    def checkpoint(self, state: dict) -> None: ...
    def save_accepted(self, index: int, accepted: AcceptedPlan) -> None: ...
    def restore(self, index: int) -> AcceptedPlan: ...  # also discards every later accepted plan
    def load_checkpoint(self) -> dict | None: ...
    def sync(self, state: dict) -> None: ...  # before resuming: drop what an unfinished round wrote
    def set_status(self, status: str, reason: str | None = None) -> None: ...


@dataclass
class MemoryStore:
    """The store kept in memory, for tests and one-off runs; `runstore.RunStore` keeps the same on disk."""

    log: list[StepRecord] = field(default_factory=list)  # the decision log, one record per round
    checkpoints: list[dict] = field(default_factory=list)
    saved: dict[int, AcceptedPlan] = field(default_factory=dict)  # accepted plans by index
    status: tuple[str, str | None] = ("new", None)

    def append_log(self, record: StepRecord) -> None:
        self.log.append(record)

    def checkpoint(self, state: dict) -> None:
        self.checkpoints.append({k: state[k] for k in runstore.Saved.__annotations__ if k in state})

    def save_accepted(self, index: int, accepted: AcceptedPlan) -> None:
        self.saved = {i: a for i, a in self.saved.items() if i < index}
        self.saved[index] = accepted

    def restore(self, index: int) -> AcceptedPlan:
        restored = self.saved[index]
        self.saved = {i: a for i, a in self.saved.items() if i <= index}
        return restored

    def load_checkpoint(self) -> dict | None:
        return dict(self.checkpoints[-1]) if self.checkpoints else None

    def sync(self, state: dict) -> None:
        self.log = [r for r in self.log if r.round <= (state.get("round") or 0)]
        self.saved = dict(enumerate(state.get("accepted") or []))

    def set_status(self, status: str, reason: str | None = None) -> None:
        self.status = (status, reason)


@dataclass
class Deps:
    """Everything the graph needs from outside. Only `config`, `agent`, `evaluate`, `baseline` and
    `load_sensitivity` are required."""

    config: dict  # the unpruned model's config (shape only)
    agent: Agent
    # perplexity of a plan on the gate's windows: a function plan -> float (fakes), or an evaluator with
    # `takes_accepted = True` called as (plan, last accepted plan) -> Evaluation (`ppl_evaluator`, task 2.5)
    evaluate: Callable[..., float | Evaluation]
    baseline: Callable[[RunConfig], Baseline]  # Baseline: FLAP cut to the run's memory limit, with val_ppl_subset
    load_sensitivity: Callable[[], SensitivityTable]
    build_prompt: Callable[[AgentRequest, str | None], str] = placeholder_prompt
    probe: Callable[[Plan, list[Choice]], SensitivityTable] | None = None  # the agent's probe action
    final_eval: Callable[[dict], dict] | None = None  # test split, tasks, measured speed (phase 4)
    store: Store = field(default_factory=MemoryStore)
    notify: Callable[[str], None] = field(default=lambda msg: print(msg, file=sys.stderr))  # limit messages


# ------------------------------------------------------------------------------------------- plan helpers


smallest_plan = D.smallest_plan


def flap_allocation(baseline: Baseline, config) -> Plan:
    """Baseline: FLAP's allocation on the agent's grid (the `start_from_flap` candidate): each layer's MLP width
    rounded down to a multiple of 64, so it still fits the limit; at least one group and 64 neurons per layer."""
    full = P.dense_plan(config)
    for i, p in baseline.plan.items():
        mlp = max(P.MLP_ROUND, (p["mlp_keep"] // P.MLP_ROUND) * P.MLP_ROUND)
        full[i] = P.LayerPlan(mlp_keep=min(mlp, full[i]["mlp_keep"]), groups_keep=max(1, p["groups_keep"]),
                              dropped=p["dropped"], wanda_sparsity=0.0)
    return full


def _rank_speed(est: E.Estimate, plan: Plan, layer_cost_s: float) -> float | None:
    """Estimated decode speed with the per-layer cost added (G1): breaks perplexity ties."""
    if not est.est_decode_tok_s:
        return None
    return E.with_layer_cost(est.est_decode_tok_s, sum(not p["dropped"] for p in plan.values()), layer_cost_s)


def measure(evaluate: Callable, plan: Plan, accepted: Plan) -> tuple[float, float]:
    """(perplexity, GPU seconds) of `plan` from `Deps.evaluate`; `accepted` goes to evaluators that take it."""
    start = time.perf_counter()
    out = evaluate(plan, accepted) if getattr(evaluate, "takes_accepted", False) else evaluate(plan)
    if hasattr(out, "gpu_seconds"):  # a `candidate_eval.Evaluation`
        return out.ppl, out.gpu_seconds
    return float(out), time.perf_counter() - start


MB = 1e6


def rise_per_mb(step_rise: float, freed_bytes: int) -> float | None:
    """Perplexity rise per MB of weight memory freed; None (infinite: cannot win) when nothing is freed."""
    return step_rise / (freed_bytes / MB) if freed_bytes > 0 else None


def choose(results: list[CandidateResult], step_limit: float) -> GateResult:
    """The gate (§4.4, FR7). Candidates whose perplexity rises more than `step_limit` over the last accepted plan
    are rejected, and so are candidates that free no weight memory. Among the rest, the smallest perplexity rise per
    MB freed (`rise_per_mb`) wins; an exact tie goes to the higher speed estimate (`rank_decode_tok_s`; a missing
    estimate counts as slowest), then to the lower candidate id. No winner (or nothing evaluated) is a ROLLBACK."""
    passing = [c for c in results if c.step_rise <= step_limit + 1e-12 and c.rise_per_mb is not None]
    if not passing:
        return GateResult(decision="ROLLBACK")
    best = min(passing, key=lambda c: (c.rise_per_mb, -(c.rank_decode_tok_s or 0.0), c.candidate_id))
    return GateResult(decision="ACCEPT", chosen_candidate=best.candidate_id)


def _outcome_text(rec: StepRecord, step_limit: float) -> str:
    if rec.outcome == "INVALID":
        return "INVALID: " + "; ".join(rec.check.reasons if rec.check else ["unreadable reply"])
    if rec.outcome == "ACCEPT":
        c = next(c for c in rec.candidates if c.candidate_id == rec.gate.chosen_candidate)
        return (f"ACCEPT candidate {c.candidate_id + 1}, perplexity {c.val_ppl:.2f} ({c.step_rise:+.1%}), "
                f"{c.freed_bytes / MB:.1f} MB freed")
    if rec.outcome == "ROLLBACK":
        best = min(c.step_rise for c in rec.candidates)
        if best <= step_limit + 1e-12:
            return f"ROLLBACK: no candidate within {step_limit:.0%} frees memory; plan unchanged"
        return f"ROLLBACK: no candidate within {step_limit:.0%} (best {best:+.1%}); plan unchanged"
    return {"probe": "probed", "rollback": "rolled back", "stop": "stopped"}.get(rec.action or "", "")


# ------------------------------------------------------------------------------------------------- graph


def build_graph(deps: Deps, checkpointer=None):
    """The compiled graph. Invoke with {"run": RunConfig}; see `run`."""
    cfg = deps.config
    store = deps.store

    def record(state: GraphState, outcome=None, check: CheckResult | None = None, reply: AgentReply | None = None,
               candidates: list[CandidateResult] | None = None, gate: GateResult | None = None,
               gpu_seconds: float = 0.0) -> dict:
        raw = state.get("raw_reply") or {}
        action = raw.get("action") if raw.get("action") in V.ACTIONS else None
        rationale = raw.get("rationale") if isinstance(raw.get("rationale"), str) else ""
        used = Counter(c.strategy for cand in (reply.candidates if reply else []) for c in cand.choices)
        call = state.get("call") or Usage()
        run = state["run"]
        rec = StepRecord(round=state["round"], action=action, rationale=rationale, reply=reply, check=check,
                         candidates=candidates or [], gate=gate, outcome=outcome, strategies_used=dict(used),
                         llm_tokens=call.tokens, llm_seconds=call.seconds,
                         llm_cost_usd=call.cost_usd if run.agent_backend == "api" else None,
                         gpu_seconds=gpu_seconds)
        store.append_log(rec)
        return {"history": [rec]}

    # ---------------------------------------------------------------------------------------------- nodes

    def load(state: GraphState) -> dict:
        return {"sensitivity": deps.load_sensitivity(), "plan": P.dense_plan(cfg), "round": 0, "invalid_in_row": 0,
                "since_accept": 0, "feedback": None, "stop_reason": None, "candidates": [], "history": [],
                "warnings": [], "usage": Usage()}

    def feasibility(state: GraphState) -> dict:
        warnings, reason = D.feasibility(cfg, state["run"])
        return {"warnings": warnings, **({"stop_reason": reason} if reason else {})}

    def baseline(state: GraphState) -> dict:
        run = state["run"]
        base = deps.baseline(run)
        if base.val_ppl_subset is None:
            raise ValueError("Baseline: FLAP needs its perplexity on the gate's windows (val_ppl_subset)")
        dense = P.dense_plan(cfg)
        ppl, _ = measure(deps.evaluate, dense, dense)
        first = AcceptedPlan(round=0, plan=dense, val_ppl=ppl, weight_bytes=E.weight_bytes(cfg, dense))
        store.save_accepted(0, first)
        return {"baseline": base, "accepted": [first], "eval_seconds": []}

    def agent_request(state: GraphState) -> AgentRequest:
        run, plan, last = state["run"], state["plan"], state["accepted"][-1]
        est, dense_est = E.estimate(cfg, plan, run.device), E.estimate(cfg, P.dense_plan(cfg), run.device)
        recent = [StepSummary(round=r.round, action=r.action, rationale=r.rationale,
                              outcome=_outcome_text(r, run.rules.step_limit))
                  for r in state.get("history", [])[-RECENT_STEPS:]]
        return AgentRequest(objective=agent_round.OBJECTIVE, device=run.device,
                            strategy_menu=agent_round.strategy_menu(cfg), rules=run.rules,
                            sensitivity=state["sensitivity"], plan=plan,
                            dense_weight_bytes=dense_est.weight_bytes, current_weight_bytes=est.weight_bytes,
                            limit_bytes=run.limit_bytes, est_decode_tok_s=est.est_decode_tok_s,
                            dense_est_decode_tok_s=dense_est.est_decode_tok_s, current_ppl=last.val_ppl,
                            last_accepted_ppl=last.val_ppl, baseline_ppl=state["baseline"].val_ppl_subset,
                            recent=recent, round=state["round"] + 1)

    def plan_node(state: GraphState) -> dict:
        run = state["run"]
        if state["usage"].calls >= run.max_agent_calls:
            return {"stop_reason": f"agent call cap reached ({run.max_agent_calls} calls)"}
        n = state["round"] + 1
        prompt = deps.build_prompt(agent_request(state), state.get("feedback"))
        start = time.perf_counter()
        try:
            reply = deps.agent.propose(AC.AgentRequest(prompt, REPLY_SCHEMA, tag=f"round {n}"))
        except AC.UsageLimitReached as e:  # also the API cost cap: pause; this round is replayed on resume
            raise RunPaused(str(e), e.resets_at) from e
        except AC.CapReached as e:
            return {"stop_reason": f"agent call cap reached ({e})"}
        except AC.MalformedReply as e:
            call = Usage(calls=1, seconds=round(time.perf_counter() - start, 2))
            return {"round": n, "raw_reply": None, "read_error": str(e), "call": call, "usage": call}
        # other agent errors (login, a call failed twice) propagate and leave the run paused with the error
        u = reply.usage
        call = Usage(calls=0 if reply.cached else 1, tokens=u.total_tokens or 0, seconds=reply.seconds,
                     cost_usd=u.cost_usd or 0.0)
        pause = None
        if getattr(reply, "pause_reason", None):
            limits = getattr(reply, "limits", None)
            pause = (reply.pause_reason, limits.resets_at() if limits else None)
        return {"round": n, "raw_reply": reply.data, "read_error": None, "call": call, "usage": call,
                "pause": pause}

    def validate(state: GraphState) -> dict:
        run = state["run"]
        raw = state.get("raw_reply")
        typed = None
        if raw is None:
            check = CheckResult(ok=False, reasons=[
                f"The reply could not be read ({state.get('read_error')}); reply with one JSON object only."])
        else:
            ctx = V.Context(cfg, state["plan"], run.limit_bytes, run.rules,
                            tried=tuple(c.plan for c in state.get("candidates", [])),
                            n_accepted=len(state["accepted"]))
            check = V.check_reply(raw, ctx)
            if check.ok and raw.get("action") == "start_from_flap":
                flap = flap_allocation(state["baseline"], cfg)
                if any(c.plan == flap for c in state.get("candidates", [])):
                    check = CheckResult(ok=False, reasons=[
                        ("Baseline: FLAP's allocation was already evaluated in an earlier round; choose another "
                        "action.")])
            if check.ok:
                try:
                    typed = AgentReply.model_validate(raw)
                except ValidationError as e:
                    check = CheckResult(ok=False, reasons=[f"The reply does not have the expected form: {e}"])
        if not check.ok:
            return {"check": check, "action": None, "gate": None, "feedback": check.feedback(),
                    "invalid_in_row": state["invalid_in_row"] + 1, "since_accept": state["since_accept"] + 1,
                    **record(state, "INVALID", check=check)}
        return {"check": check, "action": typed, "gate": None, "feedback": None, "invalid_in_row": 0,
                "round_start": len(state.get("candidates", []))}

    def route_after_validate(state: GraphState):
        reply = state.get("action")
        if reply is None:
            return "stop_check"
        if reply.action in ("probe", "rollback"):
            return reply.action
        if reply.action == "stop":
            return "final"
        last = state["accepted"][-1].val_ppl
        if reply.action == "start_from_flap":
            jobs = [{"candidate_id": 0, "choices": [], "rationale": reply.rationale,
                     "plan": flap_allocation(state["baseline"], cfg)}]
        else:
            jobs = [{"candidate_id": c.index, "choices": reply.candidates[c.index].choices,
                     "rationale": reply.candidates[c.index].rationale, "plan": c.plan}
                    for c in state["check"].candidates]
        acc = state["accepted"][-1]
        return [Send("evaluate", {"job": {**j, "last_ppl": last, "accepted_plan": acc.plan,
                                          "accepted_bytes": acc.weight_bytes}, "run": state["run"]})
                for j in jobs]

    def evaluate(state: GraphState) -> dict:
        job, run = state["job"], state["run"]
        ppl, seconds = measure(deps.evaluate, job["plan"], job["accepted_plan"])
        rise = ppl / job["last_ppl"] - 1
        est = E.estimate(cfg, job["plan"], run.device)
        freed = job["accepted_bytes"] - est.weight_bytes
        res = CandidateResult(candidate_id=job["candidate_id"], choices=job["choices"], rationale=job["rationale"],
                              plan=job["plan"], estimate=est,
                              rank_decode_tok_s=_rank_speed(est, job["plan"], run.layer_cost),
                              val_ppl=ppl, step_rise=rise, freed_bytes=freed, rise_per_mb=rise_per_mb(rise, freed))
        return {"candidates": [res], "eval_seconds": [seconds]}

    def this_round(state: GraphState) -> tuple[list[CandidateResult], float]:
        k = state["round_start"]
        results = sorted(state["candidates"][k:], key=lambda c: c.candidate_id)
        return results, sum(state["eval_seconds"][-len(results):]) if results else 0.0

    def gate(state: GraphState) -> dict:
        results, _ = this_round(state)
        return {"gate": choose(results, state["run"].rules.step_limit)}

    def accept(state: GraphState) -> dict:
        results, gpu = this_round(state)
        g = state["gate"]
        chosen = next(c for c in results if c.candidate_id == g.chosen_candidate)
        acc = AcceptedPlan(round=state["round"], plan=chosen.plan, val_ppl=chosen.val_ppl,
                           weight_bytes=chosen.estimate.weight_bytes)
        accepted = [*state["accepted"], acc]
        store.save_accepted(len(accepted) - 1, acc)
        return {"plan": chosen.plan, "accepted": accepted, "since_accept": 0,
                **record(state, "ACCEPT", check=state["check"], reply=state["action"], candidates=results, gate=g,
                         gpu_seconds=gpu)}

    def rollback(state: GraphState) -> dict:
        reply = state["action"]
        if reply.action == "rollback":  # the agent's action: restore an earlier accepted plan
            index = reply.rollback_to if reply.rollback_to is not None else len(state["accepted"]) - 2
            restored = store.restore(index)
            return {"plan": restored.plan, "accepted": state["accepted"][:index + 1],
                    "since_accept": state["since_accept"] + 1,
                    **record(state, None, check=state["check"], reply=reply)}
        # the gate's ROLLBACK: no candidate within the per-step limit; the plan stays the last accepted one
        results, gpu = this_round(state)
        return {"plan": state["accepted"][-1].plan, "since_accept": state["since_accept"] + 1,
                **record(state, "ROLLBACK", check=state["check"], reply=reply, candidates=results,
                         gate=state["gate"], gpu_seconds=gpu)}

    def probe(state: GraphState) -> dict:
        reply = state["action"]
        out: dict = {"since_accept": state["since_accept"] + 1}
        if deps.probe is None:
            out["feedback"] = ("Probing is not available in this run; the sensitivity table already has every "
                               "layer's quick tests. Choose another action.")
        else:
            out["sensitivity"] = deps.probe(state["plan"], reply.probe)
        return {**out, **record(state, None, check=state["check"], reply=reply)}

    def stop_check(state: GraphState) -> dict:
        run, rules = state["run"], state["run"].rules
        reason = None
        if E.weight_bytes(cfg, state["plan"]) <= run.limit_bytes:
            reason = "memory limit met"
        elif state["invalid_in_row"] >= rules.max_invalid_in_row:
            reason = f"{state['invalid_in_row']} invalid replies in a row"
        elif state["since_accept"] >= rules.max_rounds_without_accept:
            reason = f"{state['since_accept']} rounds without an accepted step"
        elif state["round"] >= rules.max_rounds:
            reason = f"{state['round']} rounds"
        elif state["usage"].calls >= run.max_agent_calls:
            reason = f"agent call cap reached ({run.max_agent_calls} calls)"
        store.checkpoint({**state, "stop_reason": reason})
        if state.get("pause") and not reason:  # the round is saved: stop here until the limit resets
            raise RunPaused(*state["pause"])
        return {"stop_reason": reason}

    def final(state: GraphState) -> dict:
        run = state["run"]
        out: dict = {}
        reply = state.get("action")
        if reply is not None and reply.action == "stop" and not state.get("stop_reason"):
            out.update(record(state, None, check=state["check"], reply=reply))
            out["stop_reason"] = "the agent stopped"
        searched = "baseline" in state
        plan = state["plan"]
        if searched:
            warnings = D.final_warnings(cfg, run, plan, state["accepted"][-1].val_ppl,
                                        state["baseline"].val_ppl_subset)
            if warnings:
                out["warnings"] = warnings
        else:  # stopped early: report on the early warning's closest plan (the smallest model) if it has one
            plan = next((w.closest_plan for w in state.get("warnings", []) if w.kind == "out_of_reach"
                         and w.memory_gap_gb > 0), plan)
        out["device_report"] = D.report(cfg, run, plan)
        if deps.final_eval is not None and searched:
            out["final"] = deps.final_eval({**state, **out})
        store.checkpoint({**state, **out})
        return out

    # ---------------------------------------------------------------------------------------------- edges

    g = StateGraph(GraphState)
    for name, fn in [("load", load), ("feasibility", feasibility), ("baseline", baseline), ("plan", plan_node),
                     ("validate", validate), ("evaluate", evaluate), ("gate", gate), ("accept", accept),
                     ("rollback", rollback), ("probe", probe), ("stop_check", stop_check), ("final", final)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, lambda s: ("final" if s.get("stop_reason") else "plan") if s.get("resume")
                            else "load", ["load", "plan", "final"])
    g.add_edge("load", "feasibility")
    g.add_conditional_edges("feasibility", lambda s: "final" if s.get("stop_reason") else "baseline",
                            ["baseline", "final"])
    g.add_edge("baseline", "plan")
    g.add_conditional_edges("plan", lambda s: "final" if s.get("stop_reason") else "validate", ["validate", "final"])
    g.add_conditional_edges("validate", route_after_validate,
                            ["evaluate", "probe", "rollback", "final", "stop_check"])
    g.add_edge("evaluate", "gate")
    g.add_conditional_edges("gate", lambda s: "accept" if s["gate"].decision == "ACCEPT" else "rollback",
                            ["accept", "rollback"])
    g.add_edge("accept", "stop_check")
    g.add_edge("rollback", "stop_check")
    g.add_edge("probe", "stop_check")
    g.add_conditional_edges("stop_check", lambda s: "final" if s.get("stop_reason") else "plan", ["plan", "final"])
    g.add_edge("final", END)
    return g.compile(checkpointer=checkpointer)


def preflight(deps: Deps, calls: int):
    """The pre-run usage-limit estimate for `calls` more agent calls, shown with `deps.notify`; None when the
    agent has no plan limits (API backend, fakes)."""
    estimate = getattr(deps.agent, "estimate_run", None)
    est = estimate(calls) if estimate is not None and calls > 0 else None
    if est is not None:
        deps.notify(est.message())
    return est


def _invoke(deps: Deps, start: dict, run_cfg: RunConfig) -> dict:
    """Invoke the graph, keeping the store's status: running, then finished; paused at a usage limit (returns
    the last checkpoint plus {"paused": reason, "resets_at": time}); or paused on any other exception (an
    interruption, an agent error) before it propagates."""
    steps_per_round = 6  # plan, validate, evaluate, gate, accept/rollback, stop_check
    limit = steps_per_round * (run_cfg.rules.max_rounds + 2) + 20
    deps.store.set_status("running")
    try:
        state = build_graph(deps).invoke(start, {"recursion_limit": limit})
    except RunPaused as e:
        deps.store.set_status("paused", e.reason)
        saved = deps.store.load_checkpoint() or {}
        deps.notify(f"Paused: {e.reason}. Run `pruneagent resume` after the reset.")
        return {**saved, "paused": e.reason, "resets_at": e.resets_at, "round": saved.get("round", 0)}
    except BaseException as e:
        deps.store.set_status("paused", f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)
        raise
    deps.store.set_status("finished", state.get("stop_reason"))
    return state


def run(deps: Deps, run_cfg: RunConfig) -> dict:
    """One run from start to final; returns the final state (or {"paused": ...} at a usage limit)."""
    preflight(deps, run_cfg.max_agent_calls)
    return _invoke(deps, {"run": run_cfg}, run_cfg)


def resume(deps: Deps, run_cfg: RunConfig | None = None) -> dict:
    """Continue a run from the store's last checkpoint (from the start if it never finished a round). Whatever an
    unfinished round wrote is dropped first, and the round is replayed. `run_cfg` defaults to the store's."""
    store = deps.store
    state = store.load_checkpoint()
    if state is None:
        store.sync({"round": 0, "accepted": []})
        return run(deps, run_cfg or store.run_config())
    store.sync(state)
    if run_cfg is not None and run_cfg.agent_backend != state["run"].agent_backend:
        state["run"] = run_cfg  # resumed on the other backend
    preflight(deps, state["run"].max_agent_calls - state["usage"].calls)
    return _invoke(deps, {**state, "resume": True}, state["run"])


def _factory(spec: str) -> Callable:
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def start_run(run_cfg: RunConfig, deps_factory: str, root=None, run_id: str | None = None) -> dict:
    """A run saved under `root/<run-id>/` (default `runs/`). `deps_factory` is "module:function", called as
    fn(RunConfig, run_dir) -> Deps; it is recorded so that `pruneagent resume` can rebuild the same parts."""
    store = runstore.RunStore.create(run_cfg, root or runstore.DEFAULT_ROOT, run_id, deps_factory)
    deps = _factory(deps_factory)(run_cfg, store.dir)
    deps.store = store
    return run(deps, run_cfg)


def switch_backend(run_cfg: RunConfig, backend: str) -> RunConfig:
    """The run config on the other agent backend: the API backend gets its default cost cap, Codex none."""
    if backend == run_cfg.agent_backend:
        return run_cfg
    data = run_cfg.model_dump()
    data.update(agent_backend=backend, max_cost_usd=None)
    return RunConfig.model_validate(data)


def resume_run(run_id: str | None = None, root=None, backend: str | None = None) -> dict:
    """`pruneagent resume`: the named run, or the only paused one (`runstore.ResumeError` otherwise). `backend`
    continues it on the other agent backend (recorded in the run's config; the ledger shows which backend made
    each call)."""
    store = runstore.pick_run(root or runstore.DEFAULT_ROOT, run_id)
    factory = store.info().deps_factory
    if not factory:
        raise runstore.ResumeError(f"run {store.run_id} was started without a deps factory; resume it from Python "
                                   "with graph.resume(deps)")
    run_cfg = store.run_config()
    if backend is not None and backend != run_cfg.agent_backend:
        run_cfg = switch_backend(run_cfg, backend)
        store.set_config(run_cfg)
    deps = _factory(factory)(run_cfg, store.dir)
    deps.store = store
    return resume(deps, run_cfg)


# --------------------------------------------------------------------------------------- real-run adapters


def plan_of_model(model, dense_config) -> Plan:
    """The per-layer plan of a FLAP-pruned model (no layer dropped), read from its layer shapes."""
    c = P._cfg(dense_config)
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return {i: P.LayerPlan(mlp_keep=int(b.mlp.up_proj.out_features),
                           groups_keep=int(b.self_attn.k_proj.out_features // head_dim),
                           dropped=False, wanda_sparsity=0.0)
            for i, b in enumerate(model.model.layers)}


def flap_baseline(dense, stats, ids, *, seqlen: int = 2048, device: str | None = None,
                  n_windows: int = 32) -> Callable[[RunConfig], Baseline]:
    """`Deps.baseline` for a real run: `baselines.flap_at_memory_limit` and its perplexity on the gate's windows."""

    def make(run_cfg: RunConfig) -> Baseline:
        from pruneagent import baselines
        from pruneagent import evaluate as ev

        model, out = baselines.flap_at_memory_limit(dense, stats, run_cfg.limit_bytes)
        if model is None:
            raise ValueError(f"Baseline: FLAP cannot reach the memory limit: {out.note or out}")
        r = ev.subset_perplexity(model, ids, n_windows, seqlen, device)
        return Baseline(calibration=run_cfg.calibration, memory_limit_gb=run_cfg.device.memory_limit_gb,
                        ratio=out.ratio, plan=plan_of_model(model, dense.config), weight_bytes=out.weight_bytes,
                        val_ppl_subset=r.ppl)

    return make


def ppl_evaluator(dense, scores, ids, *, seqlen: int = 2048, device: str | None = None,
                  n_windows: int = 32, store_device: str = "cpu", skip: bool = True):
    """`Deps.evaluate` for a real run: `candidate_eval.CandidateEvaluator`, which rebuilds each candidate from the
    cached scores, measures the gate's perplexity reusing the unchanged early layers' saved output, and runs the
    candidates one after another on one GPU. `skip=False` runs every layer (same numbers, slower)."""
    from pruneagent.candidate_eval import CandidateEvaluator

    return CandidateEvaluator(dense, scores, ids, seqlen=seqlen, device=device, n_windows=n_windows,
                              store_device=store_device, skip=skip)
