"""Agent graph (ARCHITECTURE.md §4.1, task 2.4): the LangGraph loop from scores to the final evaluation.

    load -> feasibility -> baseline -> plan -> validate -> evaluate (<= 3, in parallel) -> gate
         -> accept | rollback -> stop_check -> plan ... -> final

- load: the sensitivity table (quick tests, plus the measured per-layer damage when it exists) and the dense plan.
- feasibility: early warning, before any agent call, when the memory limit is out of reach (below the smallest
  model the rules allow) or already met by the unpruned model; either way the run goes straight to `final`.
- baseline: once per run, Baseline: FLAP cut to the memory limit and its perplexity on the gate's windows (shown
  to the agent), and the unpruned model's perplexity on the same windows (the first accepted plan).
- plan: one agent call. The prompt comes from `Deps.build_prompt` (a placeholder until the instructions of task
  2.9); after an invalid reply, the validator's reasons go back to the agent in the next prompt.
- validate: `validator.check_reply`; invalid replies count towards `Rules.max_invalid_in_row`.
- evaluate: one parallel task (`Send`) per candidate; perplexity from the injected `Deps.evaluate`.
- gate: among candidates within the per-step limit (`Rules.step_limit`), the lowest perplexity, ties broken by
  estimated speed; none within the limit is a ROLLBACK: the plan stays the last accepted one.
- accept / rollback: adopt the chosen candidate, or keep (gate) or restore (agent's rollback action) an
  accepted plan.
- stop_check: memory limit met; `max_invalid_in_row` invalid replies in a row; `max_rounds_without_accept`
  rounds without an accepted step; `max_rounds` rounds; `RunConfig.max_agent_calls` calls.
- final: final warnings (limit not met, or worse than Baseline: FLAP on the gate's windows) and the optional
  final evaluation hook.

A round is one agent reply, whatever its outcome; every round writes one `StepRecord` to the decision log.
Memory is weight memory only (`estimate.weight_bytes`), the measure Baseline: FLAP is cut to.

Everything that needs the model or the GPU is injected (`Deps`), so the graph runs in tests with a fake agent
backend and a fake evaluator. `flap_baseline` and `ppl_evaluator` wrap the real functions for a real run.
Checkpoints and rollback storage (tasks 2.7, 2.8) are the in-memory `MemoryStore`: a placeholder to replace.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import ValidationError

from pruneagent import agent_client as AC
from pruneagent import agent_round
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
    DeviceWarning,
    GateResult,
    Plan,
    PruneState,
    Rules,
    RunConfig,
    SensitivityTable,
    StepRecord,
    StepSummary,
    Usage,
)

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


@dataclass
class MemoryStore:
    """PLACEHOLDER for tasks 2.7 (rollback storage) and 2.8 (checkpoints and decision log): keeps everything in
    memory. A replacement needs the same four methods."""

    log: list[StepRecord] = field(default_factory=list)  # the decision log, one record per round
    checkpoints: list[dict] = field(default_factory=list)
    saved: dict[int, AcceptedPlan] = field(default_factory=dict)  # accepted plans by index

    def append_log(self, record: StepRecord) -> None:
        self.log.append(record)

    def checkpoint(self, state: dict) -> None:
        self.checkpoints.append({k: state.get(k) for k in ("round", "plan", "accepted", "usage", "stop_reason")})

    def save_accepted(self, index: int, accepted: AcceptedPlan) -> None:
        self.saved[index] = accepted

    def restore(self, index: int) -> AcceptedPlan:
        return self.saved[index]


@dataclass
class Deps:
    """Everything the graph needs from outside. Only `config`, `agent`, `evaluate`, `baseline` and
    `load_sensitivity` are required."""

    config: dict  # the unpruned model's config (shape only)
    agent: Agent
    evaluate: Callable[[Plan], float]  # perplexity of a plan on the gate's windows (task 2.5 replaces it)
    baseline: Callable[[RunConfig], Baseline]  # Baseline: FLAP cut to the run's memory limit, with val_ppl_subset
    load_sensitivity: Callable[[], SensitivityTable]
    build_prompt: Callable[[AgentRequest, str | None], str] = placeholder_prompt
    probe: Callable[[Plan, list[Choice]], SensitivityTable] | None = None  # the agent's probe action
    final_eval: Callable[[dict], dict] | None = None  # test split, tasks, measured speed (phase 4)
    store: MemoryStore = field(default_factory=MemoryStore)


# ------------------------------------------------------------------------------------------- plan helpers


def smallest_plan(config, rules: Rules) -> Plan:
    """The smallest model the rules allow: every layer at its minimum MLP width and key/value groups, and
    `max_drops_per_run` layers dropped (embeddings unchanged). Unstructured sparsity frees no memory."""
    n_layers, width, _ = P.shape(config)
    min_mlp = math.ceil(width * rules.min_mlp_fraction)
    plan = P.dense_plan(config)
    for i in range(n_layers):
        while True:
            try:
                nxt = P.step(plan, i, "narrow_mlp", config)
            except P.PlanError:
                break
            if nxt[i]["mlp_keep"] < min_mlp:
                break
            plan = nxt
        plan[i]["groups_keep"] = min(rules.min_groups, plan[i]["groups_keep"])
    for i in range(min(rules.max_drops_per_run, n_layers - 1)):
        plan[i]["dropped"] = True
    return plan


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
    kept = sum(not p["dropped"] for p in plan.values())
    return 1 / (1 / est.est_decode_tok_s + layer_cost_s * kept)


def _outcome_text(rec: StepRecord, step_limit: float) -> str:
    if rec.outcome == "INVALID":
        return "INVALID: " + "; ".join(rec.check.reasons if rec.check else ["unreadable reply"])
    if rec.outcome == "ACCEPT":
        c = next(c for c in rec.candidates if c.candidate_id == rec.gate.chosen_candidate)
        return f"ACCEPT candidate {c.candidate_id + 1}, perplexity {c.val_ppl:.2f} ({c.step_rise:+.1%})"
    if rec.outcome == "ROLLBACK":
        best = min(c.step_rise for c in rec.candidates)
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
        run = state["run"]
        limit = run.limit_bytes
        dense = E.weight_bytes(cfg, P.dense_plan(cfg))
        if dense <= limit:
            w = DeviceWarning(kind="already_met", message=(
                f"The unpruned model's weights ({dense / GB:.3f} GB) already fit the memory limit "
                f"({run.device.memory_limit_gb} GB): no pruning is needed."))
            return {"warnings": [w], "stop_reason": "memory limit already met by the unpruned model"}
        small = smallest_plan(cfg, run.rules)
        least = E.weight_bytes(cfg, small)
        if least > limit:
            gap = (least - limit) / GB
            w = DeviceWarning(kind="out_of_reach", closest_plan=small, memory_gap_gb=gap, message=(
                f"The memory limit ({run.device.memory_limit_gb} GB) is out of reach: the smallest model the "
                f"pruning rules allow still has {least / GB:.3f} GB of weights, {gap:.3f} GB over the limit."))
            return {"warnings": [w], "stop_reason": "memory limit out of reach"}
        return {}

    def baseline(state: GraphState) -> dict:
        run = state["run"]
        base = deps.baseline(run)
        if base.val_ppl_subset is None:
            raise ValueError("Baseline: FLAP needs its perplexity on the gate's windows (val_ppl_subset)")
        dense = P.dense_plan(cfg)
        ppl = deps.evaluate(dense)
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
        except AC.CapReached as e:
            return {"stop_reason": f"agent call cap reached ({e})"}
        except AC.MalformedReply as e:
            call = Usage(calls=1, seconds=round(time.perf_counter() - start, 2))
            return {"round": n, "raw_reply": None, "read_error": str(e), "call": call, "usage": call}
        # UsageLimitReached and other agent errors propagate: pausing and resuming is task 2.12.
        u = reply.usage
        call = Usage(calls=0 if reply.cached else 1, tokens=u.total_tokens or 0, seconds=reply.seconds,
                     cost_usd=u.cost_usd or 0.0)
        return {"round": n, "raw_reply": reply.data, "read_error": None, "call": call, "usage": call}

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
        return [Send("evaluate", {"job": {**j, "last_ppl": last}, "run": state["run"]}) for j in jobs]

    def evaluate(state: GraphState) -> dict:
        job, run = state["job"], state["run"]
        start = time.perf_counter()
        ppl = deps.evaluate(job["plan"])
        seconds = time.perf_counter() - start
        est = E.estimate(cfg, job["plan"], run.device)
        res = CandidateResult(candidate_id=job["candidate_id"], choices=job["choices"], rationale=job["rationale"],
                              plan=job["plan"], estimate=est,
                              rank_decode_tok_s=_rank_speed(est, job["plan"], run.layer_cost_s),
                              val_ppl=ppl, step_rise=ppl / job["last_ppl"] - 1)
        return {"candidates": [res], "eval_seconds": [seconds]}

    def this_round(state: GraphState) -> tuple[list[CandidateResult], float]:
        k = state["round_start"]
        results = sorted(state["candidates"][k:], key=lambda c: c.candidate_id)
        return results, sum(state["eval_seconds"][-len(results):]) if results else 0.0

    def gate(state: GraphState) -> dict:
        results, _ = this_round(state)
        limit = state["run"].rules.step_limit
        passing = [c for c in results if c.step_rise <= limit + 1e-12]
        if not passing:
            return {"gate": GateResult(decision="ROLLBACK")}
        best = min(passing, key=lambda c: (c.val_ppl, -(c.rank_decode_tok_s or 0.0)))
        return {"gate": GateResult(decision="ACCEPT", chosen_candidate=best.candidate_id)}

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
        return {"stop_reason": reason}

    def final(state: GraphState) -> dict:
        run = state["run"]
        out: dict = {}
        reply = state.get("action")
        if reply is not None and reply.action == "stop" and not state.get("stop_reason"):
            out.update(record(state, None, check=state["check"], reply=reply))
            out["stop_reason"] = "the agent stopped"
        searched = "baseline" in state
        warnings = []
        if searched:
            plan, last = state["plan"], state["accepted"][-1]
            size = E.weight_bytes(cfg, plan)
            if size > run.limit_bytes:
                gap = (size - run.limit_bytes) / GB
                warnings.append(DeviceWarning(kind="out_of_reach", closest_plan=plan, memory_gap_gb=gap, message=(
                    f"The search ended without reaching the memory limit: the closest plan has "
                    f"{size / GB:.3f} GB of weights, {gap:.3f} GB over the limit.")))
            elif last.val_ppl > state["baseline"].val_ppl_subset:
                warnings.append(DeviceWarning(kind="below_flap", closest_plan=plan, message=(
                    f"The result fits but keeps less quality than Baseline: FLAP on the gate's windows "
                    f"(perplexity {last.val_ppl:.2f} against {state['baseline'].val_ppl_subset:.2f}).")))
            if run.device.min_decode_tok_s is not None:
                chk = E.check(run.device, cfg, plan)
                for w in warnings:
                    w.speed_gap_tok_s = chk.speed_gap_tok_s
        if warnings:
            out["warnings"] = warnings
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
    g.add_edge(START, "load")
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


def run(deps: Deps, run_cfg: RunConfig) -> dict:
    """One run from start to final; returns the final state."""
    steps_per_round = 6  # plan, validate, evaluate, gate, accept/rollback, stop_check
    limit = steps_per_round * (run_cfg.rules.max_rounds + 2) + 20
    return build_graph(deps).invoke({"run": run_cfg}, {"recursion_limit": limit})


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
                  n_windows: int = 32) -> Callable[[Plan], float]:
    """`Deps.evaluate` for a real run: prune a copy of the dense model and measure the gate's perplexity. Task 2.5
    replaces it with the version that reuses the unchanged early layers' output."""

    def evaluate(plan: Plan) -> float:
        from pruneagent import evaluate as ev
        from pruneagent import pruner

        model = pruner.prune(dense, plan, scores)
        try:
            return ev.subset_perplexity(model, ids, n_windows, seqlen, device).ppl
        finally:
            del model

    return evaluate

