"""Agent graph (task 2.4) with fakes only: a fake agent backend behind the real `AgentClient`, a fake evaluator
with made-up perplexities and a fake Baseline: FLAP. No model, GPU or real agent call."""

import json

import pytest

from pruneagent import agent_client as AC
from pruneagent import estimate as E
from pruneagent import graph as G
from pruneagent import plan as P
from pruneagent.estimate import DeviceTarget
from pruneagent.schema import Baseline, LayerPlan, Rules, RunConfig, SensitivityRow, SensitivityTable

# a tiny model shape: 4 layers, MLP 640 wide (one narrow_mlp step = 64 neurons), 4 key/value groups
CFG = {"num_hidden_layers": 4, "intermediate_size": 640, "num_key_value_heads": 4, "num_attention_heads": 8,
       "hidden_size": 64, "head_dim": 16, "vocab_size": 1000, "tie_word_embeddings": True}
DENSE = E.weight_bytes(CFG, P.dense_plan(CFG))  # 1,309,056 bytes
NARROW = DENSE - E.weight_bytes(CFG, P.step(P.dense_plan(CFG), 0, "narrow_mlp", CFG))  # freed by one step
LIMIT = DENSE - 3 * NARROW + 1000  # three narrow_mlp steps on different layers reach it
DENSE_PPL, FLAP_PPL = 10.0, 12.0


def run_config(limit_bytes=LIMIT, **kw):
    rules = kw.pop("rules", Rules())
    return RunConfig(model_id="tiny", device=DeviceTarget("test", memory_limit_gb=limit_bytes / 1e9,
                                                          bandwidth_gb_s=100, compute_tflops=10),
                     rules=rules, **kw)


def fake_ppl(plan) -> float:
    """Made-up perplexity: +2% per narrow_mlp step, +5% per attention group removed, +30% per dropped layer,
    +1% per sparsity step."""
    ppl = DENSE_PPL
    for p in plan.values():
        if p["dropped"]:
            ppl *= 1.30
            continue
        ppl *= 1.02 ** ((640 - p["mlp_keep"]) // 64) * 1.05 ** (4 - p["groups_keep"])
        ppl *= 1.01 ** round(p["wanda_sparsity"] * 10)
    return ppl


def fake_baseline(run: RunConfig) -> Baseline:
    plan = {i: LayerPlan(mlp_keep=600, groups_keep=4, dropped=False, wanda_sparsity=0.0) for i in range(4)}
    return Baseline(memory_limit_gb=run.device.memory_limit_gb, ratio=0.1, plan=plan,
                    weight_bytes=run.limit_bytes - 1000, val_ppl_subset=FLAP_PPL)


def table() -> SensitivityTable:  # quick tests only: the measured-damage part is absent
    return SensitivityTable(rows={i: SensitivityRow(layer=i, block_influence=0.1 * (i + 1)) for i in range(4)})


class FakeBackend:
    """Answers from `policy(n, prompt)` (n counts calls from 1): a dict (sent as JSON) or a raw string."""

    name = "fake"
    model = "fake-model"

    def __init__(self, policy):
        self.policy = policy
        self.prompts: list[str] = []

    def identity(self):
        return {"backend": self.name, "model": self.model}

    def complete(self, prompt, schema, timeout):
        self.prompts.append(prompt)
        out = self.policy(len(self.prompts), prompt)
        text = out if isinstance(out, str) else json.dumps(out)
        return AC.BackendResult(text=text, usage=AC.Usage(input_tokens=100, output_tokens=20), model=self.model)


def reply(*candidates, action="apply", **kw):
    return {"action": action, "rationale": f"{action} for a reason",
            "candidates": [{"choices": [{"layer": la, "strategy": s} for la, s in c], "rationale": "c"}
                           for c in candidates], "probe": [], "rollback_to": None, **kw}


def setup(tmp_path, policy, evaluate=fake_ppl, max_calls=60, baseline=fake_baseline, **kw):
    backend = FakeBackend(policy)
    client = AC.AgentClient(backend, cache=AC.Cache(tmp_path / "cache"), ledger=AC.Ledger(tmp_path / "use.jsonl"),
                            sleep=lambda s: None, max_calls=max_calls)
    deps = G.Deps(config=CFG, agent=client, evaluate=evaluate, baseline=baseline, load_sensitivity=table, **kw)
    return deps, backend


def outcomes(state):
    return [r.outcome for r in state["history"]]


# -------------------------------------------------------------------------------------------------- tests


def test_graph_compiles_with_every_node(tmp_path):
    deps, _ = setup(tmp_path, lambda n, p: {})
    nodes = set(G.build_graph(deps).get_graph().nodes)
    assert {"load", "feasibility", "baseline", "plan", "validate", "evaluate", "gate", "accept", "rollback",
            "probe", "stop_check", "final"} <= nodes


def test_happy_path_reaches_the_memory_limit(tmp_path):
    # each round: narrow one layer (+2%) or remove an attention group on it (+5%); the gate takes the narrowing
    deps, backend = setup(tmp_path, lambda n, p: reply([((n - 1) % 4, "narrow_mlp")],
                                                         [((n - 1) % 4, "remove_attention_group")]))
    state = G.run(deps, run_config())
    assert state["stop_reason"] == "memory limit met"
    assert outcomes(state) == ["ACCEPT"] * 3
    assert all(r.gate.chosen_candidate == 0 and len(r.candidates) == 2 for r in state["history"])
    assert E.weight_bytes(CFG, state["plan"]) <= LIMIT
    assert [p["mlp_keep"] for p in state["plan"].values()] == [576, 576, 576, 640]
    assert [a.val_ppl for a in state["accepted"]] == pytest.approx([10, 10.2, 10.404, 10.61208])
    # FLAP measured once, its perplexity shown to the agent and kept in state
    assert state["baseline"].val_ppl_subset == FLAP_PPL
    assert '"baseline_ppl":12.0' in backend.prompts[0]
    assert state["usage"].calls == 3 and state["history"][0].llm_tokens == 120
    assert state["warnings"] == []  # fits, and better than Baseline: FLAP on the gate's windows
    assert deps.store.log == state["history"] and deps.store.saved[3].plan == state["plan"]


def test_invalid_replies_then_recovery(tmp_path):
    script = {1: "I would narrow some layers.",  # no JSON at all
              2: reply([(99, "narrow_mlp")]),  # a layer that does not exist
              3: reply([(0, "narrow_mlp")])}
    deps, backend = setup(tmp_path, lambda n, p: script.get(n, reply([(n - 3, "narrow_mlp")])))
    state = G.run(deps, run_config())
    assert outcomes(state)[:3] == ["INVALID", "INVALID", "ACCEPT"]
    assert state["history"][0].action is None and state["history"][1].action == "apply"
    # the validator's reasons go back to the agent in the next prompt
    assert "could not be read" in backend.prompts[1]
    assert "layer 99 does not exist" in backend.prompts[2]
    assert "rejected" not in backend.prompts[3]  # cleared after a valid reply
    assert state["stop_reason"] == "memory limit met"


def test_three_invalid_replies_in_a_row_stop_the_run(tmp_path):
    deps, _ = setup(tmp_path, lambda n, p: reply([(0, "narrow_mlp"), (0, "narrow_mlp"), (0, "narrow_mlp")]))
    state = G.run(deps, run_config())
    assert outcomes(state) == ["INVALID"] * 3
    assert state["stop_reason"] == "3 invalid replies in a row"
    assert state["plan"] == P.dense_plan(CFG)


def test_all_candidates_over_the_step_limit_roll_back(tmp_path):
    def ppl(plan):  # removing any attention group costs +50%: over the 15% per-step limit
        return fake_ppl(plan) * (1.5 if any(p["groups_keep"] < 4 for p in plan.values()) else 1.0)

    def policy(n, prompt):
        if n == 1:
            return reply([(0, "remove_attention_group")], [(1, "remove_attention_group")])
        return reply([(n - 2, "narrow_mlp")])

    deps, backend = setup(tmp_path, policy, evaluate=ppl)
    state = G.run(deps, run_config())
    first = state["history"][0]
    assert first.outcome == "ROLLBACK" and first.gate.decision == "ROLLBACK"
    assert [c.step_rise for c in first.candidates] == pytest.approx([0.575, 0.575])
    assert outcomes(state)[1:] == ["ACCEPT"] * 3
    assert state["accepted"][1].round == 2  # nothing accepted in round 1: the plan stayed unpruned
    assert "ROLLBACK: no candidate within 15%" in backend.prompts[1]


def test_agent_rollback_restores_an_earlier_accepted_plan(tmp_path):
    script = {1: reply([(0, "narrow_mlp")]), 2: reply([(1, "narrow_mlp")]),
              3: reply(action="rollback", rollback_to=0)}
    deps, _ = setup(tmp_path, lambda n, p: script.get(n, reply([(n - 2, "wanda_sparsity")])))
    state = G.run(deps, run_config(rules=Rules(max_rounds=4)))
    rb = state["history"][2]
    assert rb.action == "rollback" and rb.outcome is None
    # round 4 starts from the unpruned plan again: only layer 2 is changed
    assert state["accepted"][0].plan == P.dense_plan(CFG) and len(state["accepted"]) == 2
    assert {i for i, p in state["plan"].items() if p != P.dense_plan(CFG)[i]} == {2}
    assert state["stop_reason"] == "4 rounds"


def test_start_from_flap_evaluates_flaps_allocation(tmp_path):
    deps, _ = setup(tmp_path, lambda n, p: reply(action="start_from_flap"))
    state = G.run(deps, run_config())
    assert outcomes(state) == ["ACCEPT"]
    assert [p["mlp_keep"] for p in state["plan"].values()] == [576] * 4  # 600 rounded down to the 64 grid
    assert state["stop_reason"] == "memory limit met"


def test_stop_after_five_rounds_without_an_accepted_step(tmp_path):
    proposals = [(0, "drop_layer"), (1, "drop_layer"), (2, "drop_layer"), (3, "drop_layer"), (0, "narrow_mlp")]

    def ppl(plan):  # every change is far over the per-step limit
        return DENSE_PPL * (1 if plan == P.dense_plan(CFG) else 2)

    deps, _ = setup(tmp_path, lambda n, p: reply([proposals[n - 1]]), evaluate=ppl)
    state = G.run(deps, run_config())
    assert outcomes(state) == ["ROLLBACK"] * 5
    assert state["stop_reason"] == "5 rounds without an accepted step"
    assert state["usage"].calls == 5
    # the final warning: the limit was not reached; the closest plan is the unpruned one
    (w,) = state["warnings"]
    assert w.kind == "out_of_reach" and w.memory_gap_gb == pytest.approx((DENSE - LIMIT) / 1e9)


def test_call_cap_stops_the_run(tmp_path):
    far = DENSE - 20 * NARROW  # far enough that three accepted steps do not reach it
    deps, _ = setup(tmp_path, lambda n, p: reply([((n - 1) % 4, "narrow_mlp")]))
    state = G.run(deps, run_config(far, max_agent_calls=3))
    assert outcomes(state) == ["ACCEPT"] * 3
    assert state["stop_reason"] == "agent call cap reached (3 calls)"
    assert state["usage"].calls == 3


def test_client_call_cap_also_stops_the_run(tmp_path):
    far = DENSE - 20 * NARROW
    deps, backend = setup(tmp_path, lambda n, p: reply([((n - 1) % 4, "narrow_mlp")]), max_calls=2)
    state = G.run(deps, run_config(far))
    assert len(backend.prompts) == 2 and outcomes(state) == ["ACCEPT"] * 2
    assert state["stop_reason"].startswith("agent call cap reached (call cap reached")


def test_early_warning_when_the_limit_is_out_of_reach(tmp_path):
    def no_baseline(run):
        raise AssertionError("no baseline before an early stop")

    deps, backend = setup(tmp_path, lambda n, p: reply([(0, "narrow_mlp")]), baseline=no_baseline,
                          evaluate=lambda plan: pytest.fail("no evaluation before an early stop"))
    state = G.run(deps, run_config(100_000))  # below the smallest model the rules allow
    (w,) = state["warnings"]
    smallest = E.weight_bytes(CFG, G.smallest_plan(CFG, Rules()))
    assert w.kind == "out_of_reach" and w.memory_gap_gb == pytest.approx((smallest - 100_000) / 1e9)
    assert w.closest_plan == G.smallest_plan(CFG, Rules())
    assert sum(p["dropped"] for p in w.closest_plan.values()) == 3  # 6 allowed, but one layer must remain
    assert state["stop_reason"] == "memory limit out of reach"
    assert backend.prompts == [] and state["history"] == [] and "baseline" not in state


def test_early_warning_when_the_limit_is_already_met(tmp_path):
    deps, backend = setup(tmp_path, lambda n, p: reply([(0, "narrow_mlp")]))
    state = G.run(deps, run_config(DENSE))
    assert [w.kind for w in state["warnings"]] == ["already_met"]
    assert backend.prompts == []


def test_probe_without_a_probe_function_tells_the_agent(tmp_path):
    script = {1: reply(action="probe", probe=[{"layer": 1, "strategy": "narrow_mlp"}])}
    deps, backend = setup(tmp_path, lambda n, p: script.get(n, reply([(n - 2, "narrow_mlp")])))
    state = G.run(deps, run_config())
    assert state["history"][0].action == "probe"
    assert "Probing is not available" in backend.prompts[1]
    assert state["stop_reason"] == "memory limit met"


def test_agent_stop_ends_the_search_with_a_record(tmp_path):
    deps, _ = setup(tmp_path, lambda n, p: reply([(0, "narrow_mlp")]) if n == 1 else reply(action="stop"))
    state = G.run(deps, run_config())
    assert [r.action for r in state["history"]] == ["apply", "stop"]
    assert state["stop_reason"] == "the agent stopped"
    assert [w.kind for w in state["warnings"]] == ["out_of_reach"]
