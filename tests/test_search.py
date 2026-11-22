"""Search strategies (search.py, ADR-009): the strategy decides only what one candidate may change; the prompt for
several-layers runs is unchanged, and everything after a candidate (evaluation, gate, stop rules) is shared."""


import itertools
import re

from test_graph import CFG, DENSE, NARROW, outcomes, run_config, setup
from test_instructions import prompts_of, v1_reply

from pruneagent import graph as G
from pruneagent import search as S
from pruneagent import validator as V
from pruneagent.schema import Rules

ORIGINAL = ("Each candidate is a list\nof (layer, method) choices, applied together to the current model, one\n"
            "fixed-size step per choice.")


def test_strategy_sets_its_rules_unless_the_run_sets_them():
    assert run_config().rules.max_layers_per_candidate is None and run_config().search == "multi_layer"
    lbl = run_config(search="layer_by_layer").rules
    assert lbl.max_layers_per_candidate == 1 and lbl.max_same_steps == 7
    assert run_config(search="layer_by_layer", rules=Rules(max_same_steps=3)).rules.max_same_steps == 3
    # a saved run reads back the same rules
    cfg = run_config(search="layer_by_layer")
    again = type(cfg).model_validate_json(cfg.model_dump_json())
    assert again.rules == cfg.rules and again.search == "layer_by_layer"


def test_validator_keeps_a_layer_by_layer_candidate_on_one_layer():
    rules = run_config(search="layer_by_layer").rules
    ctx = V.Context(CFG, {}, 0, rules=rules)  # limit 0: nothing fits, so the budget rule never applies
    two = V.check_candidate([{"layer": 1, "strategy": "narrow_mlp"}, {"layer": 2, "strategy": "narrow_mlp"}], ctx)
    assert not two.ok and any("changes 2 layers (1, 2)" in r for r in two.reasons)
    deep = V.check_candidate([{"layer": 1, "strategy": "narrow_mlp"}] * 4 + [{"layer": 1, "strategy": "remove_attention_group"}], ctx)
    assert deep.ok, deep.reasons
    # the same two-layer candidate is fine for several-layers runs
    assert V.check_candidate([{"layer": 1, "strategy": "narrow_mlp"}, {"layer": 2, "strategy": "narrow_mlp"}],
                             V.Context(CFG, {}, 0)).ok


def test_prompt_states_the_strategy(tmp_path):
    policy = lambda n, p: v1_reply([((n - 1) % 4, "narrow_mlp")])
    _, multi = prompts_of(tmp_path / "m", policy, max_agent_calls=2)
    _, single = prompts_of(tmp_path / "s", policy, max_agent_calls=2, search="layer_by_layer")
    assert ORIGINAL in multi[0] and "This round:" not in multi[0]
    flat = " ".join(single[0].split())
    assert S.get("layer_by_layer").candidate_rule in flat and "Every choice in one candidate names the same layer." in flat
    assert "This round: layer 0 (layer 1 of 4, first to last)" in flat
    assert "- narrow_mlp: 1 to" in single[0] and "- or no candidates: the layer stays as it is" in single[0]
    assert "This round: layer 1 (layer 2 of 4" in " ".join(single[1].split())  # the next round, the next layer
    # nothing in a sweep's prompt offers going back, or a round count it doesn't use
    assert "return to an earlier kept plan" not in flat and "rollback_to" not in flat
    assert 'To leave the layer as it is, answer {"candidates": [], "reason": "<why, naming the evidence>"}.' in flat and "You have 4 rounds left." in flat


FOLLOW_ON = "The pass through every layer, one layer per round, ended above the memory limit."


def after_pass(n, p):
    """After the pass: several layers per step, two layers slimmed in one step."""
    return v1_reply([(1, "narrow_mlp"), (2, "narrow_mlp")])


def sweep(tmp_path, policy, **kw):
    deps, backend = setup(tmp_path, lambda n, p: policy(n, p) if "This round:" in p else after_pass(n, p), **kw)
    # a limit within the rules' reach that these policies (at most 2 steps a layer) don't reach in the pass
    return G.run(deps, run_config(search="layer_by_layer", limit_bytes=DENSE - 10 * NARROW)), backend


def test_sweep_decides_each_layer_once_in_order(tmp_path):
    # round k: cut plans for layer k only; layer 2 is left as it is
    def policy(n, p):
        layer = int(re.search(r"This round: layer (\d+)", p).group(1))
        if layer == 2:
            return {"candidates": [], "reason": "layer 2's quick tests show high damage for every cut"}
        return v1_reply([(layer, "narrow_mlp")], [(layer, "narrow_mlp")] * 2)
    state, backend = sweep(tmp_path, policy)
    assert sum("This round:" in p for p in backend.prompts) == 4  # exactly one decision per layer
    assert outcomes(state)[:4] == ["ACCEPT", "ACCEPT", "LEFT", "ACCEPT"]
    # the step log keeps why the layer was left
    assert state["history"][2].rationale == "layer 2's quick tests show high damage for every cut"
    assert backend.schemas[0] is G.SWEEP_REPLY_SCHEMA
    plans = [a.plan for a in state["accepted"]]
    changed = [[i for i in after if after[i] != before[i]] for before, after in itertools.pairwise(plans)]
    assert changed[:3] == [[0], [1], [3]]


def test_sweep_ending_above_the_limit_continues_with_several_layers_per_step(tmp_path):
    def policy(n, p):
        layer = int(re.search(r"This round: layer (\d+)", p).group(1))
        return v1_reply([(layer, "narrow_mlp")])
    state, backend = sweep(tmp_path, policy)
    assert outcomes(state)[:4] == ["ACCEPT"] * 4 and len(outcomes(state)) > 4
    assert state["stop_reason"] == "memory limit met"
    # round 5 is several layers per step: its rules, instructions and reply form, and no layer of its own
    after = " ".join(backend.prompts[4].split())
    assert FOLLOW_ON in after and "This round:" not in after
    assert S.get("multi_layer").candidate_rule in after and "return to an earlier kept plan" in after
    assert "Every choice in one candidate names the same layer." not in after
    assert backend.schemas[4] is G.REPLY_SCHEMA
    changed = [[i for i in b if b[i] != a[i]] for a, b in itertools.pairwise(a.plan for a in state["accepted"])]
    assert changed[4] == [1, 2]  # two layers in one step


def test_rules_after_the_pass_are_several_layers_per_step():
    run = run_config(search="layer_by_layer", rules=Rules(step_limit=0.1))
    after = S.follow_on_rules(run)
    assert after.max_layers_per_candidate is None and after.max_same_steps == 2 and after.step_limit == 0.1


def test_sweep_asks_again_when_a_layer_is_left_without_a_reason(tmp_path):
    def policy(n, p):
        layer = int(re.search(r"This round: layer (\d+)", p).group(1))
        if layer == 1:
            return {"candidates": []} if "Give the reason for leaving layer 1" not in p else \
                {"candidates": [], "reason": "attention and feed-forward both measured fragile"}
        return v1_reply([(layer, "narrow_mlp")])
    state, _backend = sweep(tmp_path, policy)
    assert outcomes(state)[:5] == ["ACCEPT", "INVALID", "LEFT", "ACCEPT", "ACCEPT"]
    assert state["history"][2].rationale == "attention and feed-forward both measured fragile"


def test_sweep_refuses_other_layers_and_asks_again(tmp_path):
    def policy(n, p):
        layer = int(re.search(r"This round: layer (\d+)", p).group(1))
        return v1_reply([(layer + 1, "narrow_mlp")]) if n == 1 else v1_reply([(layer, "narrow_mlp")])
    state, backend = sweep(tmp_path, policy)
    assert outcomes(state)[:2] == ["INVALID", "ACCEPT"]
    assert "this round decides layer 0 only" in backend.prompts[1]
    assert "This round: layer 0 " in " ".join(backend.prompts[1].split())  # the same layer again
    assert sum("This round:" in p for p in backend.prompts) == 5  # 4 layers, one asked again


def test_sweep_stops_early_when_the_limit_is_met(tmp_path):
    deps, backend = setup(tmp_path, lambda n, p: v1_reply(
        [(int(re.search(r"This round: layer (\d+)", p).group(1)), "narrow_mlp")] * 3))
    state = G.run(deps, run_config(search="layer_by_layer"))  # the test limit: about three slimming steps
    assert state["stop_reason"] == "memory limit met" and len(backend.prompts) < 4


def test_sweep_time_estimate_is_one_round_per_layer():
    from pruneagent import runstore as R

    rates = [{"minutes_per_mb": 0.5, "rounds_per_mb": 0.1, "minutes_per_round": 2.0},
             {"minutes_per_mb": 0.3, "rounds_per_mb": 0.2, "minutes_per_round": 3.0}]
    t = R.time_estimate(rates, 238.0, 40, sweep_layers=28)
    assert t["rounds"] == [28, 28] and t["minutes"] == [56.0, 84.0]
    assert t["max_rounds"] == 68 and t["worst_minutes"] == 68 * 3.0  # the pass, then up to 40 more rounds
    assert R.time_estimate(rates, 238.0, 40)["rounds"] != [28, 28]  # several layers per step: from MB cut


def test_the_live_view_sees_the_round_phase(tmp_path):
    seen = []
    deps, _ = setup(tmp_path, lambda n, p: v1_reply([(int(re.search(r"This round: layer (\d+)", p).group(1)), "narrow_mlp")])
                    if "This round:" in p else after_pass(n, p))
    real = deps.store.set_now
    deps.store.set_now = lambda now: (seen.append(now), real(now))
    G.run(deps, run_config(search="layer_by_layer", limit_bytes=DENSE - 10 * NARROW))
    assert {"round": 1, "layer": 0, "phase": "proposing"} in seen
    assert {"round": 1, "layer": 0, "phase": "measuring", "plans": 1} in seen
    assert seen[-1] is None  # nothing between rounds or after the end
