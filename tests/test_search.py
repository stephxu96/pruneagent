"""Search strategies (search.py, ADR-009): the strategy decides only what one candidate may change; the prompt for
several-layers runs is unchanged, and everything after a candidate (evaluation, gate, stop rules) is shared."""


import itertools

from test_graph import CFG, outcomes, run_config, setup
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
    assert lbl.max_layers_per_candidate == 1 and lbl.max_same_steps == 4
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
    assert ORIGINAL in multi[0] and "names the same layer" not in multi[0]
    flat = " ".join(single[0].split())
    assert S.get("layer_by_layer").candidate_rule in flat and "Every choice in one candidate names the same layer." in flat
    assert "at most four times" in flat or "at most 4" in flat


def test_layer_by_layer_run_changes_one_layer_per_kept_step(tmp_path):
    # rounds 1-3: two one-layer ideas, each on its own layer; the gate keeps one of them, so one layer changes
    deps, _ = setup(tmp_path, lambda n, p: v1_reply([((n - 1) % 4, "narrow_mlp")], [(n % 4, "narrow_mlp")]))
    state = G.run(deps, run_config(search="layer_by_layer"))
    assert outcomes(state).count("ACCEPT") >= 2
    plans = [a.plan for a in state["accepted"]]
    for before, after in itertools.pairwise(plans):
        assert sum(after[i] != before[i] for i in after) == 1


def test_a_two_layer_idea_is_refused_with_the_reason(tmp_path):
    deps, backend = setup(tmp_path, lambda n, p: v1_reply([(0, "narrow_mlp"), (1, "narrow_mlp")]), max_calls=2)
    state = G.run(deps, run_config(search="layer_by_layer", max_agent_calls=2))
    assert outcomes(state)[0] == "INVALID"
    assert "changes 2 layers (0, 1)" in backend.prompts[1]  # the agent is told why
