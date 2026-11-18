"""Validator (task 2.2): every rule of §4.4, the agent-readable reasons, and the three real round-1 candidates."""

import json
from pathlib import Path

import pytest

from pruneagent import estimate as E
from pruneagent import plan as P
from pruneagent import validator as V
from pruneagent.schema import AgentReply, Rules

ROOT = Path(__file__).resolve().parents[1]
# Qwen3-0.6B shape, as a plain dict (no weights needed)
CFG = {"num_hidden_layers": 28, "intermediate_size": 3072, "num_key_value_heads": 8, "num_attention_heads": 16,
       "hidden_size": 1024, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": True}
DENSE = E.weight_bytes(CFG, P.dense_plan(CFG))
NO_LIMIT = 0  # a limit nothing fits: the budget rule never applies


def ctx(plan=None, limit=NO_LIMIT, **kw):
    return V.Context(CFG, plan or {}, limit, **kw)


def ch(layer, strategy):
    return {"layer": layer, "strategy": strategy}


def reasons_of(choices, plan=None, limit=NO_LIMIT, **kw):
    c = V.check_candidate(choices, ctx(plan, limit, **kw))
    assert not c.ok and c.plan is None and c.reasons
    return " | ".join(c.reasons)


def apply(*cands, plan=None, limit=NO_LIMIT, **kw):
    return V.check_reply({"action": "apply", "rationale": "r", "candidates": list(cands)}, ctx(plan, limit, **kw))


# -------------------------------------------------------------------------------------------- round 1 (real)


def test_round1_candidates_are_all_valid():
    rec = json.loads((ROOT / "results/agent_round/round1.json").read_text())
    limit = rec["memory"]["limit_bytes"]
    assert rec["memory"]["dense_bytes"] == DENSE
    reply = {"action": "apply", "rationale": "round 1", "candidates": rec["candidates"]}
    res = V.check_reply(reply, ctx(limit=limit))
    assert res.ok, res.reasons
    assert [c.ok for c in res.candidates] == [True, True, True]
    # same plans and memory as recorded in task 1.11, all within the limit and none a step past it
    for c, v in zip(res.candidates, rec["validation"], strict=True):
        assert P.to_json(c.plan) == v["plan"] and c.weight_bytes == v["weight_bytes"] <= limit
    # the typed reply gives the same verdict
    assert V.check_reply(AgentReply.model_validate(reply), ctx(limit=limit)).ok


def test_round1_candidates_rejected_once_one_choice_is_added():
    """The same candidates with one more narrowing overshoot by more than one step."""
    rec = json.loads((ROOT / "results/agent_round/round1.json").read_text())
    choices = rec["candidates"][1]["choices"]
    once = next(i for i in range(28) if choices.count(ch(i, "narrow_mlp")) < 2)
    more = choices + [ch(once, "narrow_mlp")]
    why = reasons_of(more, limit=rec["memory"]["limit_bytes"])
    assert "frees more memory than the limit needs" in why


# ------------------------------------------------------------------------------------------------ structure


@pytest.mark.parametrize("choice, phrase", [
    ({"layer": 1, "strategy": "narrow_mlp", "amount": 2}, "has field 'amount'; give only \"layer\" and \"strategy\""),
    ({"layer": 1, "strategy": "narrow_mlp", "steps": 2, "ratio": 0.2}, "has fields 'ratio', 'steps'"),
    ({"layer": 28, "strategy": "narrow_mlp"}, "layer 28 does not exist; layers are whole numbers from 0 to 27"),
    ({"layer": -1, "strategy": "narrow_mlp"}, "layer -1 does not exist"),
    ({"layer": "1", "strategy": "narrow_mlp"}, "layer \"1\" does not exist"),
    ({"layer": True, "strategy": "narrow_mlp"}, "layer true does not exist"),
    ({"layer": 1.0, "strategy": "narrow_mlp"}, "layer 1.0 does not exist"),
    ({"layer": 1, "strategy": "quantize"}, "unknown strategy \"quantize\"; use one of narrow_mlp"),
    ({"layer": 1}, "is missing 'strategy'"),
    ("layer 1", "a choice is an object"),
])
def test_choice_structure(choice, phrase):
    assert phrase in reasons_of([choice])


def test_reasons_name_candidate_and_choice_from_one():
    res = apply([ch(1, "narrow_mlp")], [ch(2, "narrow_mlp"), ch(99, "narrow_mlp")])
    assert res.reasons == ["candidate 2, choice 2: layer 99 does not exist; layers are whole numbers from 0 to 27."]
    assert [c.ok for c in res.candidates] == [True, False]


@pytest.mark.parametrize("cand, phrase", [
    ([], "candidate 1 has no choices"),
    ({"choices": []}, "candidate 1 has no choices"),
    ({"rationale": "x"}, "candidate 1 has no list of choices"),
    ("narrow everything", "candidate 1 has no list of choices"),
    ({"choices": [ch(1, "narrow_mlp")], "target_mb": 100}, "has field 'target_mb'; a candidate has only"),
])
def test_candidate_structure(cand, phrase):
    assert phrase in reasons_of(cand)


def test_candidate_as_object_or_bare_list():
    as_obj = V.check_candidate({"choices": [ch(3, "narrow_mlp")], "rationale": "x"}, ctx())
    as_list = V.check_candidate([ch(3, "narrow_mlp")], ctx())
    assert as_obj.ok and as_list.ok and as_obj.plan == as_list.plan


@pytest.mark.parametrize("reply, phrase", [
    ({"action": "prune", "rationale": "r"}, "Unknown action \"prune\""),
    ({"action": "stop"}, "no rationale"),
    ({"action": "stop", "rationale": "  "}, "no rationale"),
    ({"action": "stop", "rationale": "r", "memory_target": 1.0}, "unknown field 'memory_target'"),
    ({"action": "stop", "rationale": "r", "candidates": [[ch(1, "narrow_mlp")]]}, "Action 'stop' takes no 'candidates'"),
    ({"action": "apply", "rationale": "r", "rollback_to": 0, "candidates": [[ch(1, "narrow_mlp")]]},
     "Action 'apply' takes no 'rollback_to'"),
    ({"action": "apply", "rationale": "r"}, "No candidates; propose 1 to 3"),
    ({"action": "apply", "rationale": "r", "candidates": []}, "No candidates"),
    ({"action": "probe", "rationale": "r"}, "needs a non-empty \"probe\" list"),
    ({"action": "probe", "rationale": "r", "probe": [ch(1, "shrink")]}, "probe 1: unknown strategy"),
    ({"action": "rollback", "rationale": "r"}, "Nothing to roll back to"),
])
def test_reply_structure(reply, phrase):
    res = V.check_reply(reply, ctx())
    assert not res.ok and any(phrase in r for r in res.reasons), res.reasons


def test_reply_not_an_object():
    assert V.check_reply(["apply"], ctx()).reasons == ["The reply is not a JSON object."]


@pytest.mark.parametrize("reply", [
    {"action": "stop", "rationale": "done"},
    {"action": "start_from_flap", "rationale": "compare"},
    {"action": "probe", "rationale": "look", "probe": [ch(3, "drop_layer"), ch(4, "narrow_mlp")]},
])
def test_valid_non_apply_replies(reply):
    assert V.check_reply(reply, ctx()).ok


def test_probe_on_dropped_layer():
    res = V.check_reply({"action": "probe", "rationale": "r", "probe": [ch(3, "narrow_mlp")]},
                        ctx({3: {"dropped": True}}))
    assert res.reasons == ["probe 1: layer 3 is already dropped in the current plan."]


def test_rollback_targets():
    three = ctx(n_accepted=3)  # start + two accepted plans: targets 0 and 1
    assert V.check_reply({"action": "rollback", "rationale": "r"}, three).ok
    assert V.check_reply({"action": "rollback", "rationale": "r", "rollback_to": 1}, three).ok
    for bad in (2, -1, "1"):
        res = V.check_reply({"action": "rollback", "rationale": "r", "rollback_to": bad}, three)
        assert not res.ok and "use 0 to 1" in res.reasons[0]


# ----------------------------------------------------------------------------------------------- step sizes


def test_one_fixed_step_per_choice():
    c = V.check_candidate([ch(1, "narrow_mlp"), ch(2, "narrow_mlp"), ch(2, "narrow_mlp"),
                           ch(3, "remove_attention_group"), ch(4, "wanda_sparsity"), ch(5, "drop_layer")], ctx())
    assert c.ok
    assert c.plan[1]["mlp_keep"] == 3072 - 320 and c.plan[2]["mlp_keep"] == 3072 - 640
    assert c.plan[3]["groups_keep"] == 7 and c.plan[4]["wanda_sparsity"] == 0.1 and c.plan[5]["dropped"]
    assert c.plan[6] == P.dense_plan(CFG)[6]
    assert c.weight_bytes == E.weight_bytes(CFG, c.plan)


def test_choice_order_does_not_matter():
    a = V.check_candidate([ch(1, "narrow_mlp"), ch(1, "wanda_sparsity"), ch(2, "drop_layer")], ctx())
    b = V.check_candidate([ch(2, "drop_layer"), ch(1, "wanda_sparsity"), ch(1, "narrow_mlp")], ctx())
    assert a.plan == b.plan


def test_steps_continue_from_current_plan():
    c = V.check_candidate([ch(1, "narrow_mlp")], ctx({1: {"mlp_keep": 2432}}))
    assert c.plan[1]["mlp_keep"] == 2112


def test_same_strategy_at_most_twice_per_layer():
    assert "narrow_mlp on layer 1 is chosen 3 times; at most 2 steps of the same strategy" in reasons_of(
        [ch(1, "narrow_mlp")] * 3)
    assert V.check_candidate([ch(1, "narrow_mlp")] * 2 + [ch(2, "narrow_mlp")] * 2, ctx()).ok


def test_rule_values_come_from_rules():
    assert V.check_candidate([ch(1, "narrow_mlp")] * 3, ctx(rules=Rules(max_same_steps=3))).ok
    assert "below the minimum of 1,536 (50%)" in reasons_of([ch(1, "narrow_mlp")] * 2, {1: {"mlp_keep": 1792}},
                                                          rules=Rules(min_mlp_fraction=0.5))


def test_layer_dropped_once_and_alone():
    assert "drop_layer on layer 4 is chosen twice; a layer can be dropped only once" in reasons_of(
        [ch(4, "drop_layer")] * 2)
    assert "layer 4 is dropped and also given narrow_mlp, wanda_sparsity" in reasons_of(
        [ch(4, "narrow_mlp"), ch(4, "drop_layer"), ch(4, "wanda_sparsity")])


# ------------------------------------------------------------------------------------------ structure rules


def test_at_most_one_drop_per_candidate():
    assert "candidate 1 drops 2 layers (3, 7); at most 1 drop_layer per candidate" in reasons_of(
        [ch(7, "drop_layer"), ch(3, "drop_layer")])


def test_at_most_six_drops_per_run():
    five = {i: {"dropped": True} for i in range(5)}
    assert V.check_candidate([ch(10, "drop_layer")], ctx(five)).ok
    six = {i: {"dropped": True} for i in range(6)}
    assert "would bring the run to 7 dropped layers; at most 6 per run (6 already dropped)" in reasons_of(
        [ch(10, "drop_layer")], six)
    assert V.check_candidate([ch(10, "narrow_mlp")], ctx(six)).ok  # other strategies still allowed


def test_no_choice_on_a_dropped_layer():
    assert "layer 3 is already dropped in the current plan; choose a remaining layer" in reasons_of(
        [ch(3, "narrow_mlp")], {3: {"dropped": True}})


def test_minimum_mlp_width():
    # 25% of 3,072 = 768; steps of 320 go 1,152 -> 832 -> 512
    assert V.check_candidate([ch(3, "narrow_mlp")], ctx({3: {"mlp_keep": 1152}})).plan[3]["mlp_keep"] == 832
    why = reasons_of([ch(3, "narrow_mlp")], {3: {"mlp_keep": 832}})
    assert ("1 narrow_mlp step on layer 3 would leave 512 of its 3,072 MLP neurons, below the minimum of 768 (25%); "
            "this layer can take 0 more narrow_mlp steps") in why
    why = reasons_of([ch(3, "narrow_mlp")] * 2, {3: {"mlp_keep": 1152}})
    assert "2 narrow_mlp steps on layer 3" in why and "can take 1 more narrow_mlp step." in why


def test_minimum_attention_groups():
    assert V.check_candidate([ch(3, "remove_attention_group")], ctx({3: {"groups_keep": 2}})).ok
    why = reasons_of([ch(3, "remove_attention_group")], {3: {"groups_keep": 1}})
    assert "would leave 0 key/value groups; at least 1 must remain (drop_layer removes the whole layer)" in why
    assert "can take 1 more remove_attention_group step." in reasons_of(
        [ch(3, "remove_attention_group")] * 2, {3: {"groups_keep": 2}})


def test_maximum_wanda_sparsity():
    assert V.check_candidate([ch(3, "wanda_sparsity")], ctx({3: {"wanda_sparsity": 0.6}})).plan[3][
        "wanda_sparsity"] == 0.7
    why = reasons_of([ch(3, "wanda_sparsity")] * 2, {3: {"wanda_sparsity": 0.6}})
    assert "would bring its unstructured sparsity to 80%, above the maximum of 70%" in why


def test_every_problem_is_reported_at_once():
    why = reasons_of([ch(1, "narrow_mlp")] * 3 + [ch(5, "drop_layer"), ch(6, "drop_layer")])
    assert "chosen 3 times" in why and "drops 2 layers" in why


# ---------------------------------------------------------------------------------------------------- budget


def saving(strategy):
    return DENSE - E.weight_bytes(CFG, P.step(P.dense_plan(CFG), 0, strategy, CFG))


def test_budget_allows_reaching_the_limit_by_up_to_one_step():
    narrow = saving("narrow_mlp")
    limit = DENSE - int(1.5 * narrow)  # needs two narrowings
    assert V.check_candidate([ch(1, "narrow_mlp"), ch(2, "narrow_mlp")], ctx(limit=limit)).ok
    why = reasons_of([ch(1, "narrow_mlp"), ch(2, "narrow_mlp"), ch(3, "narrow_mlp")], limit=limit)
    assert why.startswith("candidate 1 frees more memory than the limit needs: its weights would be ")
    assert "it would still fit without one narrow_mlp on layer 1 (which frees 1.964 MB)" in why


def test_budget_allows_candidates_that_stay_over_the_limit():
    assert V.check_candidate([ch(1, "narrow_mlp")], ctx(limit=DENSE - 10 * saving("narrow_mlp"))).ok


def test_budget_unneeded_small_step_beside_a_drop():
    limit = DENSE - saving("drop_layer") // 2  # a drop alone reaches it
    assert V.check_candidate([ch(9, "drop_layer")], ctx(limit=limit)).ok
    assert "without one narrow_mlp on layer 1" in reasons_of([ch(9, "drop_layer"), ch(1, "narrow_mlp")], limit=limit)
    # but a narrowing that is needed beside the drop is fine
    limit = DENSE - saving("drop_layer") - saving("narrow_mlp") // 2
    assert V.check_candidate([ch(9, "drop_layer"), ch(1, "narrow_mlp")], ctx(limit=limit)).ok


def test_budget_ignores_choices_that_free_no_memory():
    limit = DENSE - saving("narrow_mlp") // 2
    assert V.check_candidate([ch(1, "narrow_mlp"), ch(2, "wanda_sparsity"), ch(3, "wanda_sparsity")],
                             ctx(limit=limit)).ok


def test_budget_when_the_current_plan_already_fits():
    assert "frees more memory than the limit needs" in reasons_of([ch(1, "narrow_mlp")], limit=DENSE)
    assert V.check_candidate([ch(1, "wanda_sparsity")], ctx(limit=DENSE)).ok


def test_budget_counts_a_repeated_choice_once():
    narrow = saving("narrow_mlp")
    limit = DENSE - int(1.5 * narrow)
    assert V.check_candidate([ch(1, "narrow_mlp")] * 2, ctx(limit=limit)).ok
    assert "without one narrow_mlp on layer 1" in reasons_of([ch(1, "narrow_mlp")] * 2 + [ch(2, "narrow_mlp")],
                                                             limit=limit)


# ------------------------------------------------------------------------------------ candidates per round


def test_at_most_three_candidates():
    res = apply(*[[ch(i, "narrow_mlp")] for i in range(4)])
    assert not res.ok
    assert res.reasons == ["4 candidates proposed; at most 3 per round."]
    assert [c.ok for c in res.candidates] == [True, True, True, False]
    assert res.candidates[3].reasons == ["candidate 4 is beyond the first 3 and was not checked."]


def test_duplicate_candidates_in_one_round():
    res = apply([ch(1, "narrow_mlp"), ch(2, "drop_layer")], [ch(3, "narrow_mlp")],
                [ch(2, "drop_layer"), ch(1, "narrow_mlp")])
    assert [c.ok for c in res.candidates] == [True, True, False]
    assert res.reasons == [("candidate 3 gives the same plan as candidate 1 (the same choices, perhaps in another "
                           "order); propose a different candidate.")]


def test_candidate_repeating_an_earlier_round():
    tried = (P.step(P.dense_plan(CFG), 1, "narrow_mlp", CFG),)
    res = apply([ch(1, "narrow_mlp")], [ch(2, "narrow_mlp")], tried=tried)
    assert [c.ok for c in res.candidates] == [False, True]
    assert "already evaluated in an earlier round" in res.reasons[0]


def test_feedback_for_the_agent():
    bad = apply([ch(1, "narrow_mlp", ) | {"amount": 0.2}])
    assert bad.feedback().startswith("Your proposal was rejected:\n- candidate 1, choice 1 has field 'amount'")
    good = apply([ch(1, "narrow_mlp")])
    assert good.ok and good.feedback() == "All candidates are valid."


def test_validator_is_pure():
    plan = {3: {"mlp_keep": 2432}}
    snapshot = json.dumps(plan)
    V.check_reply({"action": "apply", "rationale": "r", "candidates": [[ch(3, "narrow_mlp")]]}, ctx(plan))
    assert json.dumps(plan) == snapshot
