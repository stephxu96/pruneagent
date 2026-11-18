"""Agent instructions v1 (task 2.9): the approved text, every placeholder filled from the code's real values, the
reply in the instructions' form (method, reason, "layer 12", rollback_to), the three-method menu and the v1 actions.
Fakes only: no model, GPU or real agent call."""

import json
import re

import pytest
from test_graph import CFG, DENSE, FLAP_PPL, LIMIT, NARROW, outcomes, run_config, setup

from pruneagent import agent_client as AC
from pruneagent import agent_round
from pruneagent import graph as G
from pruneagent import instructions as I
from pruneagent import plan as P
from pruneagent import validator as V
from pruneagent.schema import AgentReply, Rules, normalize_reply

STEP = agent_round.step_savings(CFG)


def v1_reply(*candidates):
    """A reply exactly as the instructions ask for it: method, reason, layers named "layer N"."""
    return {"candidates": [{"choices": [{"layer": f"layer {la}", "method": s} for la, s in c],
                            "reason": "quick tests"} for c in candidates]}


def mb(n_bytes):  # MB as the prompt prints them
    return I._mb(n_bytes)


# ------------------------------------------------------------------------------------------------ the text


def test_template_is_the_approved_text_with_the_three_edits():
    text = I.template()
    assert text.startswith("Your job. You are pruning a language model so its weights fit within a memory\n")
    assert "wanda" not in text.lower()  # (a)
    flat = " ".join(text.split())
    assert ("Small steps let you re-measure before committing further. You can return to an earlier kept plan, but a "
            "later step cannot undo the damage of an earlier one.") in flat  # (b)
    assert "they do not make damage reversible" not in flat
    assert "The three methods (one step each):" in text  # (c)
    assert ('To go back, answer {"rollback_to": <index of an earlier kept plan>} instead of candidates.') in flat
    assert set(I.placeholders(text)) == {
        "limit", "flap_ppl", "dense_ppl", "rejection_rule", "selection_rule", "rounds_left", "stop_rule", "x",
        "n_groups", "y", "z", "current_mb", "mb_to_free", "current_ppl", "history", "limits"}


# ------------------------------------------------------------------------------------------- the rendering


def prompts_of(tmp_path, policy, **cfg):
    deps, backend = setup(tmp_path, policy)
    state = G.run(deps, run_config(**cfg))
    return state, backend.prompts


def test_round_one_prompt_is_filled_from_the_real_values(tmp_path):
    rules = Rules(step_limit=0.10, max_rounds=12, max_rounds_without_accept=4, max_drops_per_run=2)
    state, prompts = prompts_of(tmp_path, lambda n, p: v1_reply([((n - 1) % 4, "narrow_mlp")]), rules=rules,
                                max_agent_calls=9)
    first = prompts[0]
    assert I.placeholders(first) == [] and "{limits" not in first
    flat = " ".join(first.split())
    dense_ppl = state["accepted"][0].val_ppl
    # memory: limit, now, still to free (the run config and the estimator)
    assert f"memory limit of {mb(LIMIT)} MB" in flat
    assert f"Memory now: {mb(DENSE)} MB. Still to free: {mb(DENSE - LIMIT)} MB." in flat
    # Baseline: FLAP on the gate's windows, and the unpruned model on the same windows
    assert f"FLAP's perplexity at this limit is {FLAP_PPL:.2f}." in flat
    assert f"The unpruned model's is {dense_ppl:.2f}." in flat and f"Perplexity now: {dense_ppl:.2f}." in flat
    # the gate (graph.choose) and the stop rules (graph.stop_check), from these Rules
    assert (f"rejected if its perplexity is more than 10% above the current model's (above {dense_ppl * 1.1:.2f} "
            "this round)") in flat
    assert "code keeps the one with the smallest perplexity increase per MB freed" in flat
    assert "You have 9 rounds left." in flat  # the call cap (9) binds before the round cap (12)
    assert "after 4 rounds in a row without a kept step (0 so far), after 12 rounds or 9 replies in all" in flat
    # one step of each method (the estimator) and the config's attention groups
    assert f"Frees about {mb(STEP['narrow_mlp'])} MB." in flat and mb(STEP["narrow_mlp"]) == mb(NARROW)
    assert f"removes one of the layer's 4 attention groups. Frees about {mb(STEP['remove_attention_group'])} MB" \
        in flat
    assert f"drop_layer: removes the whole layer. Frees about {mb(STEP['drop_layer'])} MB." in flat
    # the rules (validator)
    assert "Layers are numbered 0 to 3" in flat and "At most 1 drop_layer per candidate and 2 in the whole run" in flat
    assert "at least 25% of its original feed-forward width (160 of 640 neurons)" in flat
    assert "Previous rounds: none yet; this is round 1." in flat
    assert "wanda" not in first.lower()
    assert "Data for round 1." in first and "layer 3 | " in first  # the quick-test rows


def test_history_shows_every_candidate_including_rejected_ones(tmp_path):
    def ppl(plan):  # removing any attention group costs +50%: over the 15% per-step limit
        from test_graph import fake_ppl
        return fake_ppl(plan) * (1.5 if any(p["groups_keep"] < 4 for p in plan.values()) else 1.0)

    script = {1: v1_reply([(0, "narrow_mlp")], [(1, "remove_attention_group")]),
              2: {"candidates": [{"choices": [{"layer": 99, "method": "narrow_mlp"}], "reason": "r"}]},
              3: v1_reply([(1, "narrow_mlp")], [(2, "narrow_mlp"), (3, "narrow_mlp")])}
    deps, backend = setup(tmp_path, lambda n, p: script[n], evaluate=ppl)
    state = G.run(deps, run_config(rules=Rules(max_rounds=3)))
    assert outcomes(state) == ["ACCEPT", "INVALID", "ACCEPT"]
    second, third = backend.prompts[1], backend.prompts[2]
    assert I.placeholders(third) == []
    assert "round | candidate | choices | perplexity | change | MB freed | change per MB | result" in second
    assert f"1 | 1 | layer 0 narrow_mlp | 10.20 | +2.0% | {mb(NARROW)} |" in second
    assert "| kept: plan 1" in second
    assert "1 | 2 | layer 1 remove_attention_group | 15.75 | +57.5% |" in second
    assert "| rejected: over +15%" in second
    assert "2 | - | reply not usable: candidate 1, choice 1: layer 99 does not exist" in third
    assert "Kept plans you can return to: plan 0: the unpruned model" in second
    assert "plan 1: kept in round 1" in second and "(current)" in second
    # the last reply's reasons come at the end of the next prompt only
    assert "About your previous reply." in third and "About your previous reply." not in second
    # round 3: one narrowing (+2.0% for one step) beats two (+4.04% for two steps) per MB freed
    third_round = state["history"][2]
    assert third_round.gate.chosen_candidate == 0
    row = G.past_round(third_round, state["accepted"], 0.15)
    assert [c.result for c in row.candidates] == ["kept: plan 2", "not kept: candidate 1 chosen"]


def test_measured_damage_and_the_current_model_are_shown(tmp_path):
    from pruneagent.schema import SensitivityRow, SensitivityTable

    def measured():
        rows = {i: SensitivityRow(layer=i, block_influence=0.1, measured_ppl={
            "narrow_mlp": 10.1, "remove_attention_group": 10.4, "drop_layer": 2e6, "wanda_sparsity": 10.0})
            for i in range(4)}
        return SensitivityTable(dense_ppl=10.0, measured_windows=32, rows=rows)

    deps, backend = setup(tmp_path, lambda n, p: v1_reply([(n - 1, "narrow_mlp")]))
    deps.load_sensitivity = measured
    G.run(deps, run_config(rules=Rules(max_rounds=2)))
    second = backend.prompts[1]
    assert "on 32 windows (unpruned: 10.00" in second
    assert "layer 0 | 10.10 (+1.0%) | 10.40 (+4.0%) | 2,000,000 (200,000x)" in second
    assert ("The model now (layers not listed are at full size):\n- layer 0: feed-forward width 576 of 640, 4 of 4 "
            "attention groups") in second
    assert "wanda" not in second.lower()


# ------------------------------------------------------------------------------------------- reply forms


@pytest.mark.parametrize("layer", [12, "12", "layer 12", "Layer 12", " layer12 "])
def test_layer_names_are_accepted(layer):
    r = AgentReply.model_validate({"candidates": [{"choices": [{"layer": layer, "method": "narrow_mlp"}],
                                                   "reason": "r"}]})
    assert r.candidates[0].choices[0].layer == 12
    assert V.check_reply({"candidates": [[{"layer": layer, "method": "narrow_mlp"}]]},
                         V.Context(CFG_28, {}, 0)).ok


CFG_28 = {"num_hidden_layers": 28, "intermediate_size": 3072, "num_key_value_heads": 8, "num_attention_heads": 16,
          "hidden_size": 1024, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": True}


@pytest.mark.parametrize("raw", [
    {"candidates": [{"choices": [{"layer": "layer 2", "method": "narrow_mlp"}], "reason": "why"}]},
    {"candidates": [{"choices": [{"layer": 2, "strategy": "narrow_mlp"}], "rationale": "why"}]},
    {"candidates": [{"choices": [{"layer": "2", "method": "narrow_mlp"}], "rationale": "why"}], "rollback_to": None},
    {"action": "apply", "rationale": "why", "candidates": [{"choices": [{"layer": 2, "strategy": "narrow_mlp"}],
                                                             "reason": "why"}]},
])
def test_field_aliases_normalise_to_one_form(raw):
    norm = normalize_reply(raw)
    assert norm == {"action": "apply", "rationale": "why",
                    "candidates": [{"choices": [{"layer": 2, "strategy": "narrow_mlp"}], "rationale": "why"}]}
    assert V.check_reply(raw, V.Context(CFG_28, {}, 0)).ok
    assert AgentReply.model_validate(raw) == AgentReply.model_validate(norm)


def test_reply_schema_matches_the_instructions():
    schema = G.REPLY_SCHEMA
    assert set(schema["properties"]) == {"candidates", "rollback_to"}
    cand = schema["properties"]["candidates"]["items"]
    assert set(cand["properties"]) == {"choices", "reason"}
    choice = cand["properties"]["choices"]["items"]
    assert set(choice["properties"]) == {"layer", "method"} and choice["properties"]["layer"]["type"] == "string"
    assert choice["properties"]["method"]["enum"] == ["narrow_mlp", "remove_attention_group", "drop_layer"]
    assert schema["properties"]["rollback_to"]["type"] == ["integer", "null"]
    # the answer format in the text names the same fields
    text = I.template()
    assert all(f'"{k}"' in text for k in ("candidates", "choices", "layer", "method", "reason", "rollback_to"))


# ------------------------------------------------------------------------------------- menu and actions (v1)


def test_wanda_sparsity_is_rejected_with_a_clear_message():
    res = V.check_reply(v1_reply([(3, "wanda_sparsity")]), V.Context(CFG_28, {}, 0))
    assert not res.ok
    assert res.reasons == [("candidate 1, choice 1: wanda_sparsity is not one of the methods you can use; use one of "
                           "narrow_mlp, remove_attention_group, drop_layer.")]
    # the rule itself is still there for the full menu (and the pruner still applies it: the Wanda backup)
    assert V.check_reply(v1_reply([(3, "wanda_sparsity")]), V.Context(CFG_28, {}, 0, strategies=P.STRATEGIES)).ok


@pytest.mark.parametrize("raw", [
    {"action": "probe", "rationale": "look", "probe": [{"layer": 1, "strategy": "narrow_mlp"}]},
    {"action": "stop", "rationale": "done"},
    {"action": "start_from_flap", "rationale": "compare"},
])
def test_disabled_actions_are_rejected(raw):
    res = V.check_reply(raw, V.Context(CFG_28, {}, 0))
    assert not res.ok and len(res.reasons) == 1
    assert f"Action '{raw['action']}' is not available in this version" in res.reasons[0]
    assert '{"rollback_to": <index of an earlier kept plan>}' in res.reasons[0]
    assert V.check_reply(raw, V.Context(CFG_28, {}, 0, actions=V.ACTIONS)).ok  # still there behind the flag


@pytest.mark.parametrize("raw", [{"rollback_to": 1}, {"candidates": [], "rollback_to": 1},
                                 {"rollback_to": 1, "reason": "layer 3 hurt more than it saved"}])
def test_rollback_reply_is_parsed(raw):
    r = AgentReply.model_validate(raw)
    assert r.action == "rollback" and r.rollback_to == 1 and r.candidates == []
    assert V.check_reply(raw, V.Context(CFG_28, {}, 0, n_accepted=3)).ok
    bad = V.check_reply(raw, V.Context(CFG_28, {}, 0, n_accepted=2))
    assert bad.reasons == ["rollback_to 1 is not an earlier kept plan; use 0 to 0 (0 is the unpruned model)."]


@pytest.mark.parametrize("raw, phrase", [
    ({"candidates": [[{"layer": 1, "method": "narrow_mlp"}]], "rollback_to": 0}, "gives both candidates and"),
    ({"candidates": [], "rollback_to": None}, "The reply has no candidates; propose 1 to 3, or answer"),
    ({}, "The reply has no candidates"),
])
def test_both_or_neither_answer_is_explained(raw, phrase):
    res = V.check_reply(raw, V.Context(CFG_28, {}, 0, n_accepted=2))
    assert not res.ok and phrase in res.reasons[0]


def test_client_reads_a_bare_rollback_reply(tmp_path):
    deps, _backend = setup(tmp_path, lambda n, p: '{"rollback_to": 0}')
    out = deps.agent.propose(AC.AgentRequest("p", G.REPLY_SCHEMA, required=G.REPLY_REQUIRED))
    assert out.data == {"rollback_to": 0}


# ----------------------------------------------------------------------------------------- whole graph run


def test_fake_run_with_the_v1_builder_and_replies(tmp_path):
    script = {1: v1_reply([(0, "narrow_mlp")]), 2: v1_reply([(1, "narrow_mlp")]), 3: {"rollback_to": 1},
              4: v1_reply([(0, "wanda_sparsity")]), 5: {"action": "stop", "rationale": "done"}}
    deps, backend = setup(tmp_path, lambda n, p: script.get(n, v1_reply([(n - 4, "narrow_mlp")])))
    assert deps.build_prompt is I.build_prompt
    state = G.run(deps, run_config())
    log = state["history"]
    assert [r.action for r in log] == ["apply", "apply", "rollback", "apply", "stop", "apply", "apply"]
    assert outcomes(state) == ["ACCEPT", "ACCEPT", None, "INVALID", "INVALID", "ACCEPT", "ACCEPT"]
    # logged in the internal form
    assert log[0].reply.candidates[0].choices[0].strategy == "narrow_mlp"
    assert log[0].rationale == "quick tests" and log[2].reply.rollback_to == 1
    # after the rollback the plan is plan 1 again: only layer 0 narrowed, then layers 2 and 3
    assert [p["mlp_keep"] for p in state["plan"].values()] == [576, 640, 576, 576]
    assert state["stop_reason"] == "memory limit met"
    assert "you returned to kept plan 1" in backend.prompts[3]
    assert "kept, later undone by a return to an earlier plan" in backend.prompts[3]  # round 2's step
    assert "not available in this version" in backend.prompts[5]
    for prompt in backend.prompts:
        assert I.placeholders(prompt) == [] and not re.search(r"\{[a-z_]+[:}]", prompt)
    assert json.loads(json.dumps(G.REPLY_SCHEMA)) == G.REPLY_SCHEMA
