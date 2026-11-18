"""State types (task 2.1): JSON round-trip and schema validation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pruneagent import estimate as E
from pruneagent import plan as P
from pruneagent import schema as S

ROOT = Path(__file__).resolve().parents[1]
CFG = {"num_hidden_layers": 28, "intermediate_size": 3072, "num_key_value_heads": 8, "num_attention_heads": 16,
       "hidden_size": 1024, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": True}
PLAN = P.apply_steps(P.dense_plan(CFG), [{"layer": 3, "strategy": "drop_layer"},
                                         {"layer": 5, "strategy": "narrow_mlp"}], CFG)
PHONE = E.DeviceTarget("phone (example)", memory_limit_gb=1.1, bandwidth_gb_s=50, compute_tflops=2)


def probe_results():
    return json.loads((ROOT / "results/probes/qwen3-0.6b-base__cuda__float16.json").read_text())


def run():
    return S.RunConfig(model_id="Qwen/Qwen3-0.6B-Base", device=PHONE)


def baseline():
    return S.Baseline(memory_limit_gb=1.1, ratio=0.12, plan=P.normalize({7: {"mlp_keep": 2001}}, CFG),
                      weight_bytes=1_072_000_000, val_ppl_subset=16.2)


def reply():
    return S.AgentReply(action="apply", rationale="cheap layers first", candidates=[
        S.Candidate(choices=[S.Choice(layer=3, strategy="drop_layer"), S.Choice(layer=5, strategy="narrow_mlp")],
                    rationale="drop the layer with the least damage")])


def candidate_result():
    return S.CandidateResult(candidate_id=0, choices=reply().candidates[0].choices, plan=PLAN,
                             estimate=E.estimate(CFG, PLAN, PHONE), rank_decode_tok_s=40.0, val_ppl=15.1,
                             step_rise=0.03)


def step_record():
    check = S.CheckResult(ok=True, candidates=[S.CandidateCheck(index=0, ok=True, plan=PLAN,
                                                                weight_bytes=E.weight_bytes(CFG, PLAN))])
    return S.StepRecord(round=1, action="apply", rationale="cheap layers first", reply=reply(), check=check,
                        candidates=[candidate_result()], gate=S.GateResult(decision="ACCEPT", chosen_candidate=0),
                        outcome="ACCEPT", strategies_used={"drop_layer": 1, "narrow_mlp": 1}, llm_tokens=20_000,
                        llm_seconds=11.2, gpu_seconds=40.0)


def examples():
    table = S.SensitivityTable.from_probe_results(probe_results())
    r, b = run(), baseline()
    scores = S.FinalScores(ppl_test=15.0, tasks={"boolq": 0.6, "rte": 0.5}, mmlu=0.4, memory=[
        S.MemoryReport(machine="m4", params=500_000_000, file_size_mb=1070, peak_memory_mb=1500, prefill_tok_s=900,
                       decode_tok_s=30)])
    return [
        r,
        S.RunConfig(model_id="m", device=E.PRESETS["t4"], agent_backend="api", agent_provider="anthropic",
                    agent_model="pinned", rules=S.Rules(max_candidates=1), layer_cost_s=1e-4),
        b,
        table,
        reply(),
        S.AgentReply(action="probe", rationale="look", probe=[S.Choice(layer=2, strategy="wanda_sparsity")]),
        S.AgentReply(action="rollback", rationale="back", rollback_to=0),
        S.AgentRequest(objective="fit", device=PHONE, strategy_menu={s: f"one step of {s}" for s in P.STRATEGIES},
                       rules=S.Rules(), sensitivity=table, plan=PLAN, dense_weight_bytes=1_192_099_840,
                       current_weight_bytes=1_150_000_000, limit_bytes=1_100_000_000, current_ppl=14.0,
                       last_accepted_ppl=14.0, baseline_ppl=16.2,
                       recent=[S.StepSummary(round=0, action="probe", rationale="r", outcome="probed 4 layers")]),
        S.CheckResult(ok=False, reasons=["candidate 1 has no choices; give at least one."],
                      candidates=[S.CandidateCheck(index=0, ok=False, reasons=["candidate 1 has no choices"])]),
        candidate_result(),
        S.GateResult(decision="ROLLBACK"),
        S.AcceptedPlan(round=1, plan=PLAN, val_ppl=15.1, weight_bytes=E.weight_bytes(CFG, PLAN)),
        step_record(),
        S.DeviceWarning(kind="out_of_reach", message="limit below the smallest allowed model", closest_plan=PLAN,
                        memory_gap_gb=0.2),
        scores,
        S.Report(run=r, baseline=b, final_plan=PLAN, methods=[
            S.MethodResult(name="agent", plan=PLAN, weight_bytes=1_090_000_000, scores=scores),
            S.MethodResult(name="Baseline: FLAP", plan=b.plan, weight_bytes=b.weight_bytes)],
            device=S.DeviceReport(target=PHONE, fits_memory=True, speed_reported=False),
            warnings=[], strategy_counts={"drop_layer": {"proposed": 2, "accepted": 1}}, log=[step_record()],
            usage=S.Usage(calls=3, tokens=60_000, seconds=33.0)),
    ]


@pytest.mark.parametrize("obj", examples(), ids=lambda o: type(o).__name__)
def test_json_round_trip(obj):
    text = obj.model_dump_json()
    back = type(obj).model_validate_json(text)
    assert back == obj
    assert back.model_dump_json() == text
    json.loads(text)  # plain JSON


def test_plan_keys_are_ints_after_json():
    back = S.AcceptedPlan.model_validate_json(S.AcceptedPlan(round=0, plan=PLAN, val_ppl=1.5,
                                                             weight_bytes=1).model_dump_json())
    assert sorted(back.plan) == list(range(28)) and back.plan == PLAN


# ---------------------------------------------------------------------------------------------- validation


def test_memory_limit_is_required():
    t4 = {"name": "t4", "bandwidth_gb_s": 320, "compute_tflops": 65}
    with pytest.raises(ValidationError, match="memory_limit_gb"):
        S.RunConfig(model_id="m", device=t4)
    with pytest.raises(ValidationError, match="memory_limit_gb must be positive"):
        S.RunConfig(model_id="m", device=t4 | {"memory_limit_gb": 0})
    with pytest.raises(ValidationError, match="device"):
        S.RunConfig(model_id="m")
    cfg = S.RunConfig(model_id="m", device=t4 | {"memory_limit_gb": 1.0729})
    assert cfg.limit_bytes == 1_072_900_000 and cfg.calibration == S.Calibration(dataset="c4", n_samples=128,
                                                                                  seqlen=2048)


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError, match="max_candidate"):
        S.RunConfig.model_validate(run().model_dump() | {"max_candidate": 5})
    with pytest.raises(ValidationError):
        S.Rules(max_drops=3)


def test_cost_cap_only_for_api():
    assert S.RunConfig(model_id="m", device=PHONE, agent_backend="api").max_cost_usd == 5.0
    with pytest.raises(ValidationError, match="API backend only"):
        S.RunConfig(model_id="m", device=PHONE, max_cost_usd=5.0)
    with pytest.raises(ValidationError):
        S.RunConfig(model_id="m", device=PHONE, agent_backend="openai")


def test_baseline_must_fit_its_limit():
    with pytest.raises(ValidationError, match="over its memory limit"):
        S.Baseline(memory_limit_gb=1.0, ratio=0.1, plan=PLAN, weight_bytes=1_000_000_001)
    with pytest.raises(ValidationError):
        S.Baseline(method="wanda_sp", memory_limit_gb=1.0, ratio=0.1, plan=PLAN, weight_bytes=1)


@pytest.mark.parametrize("bad", [
    {"action": "apply", "rationale": "r", "candidates": [[{"layer": 1, "strategy": "narrow_mlp", "amount": 0.2}]]},
    {"action": "apply", "rationale": "r", "candidates": [[{"layer": 1, "strategy": "prune_heads"}]]},
    {"action": "apply", "rationale": "r", "candidates": [[{"layer": -1, "strategy": "narrow_mlp"}]]},
    {"action": "apply", "rationale": "r", "candidates": [{"choices": [], "ratio": 0.2}]},
    {"action": "grow", "rationale": "r"},
    {"action": "stop"},
])
def test_agent_reply_rejects_amounts_and_unknowns(bad):
    with pytest.raises(ValidationError):
        S.AgentReply.model_validate(bad)


def test_agent_reply_reads_bare_candidate_lists():
    r = S.AgentReply.model_validate({"action": "apply", "rationale": "r", "candidates": [
        [{"layer": 21, "strategy": "narrow_mlp"}, {"layer": 25, "strategy": "drop_layer"}]]})
    assert r.candidates[0] == S.Candidate(choices=[S.Choice(layer=21, strategy="narrow_mlp"),
                                                   S.Choice(layer=25, strategy="drop_layer")], rationale="")


def test_agent_reply_json_schema_has_no_amounts():
    schema = S.AgentReply.model_json_schema()
    choice = schema["$defs"]["Choice"]
    assert set(choice["properties"]) == {"layer", "strategy"} and choice["additionalProperties"] is False
    assert set(choice["properties"]["strategy"]["enum"]) == set(P.STRATEGIES)


def test_round1_reply_parses():
    rec = json.loads((ROOT / "results/agent_round/round1.json").read_text())
    r = S.AgentReply.model_validate({"action": "apply", "rationale": "round 1", "candidates": rec["candidates"]})
    assert [len(c.choices) for c in r.candidates] == [len(c["choices"]) for c in rec["candidates"]]


def test_check_result_is_consistent():
    with pytest.raises(ValidationError, match="must give its reasons"):
        S.CheckResult(ok=False)
    with pytest.raises(ValidationError, match="no failing candidate"):
        S.CheckResult(ok=True, candidates=[S.CandidateCheck(index=0, ok=False, reasons=["x"])])


def test_gate_names_the_chosen_candidate_only_on_accept():
    with pytest.raises(ValidationError):
        S.GateResult(decision="ACCEPT")
    with pytest.raises(ValidationError):
        S.GateResult(decision="ROLLBACK", chosen_candidate=1)


def test_unmeasured_device_reports_memory_fit_only():
    with pytest.raises(ValidationError, match="memory fit only"):
        S.DeviceReport(target=PHONE, fits_memory=True, speed_reported=False, est_decode_tok_s=50.0)
    assert S.DeviceReport(target=E.PRESETS["m4"], fits_memory=True, speed_reported=True, est_decode_tok_s=50.0)


def test_report_enforces_the_fair_test():
    r = run()
    other_calib = baseline().model_copy(update={"calibration": S.Calibration(dataset="wikitext2")})
    with pytest.raises(ValidationError, match="one set for every method"):
        S.Report(run=r, baseline=other_calib, final_plan=PLAN, methods=[],
                 device=S.DeviceReport(target=PHONE, fits_memory=True, speed_reported=False))
    other_limit = baseline().model_copy(update={"memory_limit_gb": 1.2})
    with pytest.raises(ValidationError, match="run's memory limit"):
        S.Report(run=r, baseline=other_limit, final_plan=PLAN, methods=[],
                 device=S.DeviceReport(target=PHONE, fits_memory=True, speed_reported=False))


def test_sensitivity_table_from_probe_results():
    t = S.SensitivityTable.from_probe_results(probe_results())
    assert sorted(t.rows) == list(range(28)) and t.measured_windows == 128 and t.dense_ppl > 13
    row = t.rows[3]
    assert set(row.measured_ppl) == set(P.STRATEGIES)
    assert {(q.strategy, q.steps) for q in row.quick_tests} >= {("narrow_mlp", 1), ("drop_layer", 1)}
    with pytest.raises(ValidationError, match="row key 1 holds layer 0"):
        S.SensitivityTable(rows={1: t.rows[0]})


def test_reducers():
    t = S.SensitivityTable.from_probe_results(probe_results())
    a = S.SensitivityTable(rows={0: t.rows[0], 1: t.rows[1]})
    b = S.SensitivityTable(rows={1: t.rows[2].model_copy(update={"layer": 1}), 2: t.rows[2]})
    merged = S.merge_sensitivity(a, b)
    assert sorted(merged.rows) == [0, 1, 2] and merged.rows[1] == b.rows[1]
    assert S.merge_sensitivity(None, b) is b
    assert S.add_usage(S.Usage(calls=1, tokens=5), S.Usage(calls=2, cost_usd=0.1)) == S.Usage(calls=3, tokens=5,
                                                                                               cost_usd=0.1)


def test_prune_state_builds_a_langgraph_graph():
    from langgraph.graph import StateGraph

    g = StateGraph(S.PruneState)
    assert {"plan", "candidates", "history", "usage", "sensitivity"} <= set(g.channels)
