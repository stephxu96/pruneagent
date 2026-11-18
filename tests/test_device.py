"""Device target (task 2.11): early warning, stop at the limit, final warnings and the device report. Fakes only:
the fake agent, evaluator and Baseline: FLAP of test_graph; no model, GPU or real agent call."""

import pytest
from pydantic import ValidationError
from test_graph import CFG, DENSE, LIMIT, NARROW, outcomes, reply, setup

from pruneagent import device as D
from pruneagent import estimate as E
from pruneagent import graph as G
from pruneagent import plan as P
from pruneagent.estimate import DeviceTarget
from pruneagent.schema import DeviceReport, Rules, RunConfig

SMALLEST = E.weight_bytes(CFG, D.smallest_plan(CFG, Rules()))
T4 = E.PRESETS["t4"].name  # a device whose speed estimate was checked against measurements


def target(limit_bytes, name="test", **kw):
    return DeviceTarget(name, memory_limit_gb=limit_bytes / 1e9, bandwidth_gb_s=320, compute_tflops=65, **kw)


def run_config(limit_bytes=LIMIT, **kw):
    return RunConfig(model_id="tiny", device=target(limit_bytes, **kw))


def phone(limit_bytes, **kw):
    return DeviceTarget("phone (example)", memory_limit_gb=limit_bytes / 1e9, bandwidth_gb_s=55, compute_tflops=2,
                        **kw)


# ------------------------------------------------------------------------------------------ feasibility


def test_reachable_limit_gives_no_early_warning():
    assert D.feasibility(CFG, run_config()) == ([], None)


def test_unreachable_limit_warns_with_the_smallest_model_and_the_gap():
    (w,), reason = D.feasibility(CFG, run_config(SMALLEST - 5_000))
    assert reason == "memory limit out of reach"
    assert w.kind == "out_of_reach" and w.closest_plan == D.smallest_plan(CFG, Rules())
    assert w.memory_gap_gb == pytest.approx(5_000 / 1e9) and w.speed_gap_tok_s == 0


def test_limit_exactly_at_the_smallest_model_is_reachable():
    assert D.feasibility(CFG, run_config(SMALLEST)) == ([], None)


def test_already_met_limit_warns_and_stops():
    (w,), reason = D.feasibility(CFG, run_config(DENSE))
    assert w.kind == "already_met" and reason == "memory limit already met by the unpruned model"
    assert "no pruning is needed" in w.message


def test_speed_out_of_reach_warns_early_but_the_search_still_runs():
    small = D.smallest_plan(CFG, Rules())
    fastest = E.check(target(LIMIT, T4), CFG, small).decode_tok_s
    (w,), reason = D.feasibility(CFG, run_config(LIMIT, name=T4, min_decode_tok_s=fastest + 10))
    assert reason is None
    assert w.kind == "out_of_reach" and w.memory_gap_gb == 0 and w.speed_gap_tok_s == pytest.approx(10)


# --------------------------------------------------------------------------------------- whole runs


def test_reachable_target_stops_when_the_weights_fit(tmp_path):
    deps, _ = setup(tmp_path, lambda n, p: reply([(n - 1, "narrow_mlp")]))
    state = G.run(deps, run_config())
    assert outcomes(state) == ["ACCEPT"] * 3  # stops right after the third step, not at the round cap
    assert state["stop_reason"] == "memory limit met" and state["warnings"] == []
    r = state["device_report"]
    assert r.fits_memory and r.memory_gap_gb == 0 and r.closest_plan is None
    assert r.weight_bytes == E.weight_bytes(CFG, state["plan"]) <= LIMIT
    assert r.kv_context == 2048 and r.kv_cache_bytes == E.kv_cache_bytes(CFG, state["plan"], 2048)
    assert not r.speed_reported and r.est_decode_tok_s is None  # "test" is not a measured device


def test_unreachable_target_reports_the_closest_plan_and_the_gap(tmp_path):
    deps, backend = setup(tmp_path, lambda n, p: reply([(0, "narrow_mlp")]))
    state = G.run(deps, run_config(SMALLEST - 5_000))
    assert backend.prompts == []  # warned before any agent call
    r = state["device_report"]
    assert not r.fits_memory and r.memory_gap_gb == pytest.approx(5_000 / 1e9)
    assert r.closest_plan == D.smallest_plan(CFG, Rules())


def test_already_met_target_reports_the_unpruned_model(tmp_path):
    deps, backend = setup(tmp_path, lambda n, p: reply([(0, "narrow_mlp")]))
    state = G.run(deps, run_config(DENSE + 1))
    assert backend.prompts == [] and [w.kind for w in state["warnings"]] == ["already_met"]
    r = state["device_report"]
    assert r.fits_memory and r.weight_bytes == DENSE and r.closest_plan is None


def test_search_that_misses_the_limit_ends_with_the_closest_plan(tmp_path):
    far = DENSE - 20 * NARROW
    deps, _ = setup(tmp_path, lambda n, p: reply([((n - 1) % 4, "narrow_mlp")]))
    state = G.run(deps, RunConfig(model_id="tiny", device=target(far), max_agent_calls=2))
    (w,) = state["warnings"]
    size = E.weight_bytes(CFG, state["plan"])
    assert w.kind == "out_of_reach" and w.closest_plan == state["plan"]
    assert w.memory_gap_gb == pytest.approx((size - far) / 1e9) == state["device_report"].memory_gap_gb
    assert state["device_report"].closest_plan == state["plan"]


def test_measured_device_reports_speed_and_the_gap_to_a_minimum(tmp_path):
    deps, _ = setup(tmp_path, lambda n, p: reply([(n - 1, "narrow_mlp")]))
    run = run_config(name=T4, min_decode_tok_s=1e9)  # unreachable speed, reachable memory
    state = G.run(deps, run)
    assert state["stop_reason"] == "memory limit met"  # the stop is on memory only
    r = state["device_report"]
    chk = E.check(run.device, CFG, state["plan"])
    assert r.speed_reported and r.est_decode_tok_s == pytest.approx(chk.decode_tok_s)
    assert r.est_decode_tok_s < chk.estimate.est_decode_tok_s  # the per-layer cost is included
    assert r.speed_gap_tok_s == pytest.approx(1e9 - r.est_decode_tok_s)
    kinds = [(w.kind, w.memory_gap_gb) for w in state["warnings"]]
    assert kinds == [("out_of_reach", 0), ("out_of_reach", 0)]  # early (speed) and final (speed)
    assert state["warnings"][-1].speed_gap_tok_s == pytest.approx(r.speed_gap_tok_s)


def test_below_flap_warning_carries_the_speed_gap():
    run = run_config(DENSE, name=T4, min_decode_tok_s=1e9)  # the plan fits; quality and speed fall short
    plan = P.dense_plan(CFG)
    (w,) = D.final_warnings(CFG, run, plan, val_ppl=13.0, baseline_ppl=12.0)
    assert w.kind == "below_flap" and w.speed_gap_tok_s > 0 and "minimum speed" in w.message
    assert D.final_warnings(CFG, run_config(DENSE), plan, val_ppl=11.0, baseline_ppl=12.0) == []


# ------------------------------------------------------------------------------------------ phone target


def test_phone_target_gets_memory_fit_only(tmp_path):
    deps, _ = setup(tmp_path, lambda n, p: reply([(n - 1, "narrow_mlp")]))
    run = RunConfig(model_id="tiny", device=phone(LIMIT))
    assert run.layer_cost == 0  # no measurements to fit a per-layer cost on
    state = G.run(deps, run)
    r = state["device_report"]
    assert r.fits_memory and not r.speed_reported
    assert r.est_decode_tok_s is None and r.speed_gap_tok_s is None
    assert r.kv_cache_bytes == E.kv_cache_bytes(CFG, state["plan"], 2048)


def test_minimum_speed_is_refused_for_a_phone():
    with pytest.raises(ValidationError, match="only on the M4 or the T4"):
        RunConfig(model_id="tiny", device=phone(LIMIT, min_decode_tok_s=20))
    assert RunConfig(model_id="tiny", device=target(LIMIT, T4, min_decode_tok_s=20)).layer_cost > 0


def test_speed_fields_are_refused_for_an_unverified_device():
    with pytest.raises(ValidationError, match="only for the M4 and the T4"):
        DeviceReport(target=phone(LIMIT), fits_memory=True, speed_reported=True, est_decode_tok_s=30.0)
    with pytest.raises(ValidationError, match="memory fit only"):
        DeviceReport(target=phone(LIMIT), fits_memory=True, speed_reported=False, speed_gap_tok_s=1.0)
    assert DeviceReport(target=target(LIMIT, "t4"), fits_memory=True, speed_reported=True, est_decode_tok_s=30.0)


def test_layer_cost_override_and_default():
    assert RunConfig(model_id="tiny", device=target(LIMIT, "m4")).layer_cost == E.LAYER_COST_S["m4"]
    assert RunConfig(model_id="tiny", device=target(LIMIT, "m4"), layer_cost_s=0).layer_cost == 0
