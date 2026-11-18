"""Candidate evaluation (task 2.5) on a tiny random Qwen3 (offline, CPU): skipping the unchanged early layers gives
the perplexity of a full run through every layer, before and after the accepted plan changes, in parallel and in
sequence."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pruneagent import evaluate as ev
from pruneagent import graph as G
from pruneagent import plan as P
from pruneagent import pruner
from pruneagent.candidate_eval import CandidateEvaluator, first_change

N_LAYERS, SEQLEN, N_WINDOWS = 5, 16, 4


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=256, num_hidden_layers=N_LAYERS,
                      num_attention_heads=8, num_key_value_heads=4, head_dim=16, max_position_embeddings=64)
    dense = Qwen3ForCausalLM(cfg).eval()
    torch.manual_seed(1)
    scores = pruner.compute_scores(dense, torch.randint(0, 97, (8, SEQLEN)), "cpu")
    ids = torch.randint(0, 97, (1, 10 * SEQLEN + 5))  # 10 windows and a remainder; the gate takes 4
    return dense, scores, ids


def full_ppl(setup, plan):
    """Reference: the pruned model run in full through `evaluate.subset_perplexity`."""
    dense, scores, ids = setup
    return ev.subset_perplexity(pruner.prune(dense, plan, scores), ids, N_WINDOWS, SEQLEN, "cpu").ppl


def evaluator(setup, **kw):
    dense, scores, ids = setup
    return CandidateEvaluator(dense, scores, ids, seqlen=SEQLEN, device="cpu", n_windows=N_WINDOWS, **kw)


def candidates(cfg):
    dense = P.dense_plan(cfg)
    return [
        P.step(dense, 0, "narrow_mlp", cfg),  # changes layer 0: nothing to skip
        P.step(dense, 2, "remove_attention_group", cfg),
        P.step(P.step(dense, 3, "wanda_sparsity", cfg), 4, "narrow_mlp", cfg),
        P.step(dense, 3, "drop_layer", cfg),
        P.step(dense, 4, "drop_layer", cfg),  # the last layer dropped
    ]


def test_first_change():
    cfg = {"num_hidden_layers": 4, "intermediate_size": 640, "num_key_value_heads": 4}
    dense = P.dense_plan(cfg)
    assert first_change(dense, dense) is None
    assert first_change(P.step(dense, 2, "narrow_mlp", cfg), dense) == 2
    dropped = P.step(dense, 1, "drop_layer", cfg)
    other = {**dropped, 1: {**dropped[1], "mlp_keep": 64}}  # a dropped layer's other fields build nothing
    assert first_change(other, dropped) is None


def test_skipping_matches_full_run(setup):
    cfg = setup[0].config
    dense_plan = P.dense_plan(cfg)
    e = evaluator(setup)
    base = e(dense_plan, dense_plan)  # first call: the accepted (unpruned) model in full, saving layer inputs
    assert base.start_layer is None and base.ppl == pytest.approx(full_ppl(setup, dense_plan), rel=1e-6)
    assert sorted(e._saved) == list(range(1, N_LAYERS))  # input to every layer but the first
    for plan, start in zip(candidates(cfg), [0, 2, 3, 3, 4], strict=True):
        r = e(plan, dense_plan)
        assert r.start_layer == start and r.refresh_seconds == 0.0 and r.gpu_seconds > 0
        assert r.ppl == pytest.approx(full_ppl(setup, plan), rel=1e-6)
    assert e.saved_bytes() == (N_LAYERS - 1) * N_WINDOWS * SEQLEN * 64 * 4  # float32 here


def test_skip_off_runs_every_layer(setup):
    cfg = setup[0].config
    e = evaluator(setup, skip=False)
    plan = candidates(cfg)[1]
    r = e(plan, P.dense_plan(cfg))
    assert r.start_layer == 0 and not e._saved
    assert r.ppl == pytest.approx(full_ppl(setup, plan), rel=1e-6)


def test_saved_inputs_follow_accepted_plan_and_rollback(setup):
    cfg = setup[0].config
    dense = P.dense_plan(cfg)
    a1 = P.step(dense, 3, "narrow_mlp", cfg)  # accepted step 1
    a2 = P.step(a1, 1, "remove_attention_group", cfg)  # accepted step 2, changes an earlier layer
    e = evaluator(setup)
    e(dense, dense)
    # after accepting a1, its saved inputs are refreshed from layer 3 on; a candidate on top of it changes layer 4
    cand = P.step(a1, 4, "narrow_mlp", cfg)
    r = e(cand, a1)
    assert r.refresh_seconds > 0 and r.start_layer == 4
    assert r.ppl == pytest.approx(full_ppl(setup, cand), rel=1e-6)
    cand = P.step(a2, 2, "wanda_sparsity", cfg)
    r = e(cand, a2)
    assert r.start_layer == 2 and r.ppl == pytest.approx(full_ppl(setup, cand), rel=1e-6)
    # roll back to a1: the saved inputs are rebuilt for it
    cand = P.step(a1, 2, "narrow_mlp", cfg)
    r = e(cand, a1)
    assert r.refresh_seconds > 0 and r.ppl == pytest.approx(full_ppl(setup, cand), rel=1e-6)
    # the accepted plan itself costs no pass once its inputs are saved
    assert e(a1, a1).ppl == pytest.approx(full_ppl(setup, a1), rel=1e-6)


def test_parallel_and_sequential_give_same_results(setup):
    cfg = setup[0].config
    dense = P.dense_plan(cfg)
    plans = candidates(cfg)[1:4]
    seq = evaluator(setup)
    sequential = [seq(p, dense).ppl for p in plans]
    par = evaluator(setup)
    with ThreadPoolExecutor(3) as pool:  # as the graph's parallel tasks call it: they queue for the device
        parallel = list(pool.map(lambda p: par(p, dense).ppl, plans))
    assert parallel == sequential
    assert sum(r.refresh_seconds > 0 for r in par.log) == 1  # the saved inputs were built once


def test_graph_default_evaluator_is_the_candidate_evaluator(setup):
    dense, scores, ids = setup
    e = G.ppl_evaluator(dense, scores, ids, seqlen=SEQLEN, device="cpu", n_windows=N_WINDOWS)
    assert isinstance(e, CandidateEvaluator) and e.takes_accepted
    plan = candidates(dense.config)[2]
    ppl, seconds = G.measure(e, plan, P.dense_plan(dense.config))
    assert ppl == pytest.approx(full_ppl(setup, plan), rel=1e-6) and seconds > 0
