import pytest

from pruneagent import mc


def test_summarize_mean_of_acc():
    raw = {t: {"acc,none": 0.1 * (i + 1), "acc_norm,none": 0.5, "acc_stderr,none": 0.01} for i, t in enumerate(mc.TASKS)}
    s = mc.summarize(raw)
    assert set(s["tasks"]) == set(mc.TASKS)
    assert s["mean_acc"] == pytest.approx(0.4)
    assert s["tasks"]["boolq"]["acc_norm"] == 0.5


def test_summarize_missing_acc_norm():
    raw = {t: {"acc,none": 0.5} for t in mc.TASKS}
    s = mc.summarize(raw)
    assert s["tasks"]["boolq"]["acc_norm"] is None
    assert s["mean_acc"] == pytest.approx(0.5)


def test_mmlu_counts_each_question_once_overall_and_per_group():
    raw = {
        "results": {"mmlu": {"acc,none": 0.9}, "mmlu_stem": {"acc,none": 0.9},
                    "mmlu_a": {"acc,none": 0.5, "acc_stderr,none": 0.1}, "mmlu_b": {"acc,none": 1.0, "acc_stderr,none": 0.0},
                    "mmlu_c": {"acc,none": 0.25, "acc_stderr,none": 0.05}},
        "n": {"mmlu_a": 100, "mmlu_b": 300, "mmlu_c": 100},
        "groups": {"mmlu_stem": ["mmlu_a", "mmlu_b"], "mmlu_humanities": ["mmlu_c"]},
    }
    s = mc.summarize_mmlu(raw)
    assert s["overall"]["questions"] == 500 and s["overall"]["acc"] == pytest.approx((50 + 300 + 25) / 500)
    assert s["groups"]["mmlu_stem"]["acc"] == pytest.approx(350 / 400)  # not the plain mean 0.75
    assert s["groups"]["mmlu_humanities"]["acc"] == pytest.approx(0.25)
    assert set(s["subjects"]) == {"mmlu_a", "mmlu_b", "mmlu_c"}  # group rows are not subjects
