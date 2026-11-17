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
