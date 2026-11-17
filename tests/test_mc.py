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


def _samples(task_bits: dict[str, str], doc_prefix: str = "q") -> dict:
    """lm-eval-style logged samples, deliberately out of question order."""
    return {t: [{"doc_id": i, "doc_hash": f"{doc_prefix}{t}{i}", "acc": float(b)}
                for i, b in reversed(list(enumerate(bits)))] for t, bits in task_bits.items()}


def test_correctness_orders_by_question():
    c = mc.correctness(_samples({"boolq": "1101"}), tasks=("boolq",))
    assert c["boolq"]["n"] == 4
    assert c["boolq"]["correct"] == "1101"


def test_paired_difference_by_task_and_mean():
    tasks = ("t1", "t2")
    a = mc.correctness(_samples({"t1": "1111", "t2": "1100"}), tasks=tasks)
    b = mc.correctness(_samples({"t1": "1100", "t2": "1100"}), tasks=tasks)
    p = mc.paired_difference(a, b, tasks=tasks)
    t1 = p["tasks"]["t1"]
    assert t1["diff"] == pytest.approx(0.5)
    assert (t1["a_only_right"], t1["b_only_right"]) == (2, 0)
    assert t1["se"] == pytest.approx((1 / 3) ** 0.5 / 2)  # differences 0,0,1,1: sample sd sqrt(1/3), n = 4
    assert t1["ci95"][0] == pytest.approx(0.5 - 1.96 * t1["se"])
    assert p["tasks"]["t2"]["diff"] == 0 and p["tasks"]["t2"]["se"] == 0
    assert p["mean"]["diff"] == pytest.approx(0.25)
    assert p["mean"]["se"] == pytest.approx(t1["se"] / 2)


def test_paired_difference_equals_difference_of_mean_accuracies():
    tasks = ("t1", "t2")
    a = mc.correctness(_samples({"t1": "10110", "t2": "011"}), tasks=tasks)
    b = mc.correctness(_samples({"t1": "01100", "t2": "111"}), tasks=tasks)
    p = mc.paired_difference(a, b, tasks=tasks)
    mean_a = sum(t["acc_a"] for t in p["tasks"].values()) / 2
    mean_b = sum(t["acc_b"] for t in p["tasks"].values()) / 2
    assert p["mean"]["diff"] == pytest.approx(mean_a - mean_b)


def test_paired_difference_rejects_different_questions():
    a = mc.correctness(_samples({"t1": "10"}), tasks=("t1",))
    b = mc.correctness(_samples({"t1": "10"}, doc_prefix="other"), tasks=("t1",))
    with pytest.raises(ValueError, match="same questions"):
        mc.paired_difference(a, b, tasks=("t1",))
