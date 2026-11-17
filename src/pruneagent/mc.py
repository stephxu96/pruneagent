"""Multiple-choice accuracy on the seven question sets used in the Wanda paper, via lm-evaluation-harness.

Each question is scored by which answer the model gives the highest probability (no examples in the prompt).
The Wanda paper reports plain accuracy (`acc`); length-normalised accuracy (`acc_norm`) is kept alongside it.

Per-question results (`evaluate_per_question`) let two models be compared on the same questions: the paired
difference has a much narrower margin of error than the difference of two separately quoted accuracies.
"""

import hashlib
import math

TASKS = ("boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa")


def summarize(results: dict, tasks: tuple[str, ...] = TASKS) -> dict:
    """Pull acc / acc_norm per task out of an lm-eval results dict and add the mean of `acc` over all tasks."""
    per_task = {}
    for task in tasks:
        r = results[task]
        per_task[task] = {
            "acc": r["acc,none"],
            "acc_norm": r.get("acc_norm,none"),
            "acc_stderr": r.get("acc_stderr,none"),
        }
    mean_acc = sum(t["acc"] for t in per_task.values()) / len(per_task)
    return {"tasks": per_task, "mean_acc": mean_acc}


def _run(model, tokenizer, device: str, batch_size: int | str, limit: int | None, log_samples: bool) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
    return lm_eval.simple_evaluate(model=lm, tasks=list(TASKS), num_fewshot=0, limit=limit, log_samples=log_samples)


def evaluate(model, tokenizer, device: str, batch_size: int | str = "auto", limit: int | None = None) -> dict:
    """Run the seven tasks with no in-prompt examples; return lm-eval's raw per-task results."""
    return _run(model, tokenizer, device, batch_size, limit, log_samples=False)["results"]


def evaluate_per_question(model, tokenizer, device: str, batch_size: int | str = "auto",
                          limit: int | None = None) -> tuple[dict, dict]:
    """As `evaluate`, and also return which questions the model got right (see `correctness`)."""
    out = _run(model, tokenizer, device, batch_size, limit, log_samples=True)
    return out["results"], correctness(out["samples"])


def correctness(samples: dict, tasks: tuple[str, ...] = TASKS) -> dict:
    """Per task, from lm-eval's logged samples: right (1) or wrong (0) per question as a string in question order,
    plus a fingerprint of the questions so that two runs can be checked to have scored the same ones."""
    out = {}
    for task in tasks:
        rows = sorted(samples[task], key=lambda s: s["doc_id"])
        out[task] = {
            "n": len(rows),
            "correct": "".join("1" if s["acc"] else "0" for s in rows),
            "questions_sha256": hashlib.sha256("".join(s["doc_hash"] for s in rows).encode()).hexdigest(),
        }
    return out


def paired_difference(a: dict, b: dict, tasks: tuple[str, ...] = TASKS, z: float = 1.96) -> dict:
    """Accuracy of A minus accuracy of B on the same questions, with a 95% interval (normal approximation).

    Per task: the mean of the per-question differences (each -1, 0 or 1), standard error = their standard
    deviation / sqrt(n). Overall: the plain mean over tasks (as the headline mean accuracy is), standard error =
    sqrt(sum of squared task errors) / number of tasks. `a` and `b` are `correctness` outputs."""
    per_task = {}
    for task in tasks:
        ta, tb = a[task], b[task]
        if ta["n"] != tb["n"] or ta["questions_sha256"] != tb["questions_sha256"]:
            raise ValueError(f"{task}: the two runs did not score the same questions")
        n = ta["n"]
        d = [int(x) - int(y) for x, y in zip(ta["correct"], tb["correct"], strict=True)]
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n)
        per_task[task] = {
            "n": n, "acc_a": ta["correct"].count("1") / n, "acc_b": tb["correct"].count("1") / n,
            "diff": mean, "se": se, "ci95": [mean - z * se, mean + z * se],
            "a_only_right": d.count(1), "b_only_right": d.count(-1),
        }
    k = len(tasks)
    mean = sum(t["diff"] for t in per_task.values()) / k
    se = math.sqrt(sum(t["se"] ** 2 for t in per_task.values())) / k
    return {"tasks": per_task, "mean": {"diff": mean, "se": se, "ci95": [mean - z * se, mean + z * se]}}
