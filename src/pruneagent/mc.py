"""Multiple-choice accuracy on the seven question sets used in the Wanda paper, via lm-evaluation-harness.

Each question is scored by which answer the model gives the highest probability (no examples in the prompt).
The Wanda paper reports plain accuracy (`acc`); length-normalised accuracy (`acc_norm`) is kept alongside it.
"""

TASKS = ("boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa")
# questions per set (the splits lm-eval scores), for the question-weighted average
TASK_SIZES = {"boolq": 3270, "rte": 277, "hellaswag": 10042, "winogrande": 1267, "arc_easy": 2376,
              "arc_challenge": 1172, "openbookqa": 500}


def weighted_accuracy(per_task: dict) -> dict:
    """Share of all questions answered correctly, each question counted once (not a plain mean of the sets),
    with its standard error. `per_task` maps set -> {"acc", "acc_stderr"}, as in `summarize(...)["tasks"]`."""
    n = sum(TASK_SIZES[t] for t in per_task)
    acc = sum(TASK_SIZES[t] * r["acc"] for t, r in per_task.items()) / n
    se = sum((TASK_SIZES[t] * (r.get("acc_stderr") or 0.0)) ** 2 for t, r in per_task.items()) ** 0.5 / n
    return {"acc": acc, "stderr": se, "questions": n}


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


def evaluate(model, tokenizer, device: str, batch_size: int | str = "auto", limit: int | None = None) -> dict:
    """Run the seven tasks with no in-prompt examples; return lm-eval's raw per-task results."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=list(TASKS), num_fewshot=0, limit=limit, log_samples=False)
    return out["results"]


# MMLU: 57 school and professional subjects, four answers each (guessing scores 25%). Scored the same way as the
# seven sets (highest-probability answer, no examples in the prompt). Only for the final run's knowledge check.
MMLU = "mmlu"
MMLU_GROUPS = {"mmlu_humanities": "Humanities", "mmlu_social_sciences": "Social sciences", "mmlu_stem": "Science and maths",
               "mmlu_other": "Other (business, health, ...)"}


def evaluate_mmlu(model, tokenizer, device: str, batch_size: int | str = 8, limit: int | None = None) -> dict:
    """Run MMLU with no in-prompt examples; return per-subject results, questions per subject and the groups."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=[MMLU], num_fewshot=0, limit=limit, log_samples=False)
    return {"results": out["results"], "n": {t: v["effective"] for t, v in out["n-samples"].items()},
            "groups": {g: list(ts) for g, ts in (out.get("group_subtasks") or {}).items()}}


def _weighted(rows: dict, n: dict) -> dict:
    total = sum(n[t] for t in rows)
    acc = sum(n[t] * r["acc"] for t, r in rows.items()) / total
    se = sum((n[t] * (r.get("acc_stderr") or 0.0)) ** 2 for t, r in rows.items()) ** 0.5 / total
    return {"acc": acc, "stderr": se, "questions": total}


def summarize_mmlu(raw: dict) -> dict:
    """Question-weighted accuracy over all subjects (each question counted once), per group, and per subject."""
    n = raw["n"]
    subjects = {t: {"acc": r["acc,none"], "acc_stderr": r.get("acc_stderr,none")}
                for t, r in raw["results"].items() if t in n}
    groups = {}
    for g in MMLU_GROUPS:
        members = [t for t in raw["groups"].get(g, []) if t in subjects]
        if members:
            groups[g] = _weighted({t: subjects[t] for t in members}, n)
    return {"overall": _weighted(subjects, n), "groups": groups, "subjects": subjects,
            "questions": {t: n[t] for t in subjects}}
