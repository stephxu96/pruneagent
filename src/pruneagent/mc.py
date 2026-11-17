"""Multiple-choice accuracy on the seven question sets used in the Wanda paper, via lm-evaluation-harness.

Each question is scored by which answer the model gives the highest probability (no examples in the prompt).
The Wanda paper reports plain accuracy (`acc`); length-normalised accuracy (`acc_norm`) is kept alongside it.
"""

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


def evaluate(model, tokenizer, device: str, batch_size: int | str = "auto", limit: int | None = None) -> dict:
    """Run the seven tasks with no in-prompt examples; return lm-eval's raw per-task results."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=list(TASKS), num_fewshot=0, limit=limit, log_samples=False)
    return out["results"]
