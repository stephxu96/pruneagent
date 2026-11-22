"""The checks on every request the web app accepts, and the progress of a model setup.

The browser never sends a command line: each request is a few fields, checked here against the same rules as the
command line, then handed to the background workflow service (`pruneagent.service.workflows`), which runs it.
"""

from __future__ import annotations

import re

from pruneagent import models as MD
from pruneagent import search as S

MODELS = MD.BUILT_IN
MAX_CUT = 0.6  # largest size cut the app offers; the run's own early warning says if it is reachable
RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
GPU = ["scripts/mac_gpu.sh"]
UV = ["uv", "run"]


class JobError(ValueError):
    """A job request that does not pass the checks."""


def _run_id(run_id: str) -> str:
    if not RUN_ID.match(run_id or ""):
        raise JobError("not a run id")
    return run_id


def command(kind: str, params: dict, known: tuple[str, ...] | list[str] = MODELS) -> list[str]:
    """The command line for one job. Only these kinds exist; every field is checked. `known` is the models a run
    may use (the built-in ones plus those the user added)."""
    if kind == "prepare":
        model = params.get("model", "")
        if not MD.valid_id(model):
            raise JobError("not a Hugging Face model name")
        return [*GPU, "caffeinate", "-i", *UV, "pruneagent", "prepare", model]
    if kind in ("start", "precheck"):
        model = params.get("model", MODELS[0])
        if model not in known:
            raise JobError(f"model must be one of {', '.join(known)}")
        try:
            cut = float(params["cut"])
        except (KeyError, TypeError, ValueError):
            raise JobError("cut must be a number, e.g. 0.10") from None
        if not 0.01 <= cut <= MAX_CUT:
            raise JobError(f"cut must be between 1% and {MAX_CUT:.0%}")
        backend = params.get("backend", "codex")
        if backend not in ("codex", "api"):
            raise JobError("backend must be codex or api")
        search = params.get("search", S.DEFAULT)
        if search not in S.STRATEGIES:
            raise JobError(f"search must be one of {', '.join(S.STRATEGIES)}")
        args = ["pruneagent", "run", "--model", model, "--memory-cut", f"{cut:.4f}", "--device", "mps",
                "--backend", backend, "--search", search]
        if kind == "precheck":
            return [*UV, *args, "--dry-run", "--json-result"]
        return [*GPU, "caffeinate", "-i", *UV, *args]
    if kind == "resume":
        return [*GPU, "caffeinate", "-i", *UV, "pruneagent", "resume", _run_id(params.get("run_id", ""))]
    if kind == "save":
        return [*UV, "pruneagent", "save", _run_id(params.get("run_id", ""))]
    if kind in CHECK_SCRIPTS:
        script = f"scripts/{CHECK_SCRIPTS[kind]}"
        return [*GPU, *UV, "python", script, f"runs/{_run_id(params.get('run_id', ''))}"]
    raise JobError(f"unknown job kind: {kind}")


CHECK_SCRIPTS = {"heldout": "heldout_check.py", "accuracy": "accuracy_check.py", "mmlu": "mmlu_check.py"}
PREPARE_STEPS = 4
_STEP = re.compile(r"Step (\d+) of (\d+): (.+)")
_ENTRY = re.compile(r"\[(\d+)/(\d+)\]")


def prepare_progress(text: str) -> dict:
    """How far a model setup is, from its output: the step (1 load the model, 2 sample text, 3 part usage, 4 cut
    tests) and, in step 4, the cut tests measured. `share` is the whole setup's share done (step 4 is most of the
    time)."""
    steps = _STEP.findall(text)
    step = int(steps[-1][0]) if steps else 0
    label = steps[-1][2].strip() if steps else "starting"
    done = total = 0
    if step == PREPARE_STEPS:
        tail = text[text.rfind(f"Step {PREPARE_STEPS} of"):]
        entries = _ENTRY.findall(tail)
        if entries:
            done, total = int(entries[-1][0]), int(entries[-1][1])
    # share of the whole setup done when each step starts (the cut tests are most of the time)
    weight = {0: 0.0, 1: 0.0, 2: 0.03, 3: 0.06, 4: 0.1}[min(step, 4)]
    share = weight + (0.9 * done / total if total else 0.0)
    return {"step": step, "of": PREPARE_STEPS, "label": label, "done": done, "total": total, "share": min(share, 1.0)}
