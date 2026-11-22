"""`pruneagent run`: one real agent run, end to end (the graph of `graph.py` with real parts).

    scripts/mac_gpu.sh caffeinate -i uv run pruneagent run --model Qwen/Qwen3-0.6B-Base --memory-cut 0.10 \\
        --device mps --backend codex

What the run is built from (`build_deps`):
- the unpruned model in `RunConfig.dtype` (float16) on the device;
- the pruning scores from the shared C4 calibration set (128 x 2,048 tokens), cached in `checkpoints/scores/`
  (computed and saved first if missing, on the GPU);
- the sensitivity table: quick tests from `results/probes/`, plus the measured per-layer damage from
  `results/damage_table/` when that file exists (otherwise the run goes ahead without it, with a warning);
- the real candidate evaluator (`candidate_eval.CandidateEvaluator`) on the gate's WikiText-2 validation windows;
- Baseline: FLAP cut to the memory limit (`baselines.flap_at_memory_limit`) and its perplexity on the same windows;
- the agent client (`agent_client.AgentClient`: Codex by default, with its usage limits and pause at 90%);
- the prompt builder: the v1 instructions (`PROMPT_BUILDER`, task 2.9) when they are in the code, otherwise the
  graph's placeholder prompt, with a warning.

Resume: the run records `FACTORY` and writes the paths it used to `runs/<run-id>/setup.json`, so `pruneagent
resume` rebuilds the same parts from the run directory alone.

Dry run (`dry_run`): builds the same parts with the model on the CPU, never computes scores, and renders round
1's prompt by running the graph up to the first agent call, with stand-in perplexities for the unpruned model and
Baseline: FLAP. No agent call and no GPU work; nothing is written under `runs/`.

On the Mac the GPU is shared: a real run on `mps` refuses to start unless `scripts/mac_gpu.sh` holds the GPU lock.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pruneagent import agent_client as AC
from pruneagent import estimate as E
from pruneagent import graph as G
from pruneagent import plan as P
from pruneagent import runstore
from pruneagent.schema import Baseline, Rules, RunConfig, SensitivityTable

FACTORY = "pruneagent.realrun:deps_from_run_dir"
SETUP_FILE = "setup.json"
PROMPT_BUILDER = "pruneagent.instructions:build_prompt"  # the v1 instructions (task 2.9)
SCORES_DIR = Path("checkpoints/scores")
PROBES_DIR = Path("results/probes")
DAMAGE_DIR = Path("results/damage_table")
DAMAGE_SUFFIX = "__c4.json"  # damage tables measured on the C4 gate passages (the WikiText ones end in plain .json)
DEVICES_DIR = Path("results/speed/devices")
TARGET_FOR_DEVICE = {"mps": "m4", "cuda": "t4", "cpu": "m4"}  # the device target whose speed estimate applies
DRY_DENSE_PPL, DRY_FLAP_PPL = 10.0, 11.0  # stand-ins in a dry run's prompt (nothing is measured)

Notify = Callable[[str], None]


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


class SetupError(Exception):
    """The run cannot be built: a missing file, a wrong calibration set, or the Mac GPU lock not held."""


# ---------------------------------------------------------------------------------------------- settings


@dataclass
class Setup:
    """Where a run's parts come from, beyond its `RunConfig`; saved as `runs/<run-id>/setup.json`."""

    device: str  # torch device: mps, cuda or cpu
    scores: str  # cached scores file (computed there if missing)
    probes: str  # quick-test file
    damage: str  # measured per-layer damage file (used if it exists)
    data_dir: str | None = None  # data cache (None: $PRUNEAGENT_CACHE or ./data_cache)
    revision: str | None = None  # model revision (the tokenizer revision recorded in the data cache)
    prompt_builder: str | None = PROMPT_BUILDER  # "module:function"; None: the graph's placeholder

    def save(self, run_dir: Path) -> None:
        (Path(run_dir) / SETUP_FILE).write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, run_dir: Path) -> Setup:
        return cls(**json.loads((Path(run_dir) / SETUP_FILE).read_text()))


def _slug(model_id: str) -> str:
    from pruneagent import data as d

    return d.slug(model_id)


def _find(directory: Path, model_id: str, device: str, dtype: str, suffix: str) -> Path:
    """`<dir>/<model>__<device>__<dtype><suffix>`, else the same model and dtype from another device, else the
    preferred name (which may not exist)."""
    preferred = directory / f"{_slug(model_id)}__{device}__{dtype}{suffix}"
    if preferred.exists():
        return preferred
    others = sorted(directory.glob(f"{_slug(model_id)}__*__{dtype}{suffix}"))
    return others[0] if others else preferred


def make_setup(model_id: str, device: str, dtype: str = "float16", data_dir: Path | None = None,
               scores: Path | None = None, probes: Path | None = None, damage: Path | None = None,
               prompt_builder: str | None = PROMPT_BUILDER) -> Setup:
    """Default paths for a model, device and dtype; every path stored absolute, so a resume works from anywhere."""
    from pruneagent import data as d

    data_dir = Path(data_dir) if data_dir else d.cache_dir()
    entry = d.read_manifest(data_dir)["entries"].get(d.calib_name(model_id), {})
    scores = scores or SCORES_DIR / f"{_slug(model_id)}__{device}__{dtype}.safetensors"
    probes = probes or _find(PROBES_DIR, model_id, device, dtype, ".json")
    damage = damage or _find(DAMAGE_DIR, model_id, device, dtype, DAMAGE_SUFFIX)
    return Setup(device=device, scores=str(Path(scores).resolve()), probes=str(Path(probes).resolve()),
                 damage=str(Path(damage).resolve()), data_dir=str(data_dir.resolve()),
                 revision=(entry.get("tokenizer") or {}).get("revision"), prompt_builder=prompt_builder)


def device_target(device: str, target: str | None = None) -> E.DeviceTarget:
    """The device target for the speed estimate: `target` (a preset, a JSON file or JSON), else the measured
    device file for this device (results/speed/devices/m4.json on the Mac), else its datasheet preset."""
    if target:
        return E.load_target(target)
    name = TARGET_FOR_DEVICE.get(device, "m4")
    measured = DEVICES_DIR / f"{name}.json"
    return E.load_target(str(measured) if measured.exists() else name)


def make_run_config(model_id: str, config, device: str, *, memory_limit_mb: float | None = None,
                    memory_cut: float | None = None, backend: str = "codex", max_rounds: int | None = None,
                    dtype: str = "float16", target: str | None = None, **kw) -> RunConfig:
    """The run's settings. The memory limit (weights only) is `memory_limit_mb` (MB = 1e6 bytes) or `memory_cut`
    less than the unpruned model's weights; exactly one of the two."""
    if (memory_limit_mb is None) == (memory_cut is None):
        raise SetupError("give exactly one of --memory-limit-mb and --memory-cut")
    if memory_cut is not None:
        if not 0 < memory_cut < 1:
            raise SetupError(f"--memory-cut is a fraction between 0 and 1, not {memory_cut}")
        limit_bytes = int(E.weight_bytes(config, P.dense_plan(config)) * (1 - memory_cut))
    else:
        if memory_limit_mb <= 0:
            raise SetupError(f"--memory-limit-mb must be positive, not {memory_limit_mb}")
        limit_bytes = int(memory_limit_mb * 1e6)
    dev = dataclasses.replace(device_target(device, target), memory_limit_gb=limit_bytes / 1e9)
    rules = Rules(max_rounds=max_rounds) if max_rounds else Rules()
    agent = {"agent_model": AC.CODEX_MODEL, "agent_reasoning_effort": AC.CODEX_REASONING_EFFORT}
    return RunConfig(model_id=model_id, device=dev, rules=rules, agent_backend=backend, dtype=dtype, gpu=device,
                     **{"gate_text": "c4", **agent, **kw})


# ------------------------------------------------------------------------------------------------ parts


def load_sensitivity(probes: Path, damage: Path | None, notify: Notify = _stderr) -> SensitivityTable:
    """Quick tests from a `pruneagent probe` file, plus the measured per-layer damage when its file exists. The
    probe file's own few whole-model measurements are left out: they were taken on other windows and hardware."""
    probes = Path(probes)
    if not probes.exists():
        raise SetupError(f"no quick-test file at {probes}; run `pruneagent probe` first")
    table = SensitivityTable.from_probe_results({"table": json.loads(probes.read_text())["table"]})
    if damage is None or not Path(damage).exists():
        notify(f"Warning: no measured per-layer damage at {damage}; the agent sees the quick tests only.")
        return table
    merge = getattr(table, "with_measured_damage", None)
    if merge is None:
        notify(f"Warning: {damage} exists but this code cannot read it yet (the damage-table loader is not "
               "merged); the agent sees the quick tests only.")
        return table
    record = json.loads(Path(damage).read_text())
    table = merge(record)
    measured = sum(len(r.measured_ppl) for r in table.rows.values())
    expected = len(table.rows) * len(P.STRATEGIES)
    notify(f"Measured per-layer damage: {measured} of {expected} entries from {damage}.")
    if measured < expected:
        notify(f"Warning: the measured damage table is incomplete ({measured} of {expected}).")
    return table


def prompt_builder(spec: str | None, notify: Notify = _stderr):
    """The prompt builder named "module:function", or None (the graph's placeholder) when it is not in the code."""
    if not spec:
        return None
    module, _, name = spec.partition(":")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        notify(f"Warning: prompt builder {spec} not found; using the graph's placeholder prompt.")
        return None


def make_agent(run_cfg: RunConfig, run_id: str | None) -> AC.AgentClient:
    """The agent client for the run's backend (Codex: pinned model and effort, usage limits, pause at 90%)."""
    model = run_cfg.agent_model or AC.CODEX_MODEL
    effort = run_cfg.agent_reasoning_effort or AC.CODEX_REASONING_EFFORT
    if run_cfg.agent_backend == "codex":
        backend = AC.CodexBackend(model=model, reasoning_effort=effort)
    else:
        backend = AC.ApiBackend(model=model, reasoning_effort=effort, provider=run_cfg.agent_provider or "openai")
    cache = AC.Cache(AC.DEFAULT_CACHE_DIR / "per_run" / run_id) if run_cfg.agent_cache_scope == "run" and run_id \
        else AC.Cache()
    return AC.AgentClient(backend, cache=cache, run_id=run_id, max_calls=run_cfg.max_agent_calls,
                          max_cost_usd=run_cfg.max_cost_usd)


def gpu_lock() -> Path:
    return Path(os.environ.get("TMPDIR") or "/tmp") / "pruneagent-mac-gpu.lock"


def check_gpu_lock(device: str) -> None:
    """On the Mac GPU, refuse to start unless `scripts/mac_gpu.sh` holds the lock (its holder is alive)."""
    if device != "mps":
        return
    pid_file = gpu_lock() / "pid"
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return
    except (OSError, ValueError):
        pass
    raise SetupError("the Mac GPU is shared: start the run through the GPU lock, e.g.\n  scripts/mac_gpu.sh "
                     "caffeinate -i uv run pruneagent run ...")


def _check_calibration(run_cfg: RunConfig, manifest: dict, model_id: str) -> None:
    from pruneagent import data as d

    entry = manifest["entries"].get(d.calib_name(model_id))
    if entry is None:
        raise SetupError(f"no C4 calibration set for {model_id} in the data cache; run `pruneagent data` first")
    c = run_cfg.calibration
    if list(entry["shape"]) != [c.n_samples, c.seqlen]:
        raise SetupError(f"the cached calibration set is {entry['shape']}, the run expects "
                         f"{c.n_samples} x {c.seqlen}")


def _check_damage_text(run_cfg: RunConfig, damage: Path) -> None:
    """The damage table the agent sees must be measured on the gate's own passages."""
    from pruneagent import data as d

    if run_cfg.gate_text == "wikitext2":  # earlier runs: kept as they ran
        return
    want = d.gate_text_name(run_cfg.gate_text, run_cfg.model_id)
    if not damage.exists():  # load_sensitivity warns; the app only starts models whose table exists
        return
    got = json.loads(damage.read_text()).get("data")
    if got != want:
        raise SetupError(f"the damage table at {damage} was measured on {got}, not {want}; set the model up again")


@dataclass
class Built:
    """The parts of a run, for the report and the dry run."""

    deps: G.Deps
    scores_cached: bool
    notes: list[str]


def build(run_cfg: RunConfig, setup: Setup, run_id: str | None, *, dry_run: bool = False,
          notify: Notify = _stderr) -> Built:
    """Every part of the run. A dry run loads the model on the CPU and never computes scores."""
    from pruneagent import data as d
    from pruneagent import evaluate as ev

    data_dir = Path(setup.data_dir) if setup.data_dir else None
    manifest = d.read_manifest(data_dir)
    _check_calibration(run_cfg, manifest, run_cfg.model_id)
    device = "cpu" if dry_run else setup.device
    if not dry_run:
        check_gpu_lock(device)
    notes: list[str] = []

    _check_damage_text(run_cfg, Path(setup.damage))
    table = load_sensitivity(Path(setup.probes), Path(setup.damage), notify)
    notes.append(f"quick tests: {setup.probes}")
    builder = prompt_builder(setup.prompt_builder, notify)
    notes.append(f"prompt: {setup.prompt_builder if builder else 'graph placeholder (v1 instructions not found)'}")

    notify(f"Loading {run_cfg.model_id} in {run_cfg.dtype} on {device} ...")
    dense = ev.load_model(run_cfg.model_id, setup.revision, run_cfg.dtype, device)
    ids = d.gate_tokens(run_cfg.gate_text, run_cfg.model_id, data_dir, setup.revision)
    seqlen = run_cfg.calibration.seqlen

    from pruneagent.cli import _scores

    scores = _scores(dense, run_cfg.model_id, setup.revision or "", device, run_cfg.dtype, data_dir,
                     Path(setup.scores), compute=not dry_run)
    cached = scores is not None
    agent = make_agent(run_cfg, run_id)
    if cached:
        evaluate = G.ppl_evaluator(dense, scores, ids, seqlen=seqlen, device=device, n_windows=run_cfg.gate_windows)
        baseline = G.flap_baseline(dense, scores.stats, ids, seqlen=seqlen, device=device,
                                   n_windows=run_cfg.gate_windows)
        notes.append(f"scores: {setup.scores}")
    else:  # dry run only: a real run computes them
        notes.append(f"scores: not cached at {setup.scores}; a real run computes them first, on the GPU")

        def evaluate(plan):
            raise SetupError("no scores")

        def baseline(run):
            raise SetupError("no scores")

    deps = G.Deps(config=dense.config, agent=agent, evaluate=evaluate, baseline=baseline,
                  load_sensitivity=lambda: table, notify=notify)
    if builder is not None:
        deps.build_prompt = builder
    return Built(deps=deps, scores_cached=cached, notes=notes)


def deps_from_run_dir(run_cfg: RunConfig, run_dir: Path) -> G.Deps:
    """The deps factory recorded with every `pruneagent run` (`FACTORY`): rebuilds the run's parts from its
    `setup.json`, for the run itself and for `pruneagent resume`."""
    return build(run_cfg, Setup.load(run_dir), Path(run_dir).name).deps


# ------------------------------------------------------------------------------------------------ entry points


def start(run_cfg: RunConfig, setup: Setup, root: Path | str = runstore.DEFAULT_ROOT, run_id: str | None = None,
          notify: Notify = _stderr) -> tuple[str, dict]:
    """A real run saved under `root/<run-id>/`; returns (run id, final state or {"paused": ...})."""
    check_gpu_lock(setup.device)  # before the run directory is made
    store = runstore.RunStore.create(run_cfg, root, run_id, FACTORY)
    setup.save(store.dir)
    notify(f"Run {store.run_id}: {store.dir}")
    try:
        deps = build(run_cfg, setup, store.run_id, notify=notify).deps
    except BaseException as e:
        store.set_status("paused", f"setup failed: {type(e).__name__}: {e}")
        raise
    deps.store = store
    return store.run_id, G.run(deps, run_cfg)


class _Rendered(Exception):
    pass


class _PromptCatcher:
    """Stands in for the agent in a dry run: keeps the first prompt and stops the graph there."""

    def __init__(self):
        self.prompt: str | None = None

    def propose(self, request):
        self.prompt = request.prompt
        raise _Rendered


def dry_run(run_cfg: RunConfig, setup: Setup, notify: Notify = _stderr) -> dict:
    """Build every part (model on the CPU, no score computation), then render round 1's prompt through the graph
    with stand-in perplexities. No agent call, no GPU, nothing saved. Returns {"prompt", "notes", "estimate",
    "stop_reason"}; `prompt` is None when the early warning ends the run before any agent call."""
    built = build(run_cfg, setup, None, dry_run=True, notify=notify)
    real = built.deps
    est = real.agent.estimate_run(run_cfg.max_agent_calls)
    catcher = _PromptCatcher()
    dense_plan = P.dense_plan(real.config)

    def stand_in_baseline(run: RunConfig) -> Baseline:
        return Baseline(calibration=run.calibration, memory_limit_gb=run.device.memory_limit_gb, ratio=0.0,
                        plan=dense_plan, weight_bytes=run.limit_bytes, val_ppl_subset=DRY_FLAP_PPL)

    deps = dataclasses.replace(real, agent=catcher, evaluate=lambda plan: DRY_DENSE_PPL,
                               baseline=stand_in_baseline, store=G.MemoryStore())
    state = {}
    try:
        state = G.build_graph(deps).invoke({"run": run_cfg}, {"recursion_limit": 50})
    except _Rendered:
        pass
    return {"prompt": catcher.prompt, "notes": built.notes, "scores_cached": built.scores_cached,
            "estimate": est.message() if est is not None else None,
            "usage": None if est is None else {
                "calls": est.calls, "calls_fit": est.calls_fit, "fits": est.fits,
                "windows": [{"name": w.name, "used_pct": w.used, "pct_per_call": w.pct_per_call,
                             "resets_at": w.resets_at.isoformat() if w.resets_at else None} for w in est.windows]},
            "stop_reason": state.get("stop_reason"),
            "warnings": [w.message for w in state.get("warnings", [])]}
