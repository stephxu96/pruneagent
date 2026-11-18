"""Candidate evaluation (task 2.5; ARCHITECTURE.md FR6 and §4.1 "Skip unchanged early layers").

Each candidate plan is rebuilt from the cached scores (`pruner.prune`: slicing and masking, no score
recomputation) and its perplexity measured on the gate's fixed 32-window subset, exactly as
`evaluate.subset_perplexity` computes it.

Skipping unchanged early layers: a candidate usually differs from the last accepted plan only from some layer k
on. Layers before k are identical and see the same text, so their output is identical too. The evaluator keeps,
for the accepted plan, the input to every layer on every window (on `store_device`, CPU by default: about 134 MB
per layer for 32 windows of 2,048 tokens on Qwen3-0.6B in 16-bit, ~3.6 GB for 27 layers). A candidate then
starts from the saved input to layer k and recomputes layer k and everything after it. A candidate that changes
layer 0 runs in full.

The saved inputs follow the accepted plan passed with each call: the first call runs the accepted plan in full
and saves them; when the accepted plan later changes (an accepted step, or a rollback), they are refreshed from
the first layer where the old and new accepted plans differ. The refresh time counts towards the candidate that
triggered it.

Candidates run one after another on one device: calls hold a lock, so the graph's parallel evaluation tasks
queue for the GPU, and each pruned copy is released before the next is built (one dense model plus one pruned
copy in memory at a time). GPU seconds are measured inside the lock, so waiting in the queue is not counted.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass

import torch

from pruneagent import evaluate as ev
from pruneagent import plan as P
from pruneagent import pruner
from pruneagent.wanda import _run_block, _StopForward


@dataclass
class Evaluation:
    """One candidate's measurement."""

    ppl: float  # perplexity on the gate's windows
    gpu_seconds: float  # device time for this candidate, including any refresh of the saved layer inputs
    start_layer: int | None  # first layer recomputed (0: full pass); None: same plan as the accepted one
    refresh_seconds: float = 0.0  # part of gpu_seconds spent refreshing the saved inputs for a new accepted plan


def _layer_key(p: P.LayerPlan) -> tuple:
    return ("dropped",) if p["dropped"] else (p["mlp_keep"], p["groups_keep"], float(p["wanda_sparsity"]))


def first_change(plan: P.Plan, other: P.Plan) -> int | None:
    """First layer where two normalized plans build different layers (None if they build the same model)."""
    for i in sorted(plan):
        if _layer_key(plan[i]) != _layer_key(other[i]):
            return i
    return None


def _free(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


class CandidateEvaluator:
    """`graph.Deps.evaluate` for a real run. Call as `evaluator(plan, accepted)`; `accepted` (the last accepted
    plan) enables skipping the unchanged early layers, and without it the plan runs in full."""

    takes_accepted = True  # the graph passes the last accepted plan

    def __init__(self, dense, scores: pruner.Scores, ids: torch.Tensor, *, seqlen: int = 2048,
                 device: str | None = None, n_windows: int = ev.GATE_WINDOWS, store_device: str = "cpu",
                 skip: bool = True):
        self.dense, self.scores, self.seqlen = dense, scores, seqlen
        self.device = device or ev.pick_device()
        self.store_device, self.skip = store_device, skip
        wins = ev.windows(ids, seqlen)
        self.wins = wins[torch.as_tensor(ev.subset_indices(len(wins), n_windows), dtype=torch.long)]
        self.n_layers = P.shape(dense.config)[0]
        self._lock = threading.Lock()
        self._ref: P.Plan | None = None  # the plan whose layer inputs are saved
        self._ref_ppl: float | None = None
        self._saved: dict[int, torch.Tensor] = {}  # layer -> [windows, seqlen, hidden]: input to that layer
        self.log: list[Evaluation] = []  # every evaluation, in order

    # ------------------------------------------------------------------------------------------------ public

    def __call__(self, plan: dict, accepted: dict | None = None) -> Evaluation:
        cfg = self.dense.config
        plan = P.validate(plan, cfg)
        with self._lock:
            ev.synchronize(self.device)
            t0 = time.perf_counter()
            refresh = 0.0
            if not self.skip or accepted is None:
                start, ppl = 0, self._pass(plan, 0, save=False)
            else:
                refresh = self._sync(P.validate(accepted, cfg))
                start = first_change(plan, self._ref)
                ppl = self._ref_ppl if start is None else self._pass(plan, start, save=False)
            ev.synchronize(self.device)
            out = Evaluation(ppl=ppl, gpu_seconds=time.perf_counter() - t0, start_layer=start,
                             refresh_seconds=refresh)
            self.log.append(out)
            return out

    def saved_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._saved.values())

    # ----------------------------------------------------------------------------------------------- internals

    def _sync(self, accepted: P.Plan) -> float:
        """Make the saved layer inputs those of `accepted`; returns the seconds it took (0 if already so)."""
        if self._ref is not None:
            d = first_change(accepted, self._ref)
            if d is None:
                return 0.0
        else:
            d = 0
        t0 = time.perf_counter()
        self._ref_ppl = self._pass(accepted, d, save=True)
        self._ref = accepted
        ev.synchronize(self.device)
        return time.perf_counter() - t0

    def _layer_kwargs(self, model, inp: torch.Tensor) -> dict:
        """The arguments the model passes to its decoder layers (rotary embeddings, mask), captured once."""
        box: dict = {}

        def hook(_module, _args, kwargs):
            box.update({k: v for k, v in kwargs.items() if k not in ("hidden_states", "past_key_values")})
            raise _StopForward

        handle = model.model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            model(inp, use_cache=False)
        except _StopForward:
            pass
        finally:
            handle.remove()
        box["use_cache"] = False
        return box

    @torch.no_grad()
    def _pass(self, plan: P.Plan, start: int, save: bool) -> float:
        """Perplexity of `plan` from layer `start` on (earlier layers' output read from the saved inputs). With
        `save`, the inputs to every later layer are saved (the plan becomes the reference)."""
        model = pruner.prune(self.dense, plan, self.scores).to(self.device).eval()
        try:
            blocks = model.model.layers
            kept = [i for i in range(self.n_layers) if not plan[i]["dropped"]]
            pos = {d: j for j, d in enumerate(kept)}
            kwargs = self._layer_kwargs(model, self.wins[0].unsqueeze(0).to(self.device))
            losses = []
            for w, win in enumerate(self.wins):
                inp = win.unsqueeze(0).to(self.device)
                if start == 0:
                    h = model.model.embed_tokens(inp)
                else:
                    h = self._saved[start][w:w + 1].to(self.device)
                for d in range(start, self.n_layers):
                    if save and d > start:
                        self._slot(d, h)[w].copy_(h[0])
                    if d in pos:
                        h = _run_block(blocks[pos[d]], h, kwargs)
                losses.append(ev.window_loss(model.lm_head(model.model.norm(h)), inp))
            return ev.ppl_of(losses, self.seqlen)
        finally:
            del model
            _free(self.device)

    def _slot(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        t = self._saved.get(layer)
        if t is None or t.dtype != h.dtype:
            t = torch.empty((len(self.wins), *h.shape[1:]), dtype=h.dtype, device=self.store_device)
            self._saved[layer] = t
        return t
