"""The replay's exact removed parts (views.removal_order) match what the pruner removes."""

import torch
from safetensors.torch import save_file

from pruneagent import pruner
from pruneagent.service import views as V


def test_ranks_match_pruner(tmp_path):
    g = torch.Generator().manual_seed(0)
    mlp = torch.randn(3, 50, generator=g)
    mlp[0, 10] = mlp[0, 11]  # a tie: the lower index is kept first
    attn = torch.randn(3, 8, generator=g)
    path = tmp_path / "scores.safetensors"
    save_file({"mlp": mlp, "attn": attn}, str(path))
    r = V._ranks(str(path), 0.0)
    for score, ranks in ((mlp, r["mlp_rank"]), (attn, r["attn_rank"])):
        for layer in range(score.shape[0]):
            for keep in range(score.shape[1] + 1):
                kept = pruner.top_mask(score[layer], keep).tolist()
                assert kept == [rank < keep for rank in ranks[layer]]
