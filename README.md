# PruneAgent

An LLM agent that prunes Qwen3 to fit a memory limit you set for your device (and, optionally, a speed target) while keeping as much quality as possible. The comparison is Baseline: FLAP, an existing pruning method cut to the same memory limit: the question is whether the agent keeps more quality (perplexity and accuracy) than FLAP does. The pruned model is actually smaller and faster. Design: [`ARCHITECTURE.md`](ARCHITECTURE.md); plan: [`WORKPLAN.md`](WORKPLAN.md); decisions: [`adr/`](adr/).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`). uv installs Python 3.12 and the pinned dependencies from `uv.lock`.

```bash
uv sync              # create .venv from the lock file
uv run pytest        # tests
uv run ruff check .  # lint
uv run pruneagent version
```

## Project board

```bash
python3 board/server.py   # http://127.0.0.1:8765
```

See [`board/README.md`](board/README.md).
