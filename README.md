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

## Agent backend

The agent's decisions come from `pruneagent.agent_client`. By default it calls Codex on your ChatGPT plan; an API backend is also built in, but it has not yet made a real call.

```python
from pruneagent.agent_client import AgentClient, AgentRequest

reply = AgentClient(run_id="my-run").propose(AgentRequest(prompt, schema))
reply.data  # the agent's JSON answer
```

**One-time setup:** install Codex (`codex --version` should print 0.155.1 or later) and run `codex login` with your ChatGPT account. The client uses that login and nothing else from your Codex settings.

**What each call does:**

- Runs Codex in a new, empty folder, read-only, with no tools, and saves no conversation history.
- Uses the model `gpt-5.6-terra` at `high` reasoning effort (chosen 2026-11-17, see results/effort_check/summary.md), set in `agent_client.py` (`CODEX_MODEL`, `CODEX_REASONING_EFFORT`).
- Skips Codex's built-in instructions, your personal Codex settings, plugins, skills and add-ons. It sends one line of instructions instead ([`codex_instructions.md`](src/pruneagent/codex_instructions.md)).
- Saves every answer in `runs/agent_cache/`. Asking the identical question again returns the saved answer without a call.
- Writes one line per call to `runs/agent_usage.jsonl`: time, run, backend, model, tokens, seconds and outcome.
- Waits up to 3 minutes for each call and tries once more after a dropped connection or timeout. It does not retry after a usage limit, a login problem or an answer that is not the requested JSON.
- Stops a run after 60 calls. On the API backend, it also stops at $5.

**Size of a call:** a one-word reply used 19,426 input tokens with Codex's normal setup (measured 2026-11-17) and 3,135 with this setup. Nearly all of those 3,135 tokens are Codex's own fixed part, which no supported setting we found removes; a real question adds its own length on top.

**API backend:** `ApiBackend` reads `OPENAI_API_KEY` from the environment or from `.env`, which git ignores. Pass prices per million tokens to have the cost recorded and the $5 cap enforced.
