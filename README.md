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

## Web app

Watch the AI cut a model down, round by round, and compare it with the standard method (FLAP). It has two parts
that run in the background on this Mac: the **web app** (what the browser shows) and the **workflow service**
(model setup, pruning runs and evaluation). Closing the browser, restarting the app or updating it never stops
work in progress (docs/APP_DESIGN.md §4).

```bash
uv sync --extra web
(cd web && npm install && npm run build)   # once
scripts/install_services.sh                # web app + workflow service, started at login, restarted if they stop
open http://127.0.0.1:8125
```

**Updating:** after merging, run `scripts/deploy.sh`. It makes a new release (a fixed copy of the code with its own
environment, in `releases/`), swaps in the new interface and restarts both services. Work already running finishes on
the release it started with.

Without the background services: `uv run pruneagent worker` and `uv run --extra web pruneagent web` in two
terminals. `scripts/install_services.sh --remove` takes the services out. Open a round directly with
`#/run/<run-id>/<round>`. For interface work, `cd web && npm run dev` serves it on :5173 with the API forwarded.

### Your own model

Any Qwen3, Qwen2, Llama or Mistral model from Hugging Face can be used. In the app, open **Select your model**, type the
model's name (e.g. `HuggingFaceTB/SmolLM2-360M`), and press **Check**: it says whether the model fits this computer
and how long preparing it takes. **Prepare this model** then measures it once on this computer (its text, how much
each part is used, and the quality cost of each cut); this can be paused and continued. From the command line:

```bash
uv run pruneagent check-model HuggingFaceTB/SmolLM2-360M
uv run pruneagent prepare HuggingFaceTB/SmolLM2-360M      # ~1.5-3 h on an M4 Mac for a 0.4-2B model
```

Models that need a license on Hugging Face (e.g. Llama) need `hf auth login` first. Models built from expert parts
(mixture of experts), and other families such as Gemma and Phi, are refused with a reason.

## Agent backend

The agent's decisions come from `pruneagent.agent_client`. By default it calls Codex on your ChatGPT plan; an API backend is also built in, but it has not yet made a real call.

```python
from pruneagent.agent_client import AgentClient, AgentRequest

reply = AgentClient(run_id="my-run").propose(AgentRequest(prompt, schema))
reply.data  # the agent's JSON answer
```

**One-time setup:** install Codex (`codex --version` should print 0.155.1 or later) and run `codex login` with your ChatGPT account. The client uses that login and nothing else from your Codex settings.

**What each call does:**

- Runs Codex in a new, empty folder, read-only, with no tools.
- Saves each call's Codex conversation in `~/.codex-pruneagent/`, not with your own Codex conversations, because only a saved conversation records how much of your plan's usage limits is left. That folder uses your normal Codex login (its `auth.json` is a link to the one in `~/.codex`). Set `PRUNEAGENT_CODEX_HOME` to use another folder.
- Uses the model `gpt-5.6-terra` at `high` reasoning effort (chosen 2026-11-17, see results/effort_check/summary.md), set in `agent_client.py` (`CODEX_MODEL`, `CODEX_REASONING_EFFORT`).
- Skips Codex's built-in instructions, your personal Codex settings, plugins, skills and add-ons. It sends one line of instructions instead ([`codex_instructions.md`](src/pruneagent/codex_instructions.md)).
- Saves every answer in `runs/agent_cache/`. Asking the identical question again returns the saved answer without a call.
- Writes one line per call to `runs/agent_usage.jsonl`: time, run, backend, model, tokens, seconds and outcome.
- Waits up to 3 minutes for each call and tries once more after a dropped connection or timeout. It does not retry after a usage limit, a login problem or an answer that is not the requested JSON.
- Stops a run after 60 calls. On the API backend, it also stops at $5.

**Size of a call:** a one-word reply used 19,426 input tokens with Codex's normal setup (measured 2026-11-17) and 3,135 with this setup. Nearly all of those 3,135 tokens are Codex's own fixed part, which no supported setting we found removes; a real question adds its own length on top.

**API backend:** `ApiBackend` reads `OPENAI_API_KEY` from the environment or from `.env`, which git ignores. Pass prices per million tokens to have the cost recorded and the $5 cap enforced.

## Starting a run

A run on the Mac must hold the shared GPU lock (`scripts/mac_gpu.sh`) and keep the Mac awake (`caffeinate -i`). For a memory limit 10% below the unpruned Qwen3-0.6B-Base, with Codex as the agent:

```bash
scripts/mac_gpu.sh caffeinate -i uv run pruneagent run --model Qwen/Qwen3-0.6B-Base --memory-cut 0.10 --device mps --backend codex
```

- `--memory-cut 0.10` sets the limit to 10% less weight memory than the unpruned model; `--memory-limit-mb 1000` sets it in MB instead. Give one of the two.
- `--max-rounds N` stops after N agent replies (default 40).
- `--dry-run` loads everything and prints the agent's first prompt, then stops. It calls no agent, uses no GPU and saves nothing, so it needs no GPU lock: `uv run pruneagent run --memory-cut 0.10 --device mps --dry-run`. The two perplexities in that prompt are placeholders, because nothing is measured.

What the run loads:

- the model in 16-bit on the GPU;
- its pruning scores from `checkpoints/scores/`, computed from the C4 calibration text and saved there first if they are missing;
- the quick tests from `results/probes/`, and the measured per-layer damage from `results/damage_table/` if that file exists (without it the run starts anyway and says so);
- Baseline: FLAP cut to the same memory limit, measured on the same validation text as every candidate.

The run is saved under `runs/<run-id>/` together with the files it was built from (`setup.json`), so `pruneagent resume` rebuilds it as it was. On the Mac, resume through the lock too: `scripts/mac_gpu.sh caffeinate -i uv run pruneagent resume`.

## Runs, checkpoints and resume

Each run saves its progress in its own folder, `runs/<run-id>/`, named like `20261116-1432-qwen3-0.6b-codex`:

- `run.sqlite`: the run's status, its progress at the end of every round, and the list of accepted plans.
- `plans/accepted-000.json`, `accepted-001.json`, ...: each accepted plan (how much of each layer is kept). `000` is the unpruned model. No weights are stored: a plan rebuilds its model in seconds.
- `decisions.jsonl`: the decision log, one line per round: what the agent chose and why, every candidate's perplexity, and whether it was accepted.

When the agent rolls back to an earlier accepted plan, the plans accepted after it are deleted.

```
pruneagent runs              # every run: status, last round, accepted plans
pruneagent resume            # continue the only paused run
pruneagent resume <run-id>   # continue a given run
pruneagent resume --backend api   # continue on the API backend instead of Codex
```

A run that stopped part-way (a usage limit, Ctrl-C, or a Colab runtime that disconnected) continues from the end of its last finished round. The unfinished round is repeated, but its agent answer comes from `runs/agent_cache/`, so no agent call is made twice. If several runs are paused, `pruneagent resume` without an id lists them and asks for one.

## Usage limits

Your ChatGPT plan allows a certain amount of Codex use every 5 hours and every week. A run makes up to 60 agent calls; at today's sizes that is roughly 15–30% of a 5-hour allowance.

```
pruneagent limits            # how much is used now, and whether a 60-call run fits (makes no call)
```

- **Before a run** (and before a resume), it says how many calls fit before either allowance reaches 90%, and warns if that is fewer than the run may need. A warning does not stop the run.
- **After every call**, it records how much of each allowance is used in `runs/agent_usage.jsonl`.
- **At 90% of either allowance**, the run finishes the round it is in, saves it and stops, with status `paused: ... resets at <time>` in `pruneagent runs`. If Codex refuses a call because the allowance is used up, the run stops the same way and redoes that round later. Stopping at 90% also keeps a run from drawing on paid credits.
- **`pruneagent resume`** continues after the reset time. Run before it, it stops again straight away without making a call.

Until enough calls have been recorded (about 20–30), the estimate assumes each call uses 0.4% of the 5-hour allowance and 0.2% of the weekly one (measured 2026-11-16); after that it uses what the recorded calls actually used.
