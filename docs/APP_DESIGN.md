# PruneAgent App Design

- **Status:** draft for Steph's review, 2026-11-19
- **Scope:** the web application (task 3.6 onward): who it serves, its two user stories, its pages, how it runs on the
  computer, and its look and wording.
- **Relation to other documents:** ADR-008 decided *that* there is a web app on top of the tools, and that rule
  stays (the app adds no pruning or measuring logic of its own). This document designs the app itself and replaces
  ADR-008's list of five screens. Once approved, its decisions go into ADR-009 and ARCHITECTURE.md §8b.

---

## 1. Who it's for

| Person | Wants | Knows |
|---|---|---|
| **Someone with a computer and a model** (the main user) | A smaller copy of the model that fits their computer, without losing much quality | Common sense; how big their computer's memory is. Not pruning, not training, not the command line |
| **A researcher** (Steph) | To start, watch and compare runs; to trust the numbers | Everything; also uses the command line |
| **An audience** (instructors, a demo) | To see what the AI did and whether it beat the standard method | Not computer science |

The main user decides the design. Researcher features stay, one level down.

### What the app must do

The app serves two separate user stories (§2):

- **Model setup:** put a model on this computer, ready to prune. Once per model; can take hours.
- **Model pruning:** make a smaller copy of a set-up model that fits a device. Once per device or size.

Across both:

1. **No surprises before starting:** whether it can work, how long it takes and how much of the AI usage limits
   it uses, all before anything starts.
2. **Visible progress:** where a long job is and how long is left.
3. **A result you can trust:** quality against the original and against the standard method (FLAP), and whether it
   now fits.
4. **The model in hand:** a download that runs elsewhere.
5. **Nothing lost:** closing the lid, a usage limit, a crash or an app update never loses work; everything can be
   continued.

### Not in scope

Hosting the app for other people; choosing neurons by hand; training; weight quantization (phase 5); models from a
folder on the computer (later).

---

## 2. Two user stories

Maya has an AI model she likes and a Mac mini with 16 GB of memory. She wants the model to run well on that Mac mini.
She knows nothing about pruning.

```mermaid
flowchart LR
    subgraph Setup[Model setup: once per model]
        S1[Name the model] --> S2[Check] --> S3[Set up:<br/>load the model,<br/>generate assets] --> S4[Ready]
    end
    subgraph Pruning[Model pruning: once per device]
        P1[Select your model] --> P2[Where it runs] --> P3[The answer:<br/>size, quality,<br/>time, usage] --> P4[Start] --> P5[Watch] --> P6[Your model]
    end
    S4 -. appears in .-> P1
    P1 -. not set up yet .-> S1
```

### 2.1 Model setup

**"I want this model on my computer, ready to prune."**

| Step | Maya sees | Maya decides | The app does | When something's wrong |
|---|---|---|---|---|
| **1 Name the model** | A box for the model's name on Hugging Face; which kinds work | Which model | — | — |
| **2 Check** | Whether it can be used here, its kind, its size, how long setup takes | Set it up, or choose another | Reads the model's details (not its weights): supported kind, fits this computer | Not supported, too big, needs a license, misspelled: one plain sentence and "Choose another" |
| **3 Set up** | Two parts with progress and time left: **load the model** (download it) and **generate assets** (sample text in the model's own pieces, how much it uses each part, the quality cost of each cut) | Pause, continue | Runs `pruneagent setup` | Stopped or failed: "Continue setup" picks up where it stopped |
| **4 Ready** | The model listed as ready | Prune it now, or later | — | — |

Setup ends at "ready". Pruning is a separate story that can start right away or days later.

### 2.2 Model pruning

**"I want a smaller version of my model that fits my device."**

| Step | Maya sees | Maya decides | The app does | When something's wrong |
|---|---|---|---|---|
| **1 Select your model** | Her set-up models, with their size | Which one | — | The one she wants isn't there: a link to model setup |
| **2 Where it runs** | This computer (read automatically) or a preset device: Mac mini, laptop, phone (presets lock memory, Custom unlocks it); how much to keep free for other apps | Which device | Works out the space the model may take: memory − other apps − working memory | Device too small for this model at any size: says so, suggests a smaller model |
| **3 The answer** | "To fit, it must be 20% smaller. Expect about 1.7–1.9× worse than the original; the standard method would be 13× worse. About 1 h 15 min–1 h 35 min. Uses about 14% of your 5-hour limit." | Go ahead, or change the device | Picks the size from the device (with a buffer) and runs the feasibility check, both on its own. No size to choose, no check button | Can't fit: why, and what would |
| **4 Start** | A confirmation: model, device, size, time, usage; "keep the lid open" | Start or cancel | Queues the run, then the final check | Mac busy with another job: "waits for the Mac to be free" |
| **5 Watch** | Progress at a glance: size against target, quality so far, time left; the round-by-round story below | Pause, continue | Streams each round as it lands | Lid closed, usage limit, crash: "Paused" or "Interrupted" with the reason and Continue |
| **6 Your model** | Fits ✓; quality against the original and FLAP; memory requirement; speed; download | — | Shows the final check; builds the saved model | Target missed: the closest result and the gap |

The expected-quality range in step 3 comes from past runs at that model and size; with none yet, it says so.

### 2.3 What changes from today because of the stories

- **The user never chooses a percentage.** The size follows from the device. Today she picks "How much smaller?" on a
  slider and presses "Feasibility check".
- **The final check runs on its own** after every run. Today's time estimate already counts its 20 minutes, but the
  user has to start it by hand, so a run never reaches "Your model" unless she knows to.
- **Download gives the model,** not only the plan file, which works only with this tool (FR12: safetensors plus plan).
- **Setup and pruning are separate.** Today the model page mixes them ("Make it smaller →" at the end of setup,
  "isn't prepared yet" inside New run).

## 3. Pages and navigation

| Page | Address | Story | Holds | Today |
|---|---|---|---|---|
| **Runs** (home) | `#/` | Pruning | Active runs first, each with a progress bar; then finished runs; one main button: "Prune a model" | Run cards plus the "Your computer" panel |
| **Prune a model** | `#/new` | Pruning, steps 1–4 | Step bar: Select your model → Where it runs → The answer → Start | Split between the home panel and New run |
| **Run** | `#/run/<id>` | Pruning, steps 5–6 | While running: the progress view. When finished: the result and download. Below both: "How it got there", the round-by-round replay | The replay, with results under it |
| **Models** | `#/models` | Setup | The models on this computer (ready, setting up, paused) and "Set up a model", which opens the setup steps | "Select your model", which mixed both stories |
| **Compare** | `#/compare` | — | Every finished run against FLAP across sizes (researchers, demo) | Same |
| **Activity** | `#/activity` | — | Every job (checks, runs, setups), with outcome and system logs | The History list on New run |

**Navigation:** Runs · Prune a model · Models · Compare. Activity sits in the footer: it's for fixing problems.

## 4. How it runs on the computer

### 4.1 The pieces, and what each one is the source of truth for

```mermaid
flowchart TB
    B[Browser] -- HTTP, live updates --> S[App server]
    S -- queue --> Q[(Job list<br/>runs/web_jobs.sqlite)]
    S -- reads --> RF[(Run folders<br/>runs/&lt;id&gt;/)]
    S -- reads --> MS[(Model assets<br/>scores, quick tests,<br/>damage tables, text)]
    S -- starts --> W[Job process<br/>own process group]
    W -- GPU lock --> G{{scripts/mac_gpu.sh}}
    W -- writes --> RF
    W -- writes --> MS
    T[Terminal] -- same commands --> W
```

| Piece | Source of truth for | Survives an app restart |
|---|---|---|
| **Run folder** `runs/<id>/` | Everything about a run: settings, each round, kept plans, checkpoint, final check, and (new) its own copy of the AI's instructions | Yes |
| **Model assets** (scores, quick tests, damage table, sample text) | Whether a model is set up on this computer | Yes |
| **Model list** `runs/models.json` | Which added models the user sees | Yes |
| **Job list** `runs/web_jobs.sqlite` | What the app was asked to do, each job's log and outcome. *Not* the run's state: that is in the run folder | Yes |
| **Job process** | Doing the work: one process group per job, started with the same command a user would type | Yes (its own process group) |
| **App server** | Nothing of its own: it reads the files above | — (it is what restarts) |
| **GPU lock** | One GPU job at a time, from the app or a terminal | Yes |

The rule that keeps it simple: **the server holds no state that matters.** It can stop and start at any time.

### 4.2 Jobs

| Kind | Command | Uses the GPU | Ends with |
|---|---|---|---|
| Feasibility check | `pruneagent run --dry-run --json-result` | No | A JSON line: reachable, time, usage |
| Run | `pruneagent run` | Yes | The run folder's state: finished, paused, interrupted |
| Continue | `pruneagent resume <id>` | Yes | Same |
| Final check | `scripts/heldout_check.py`, `scripts/accuracy_check.py` | Yes | `heldout.json`, `accuracy.json` |
| Model setup | `pruneagent setup <model>` | Yes | The model and its assets |
| Build the download (new) | `pruneagent save <id>` | No | The saved model |

**States:** waiting → running → finished, paused, failed. A paused job can be continued.

**Follow-up jobs.** A run queues its own final check when it ends with the target met (new). The user doesn't start it.

**After an app restart (new).** Each job's process id is stored. On start, the server looks at every job marked
running: if its process is alive, it keeps watching it; if not, it reads the outcome from what the job left behind
(the run folder's state, the result files). No job is marked "Interrupted" while it is still working.

### 4.3 Updating the app while something runs

Updating the app must never disturb a run in progress. What each kind of update touches, and the rule for it:

| Update | Risk to a running job | Rule |
|---|---|---|
| Interface (pages, styles) | None | Build into a new folder and swap it in at once, so no one sees a half-built page |
| App server code | The server restarts | Jobs are re-attached (§4.2) |
| Run code | None: a running process keeps the code it loaded | — |
| AI instructions | **High today:** a run re-reads them every round, so an update would change what the AI sees mid-run | Each run copies its instructions into its run folder at the start and reads only that copy (new). This also records exactly what the AI saw |
| Check scripts, for a later final check | The check uses the newer code | Accepted: checks measure, they don't decide. Each result file records the code version it ran with |
| Python packages (`uv sync`) | A running job may load a package later and get the new one | Wait until no GPU job runs |

**One command does an update:** `scripts/deploy.sh` merges, builds the interface and swaps it in, restarts the server
if its code changed, and leaves package updates for later if a job is running.

### 4.4 Live updates

The run page gets each round as it lands (server-sent events, as today). The job list and the progress bars on the
Runs page refresh every 3 seconds. That's fast enough for jobs that take minutes per step.

### 4.5 When things go wrong

| What happens | What the user sees | What they can do |
|---|---|---|
| Lid closed, computer asleep | "Interrupted: the computer went to sleep" | Continue; it picks up at the last kept round |
| Usage limit reached | "Paused: usage limit, resets at 16:41" | Continue after the reset |
| A job fails | "Didn't finish", with the last error in one line; system logs in a fold | Continue, or report the logs |
| The Mac is busy with another GPU job | "Waiting for the Mac to be free (busy with: …)" | Wait, or cancel |
| App closed or updated | Nothing changes: runs carry on | Open the app again |
| A model needs a license or isn't supported | One plain sentence, at step 2 | Choose another model |

### 4.6 Safety

Local and single-user: the server listens on 127.0.0.1 only. The browser never sends a command line, only checked
fields (model name pattern, size between 1% and 60%, run id pattern). Keys and the Codex login stay on the server.

---

## 5. Look and wording

### 5.1 Shared parts

Every page is built from these parts and nothing else, so a dialog or a status looks the same everywhere.

| Part | Used for | Notes |
|---|---|---|
| **Page header** | Title, one back link | No intro sentence unless it says something the page doesn't |
| **Step bar** | Prune a model; Model setup | Numbered; done steps get ✓; the current one is highlighted |
| **Panel** | Any group of controls or facts | One border style, one radius |
| **Fact list** | Label → value pairs (dialogs, settings) | Labels are nouns: "Target size", "Time" |
| **Tiles** | Check results: Size, Time, Usage limits | Big value, small note under it |
| **Status badge** | Waiting, Running, Paused, Interrupted, Finished, Didn't finish, Ready | One color per state, same everywhere, always with a word |
| **Progress bar** | Runs, model setup | Always with words: "57 of 128 cuts tested · about 55 minutes left" |
| **Dialog** | Confirmations | Browser `<dialog>`: Escape and the backdrop close it; the safe button has focus |
| **Fold** | System logs, details | Always has an arrow; the summary says what's inside |
| **Notice** | Information, warning, problem | One line in bold, then the reason |
| **Buttons** | One main (filled) button per view; others outlined | The main button names the next step: "Set up this model", "Start run" |

### 5.2 Wording rules

- Plain, short words; what it means for the user, not how it works ("fits your Mac", not "under the memory limit").
- Labels are nouns: "Feasibility check", "Usage limits", "Current state". Not "Check the size".
- Times are ranges and include the final check; a worst case if a run can use all its rounds.
- Usage limits as the share of the 5-hour and weekly limits this run uses.
- "Memory requirement" for the memory a model needs while answering (memory access cost); layers numbered from 0.
- Quality as "X% worse than the original" or "N× worse".
- "Model setup", "Set up", "Setting up"; never "prepare" on screen.
- No internal names on screen ("damage table", "probe", "qwen3_moe"); they stay in system logs.

### 5.3 Screens of every size

Works from phone width up: a 16px side margin, nothing wider than the screen, long model names wrap. Keyboard
works everywhere; motion stops when the computer asks for less motion; light and dark themes.

---

## 6. What changes from today

| Area | Today | Designed | Task |
|---|---|---|---|
| Two stories | One "Select your model" page mixes setup and pruning; New run says "isn't prepared yet" | Models page for setup (§2.1); Prune a model for pruning (§2.2) | 3.12 (new) |
| Size | A "How much smaller?" slider and a "Feasibility check" button | Size from the device, check runs on its own; the answer in plain words | 3.12 |
| Home | Run cards plus the computer panel | Runs, active first with progress; the computer panel becomes "Where it runs" | 3.12 |
| Setup wording | "Prepare this model", "Preparing … for pruning"; `pruneagent prepare` | "Set up this model"; two parts: load the model, generate assets; `pruneagent setup` | 3.12 |
| Active run | Replay controls only | A progress view at the top (§2.2 step 5) | 3.10 |
| Final check | Started by hand on the run page | Queued on its own when a run meets its target | 3.13 (new) |
| Download | The plan file | The saved model (plus plan) | 3.14 (new) |
| Job history | At the bottom of New run | Activity page | 3.12 |
| App updates | Can change a running run's instructions; restart marks jobs "Interrupted" | §4.3 rules and `scripts/deploy.sh` | 3.11 |
| Look | Each part styled where it was first needed | The shared parts of §5.1 | 3.12 |

Suggested order: **3.11** (updates stop being risky) → **3.12** (two stories, structure) → **3.10**, **3.13**, **3.14**.

---

## 7. Questions for Steph

1. **Final check on its own** after every run (adds about 20 minutes of Mac time per run, already in the time
   shown)? Recommended: yes.
2. **Compare page:** visible to everyone, or only from a link on finished runs?
