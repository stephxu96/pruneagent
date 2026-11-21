import { Component, useEffect, useMemo, useState, type ReactNode } from "react";
import { MapLegend, ModelMap } from "./ModelMap";
import { Neurons3D } from "./Neurons3D";
import { PathChart } from "./PathChart";
import { NeuronMap, StepGrid, type Removal } from "./StepsView";
import {
  api, changedLayers, changesText, duration, lossText, mb, modelName, outcomeText, pct, qualityLoss, stateText, sweepLayer, timeLeftText,
  type Results, type RunDetail,
} from "./lib";
import { StepBar, type Job } from "./parts";

export function RunPage({ id, start = 0 }: { id: string; start?: number }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [at, setAt] = useState(start); // round being shown; 0 = the original model
  const [playing, setPlaying] = useState(false);
  const [showFlap, setShowFlap] = useState(false);
  const [follow, setFollow] = useState(true);
  // the view can be named in the address (…/run/<id>/<round>?view=exact), to link to it
  const asked = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("view");
  const [view3, setView] = useState<"3d" | "flat" | "exact">(
    asked === "exact" || asked === "flat" ? asked : CAN_3D ? "3d" : "flat");
  const [removal, setRemoval] = useState<Removal | null>(null);

  const [pending, setPending] = useState<Job | null>(null); // requested, but the run hasn't started yet
  const load = () =>
    api<RunDetail>(`/api/runs/${id}`).then((r) => {
      setRun(r);
      setPending(null);
      if (follow && r.state === "running") setAt(r.rounds.length);
    }).catch((e) => {
      api<Job[]>("/api/jobs").then((js) => {
        const j = js.find((x) => x.run_id === id && (x.kind === "start" || x.kind === "resume"));
        if (j) setPending(j);
        else setErr(String(e));
      }).catch(() => setErr(String(e)));
    });

  // until the run exists, look again every few seconds
  useEffect(() => {
    if (!pending) return;
    const t = setTimeout(load, 3000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending]);

  useEffect(() => {
    load();
    api<Results>(`/api/runs/${id}/results`).then(setResults).catch(() => undefined);
    api<Removal>(`/api/runs/${id}/removal`).then(setRemoval).catch(() => setRemoval(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // live updates while the run is going
  useEffect(() => {
    if (run?.state !== "running") return;
    const es = new EventSource(`/api/runs/${id}/events`);
    es.onmessage = () => load();
    return () => es.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, run?.state]);

  const last = run?.rounds.length ?? 0;
  // arrow keys step through the rounds (not while typing or using the slider)
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return;
      if (e.key === "ArrowRight") { setPlaying(false); setAt((a) => Math.min(last, a + 1)); }
      if (e.key === "ArrowLeft") { setPlaying(false); setAt((a) => Math.max(0, a - 1)); }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [last]);
  useEffect(() => {
    if (!playing) return;
    if (at >= last) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setAt((a) => a + 1), 2500); // time to read the round's ideas below the map
    return () => clearTimeout(t);
  }, [playing, at, last]);

  const view = useMemo(() => {
    if (!run) return null;
    const steps = run.steps.filter((s) => s.round <= at);
    const step = steps[steps.length - 1];
    const prev = steps[steps.length - 2];
    const round = at > 0 ? run.rounds[at - 1] : null;
    const changed = round?.outcome === "ACCEPT" ? changedLayers(prev?.layers, step.layers) : new Set<number>();
    return { step, prev: round?.outcome === "ACCEPT" ? prev : undefined, round, changed };
  }, [run, at]);

  if (err) return <p className="error">Could not load this run: {err}</p>;
  if (pending && !run) {
    return (
      <div className="run">
        <a href="#/" className="back">← Past runs</a>
        <h1>{modelName(pending.params.model ?? "")}{pending.params.cut != null ? `, ${pct(pending.params.cut)} smaller` : ""}</h1>
        <p className="status">
          <span className={`badge ${pending.status === "failed" ? "failed" : ""}`}>
            {pending.status === "queued" ? "Waiting for the Mac to be free" : pending.status === "failed" ? "Didn't start" : "Starting"}
          </span>
          {pending.status === "failed" && pending.reason}
        </p>
        <p className="muted">It starts as soon as the Mac is free. This page follows it on its own.</p>
      </div>
    );
  }
  if (!run || !view) return <p className="muted">Loading…</p>;

  const base = run.original_error ?? 1;
  const orig = run.original_mb ?? 0;
  const { step, prev, round, changed } = view;
  const needed = orig - run.limit_mb;
  const savedNow = orig - step.size_mb;
  const progress = Math.min(1, savedNow / needed);
  
  return (
    <div className="run">
      <header className="run-head">
        <a href="#/" className="back">← Past runs</a>
        <h1>
          Making {modelName(run.model)} {run.cut != null ? pct(run.cut) : ""} smaller
        </h1>
        <Settings run={run} orig={orig} />
        <p className="status">
          <span className={`badge ${run.state}`}>{stateText[run.state] ?? run.state}</span>
          {run.rounds.length} rounds · {run.kept_steps} steps kept
          {run.state === "running" && run.time_left && <> · {timeLeftText(run.time_left)}</>}
          {run.state === "running" && (
            <label className="follow">
              <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow live
            </label>
          )}
        </p>
      </header>

      <RunProgress run={run} onChange={load} />

      <section className="player">
        <button className="primary" onClick={() => { if (at >= last) setAt(0); setPlaying(!playing); }}>
          {playing ? "Pause" : at >= last ? "Replay" : "Play"}
        </button>
        <button className="secondary step-btn" aria-label="Previous round" disabled={at <= 0}
          onClick={() => { setPlaying(false); setAt((a) => Math.max(0, a - 1)); }}>◀</button>
        <button className="secondary step-btn" aria-label="Next round" disabled={at >= last}
          onClick={() => { setPlaying(false); setAt((a) => Math.min(last, a + 1)); }}>▶</button>
        <input type="range" min={0} max={last} value={at} aria-label="Round"
          onChange={(e) => { setPlaying(false); setAt(Number(e.target.value)); }} />
        <span className="where">{at === 0 ? "Original model" : `Round ${at} of ${last}`}</span>
      </section>

      <div className="timeline" aria-label="Rounds">
        {run.rounds.map((r) => (
          <button key={r.round} title={`Round ${r.round}: ${outcomeText[r.outcome ?? ""] ?? "no change"}`}
            className={`tick-btn ${r.outcome?.toLowerCase() ?? ""} ${r.round === at ? "on" : ""}`}
            onClick={() => { setPlaying(false); setAt(r.round); }} />
        ))}
      </div>

      <section className="stage">
        <div className="gauges">
          <div className="gauge">
            <div className="g-label">Size</div>
            <div className="g-value">{mb(step.size_mb)}</div>
            <div className="bar"><div className="fill" style={{ width: `${progress * 100}%` }} /></div>
            <div className="g-note">{progress >= 1 ? "Goal met ✓" : `${mb(Math.max(0, step.size_mb - run.limit_mb))} still to cut`}</div>
          </div>
          <div className="gauge">
            <div className="g-label">Quality loss</div>
            <div className="g-value">{lossText(qualityLoss(step.error, base))}</div>
            <div className="g-note">vs. the original</div>
          </div>
          {step.mac_mb != null && (
            <div className="gauge">
              <div className="g-label">Memory requirement</div>
              <div className="g-value">{mb(step.mac_mb)}</div>
              <div className="g-note" title={`The model plus its notes on a ${run.mac_context}-token conversation. It is also the memory access cost (MAC): all of it is read once for every token written.`}>
                model + working memory
              </div>
            </div>
          )}
          {run.flap?.error && (
            <div className="gauge flap-g">
              <div className="g-label">FLAP at the same size</div>
              <div className="g-value">{lossText(qualityLoss(run.flap.error, base))}</div>
              <div className="g-note">
                <label><input type="checkbox" checked={showFlap} onChange={(e) => setShowFlap(e.target.checked)} /> FLAP's cut</label>
              </div>
            </div>
          )}
        </div>

        <div className="view-toggle" role="group" aria-label="View">
          {CAN_3D && <button className={view3 === "3d" ? "on" : ""} onClick={() => setView("3d")}>3D neurons</button>}
          <button className={view3 === "flat" ? "on" : ""} onClick={() => setView("flat")}>Flat</button>
          {removal && <button className={view3 === "exact" ? "on" : ""} onClick={() => setView("exact")}>Exact neurons</button>}
        </div>
        {view3 === "exact" && removal ? (
          <NeuronMap removal={removal} full={{ mlp: run.shape.mlp_full, groups: run.shape.groups_full }}
            prev={showFlap && run.flap ? run.steps[0].layers : (prev ?? step).layers}
            cur={showFlap && run.flap ? run.flap.layers : step.layers}
            caption={showFlap ? "FLAP, same size" : at === 0 ? "Original model" : prev ? `Round ${at}` : `Round ${at}: nothing kept`} />
        ) : view3 === "3d" ? (
          <Fallback onError={() => setView("flat")}><Neurons3D
            layers={showFlap && run.flap ? run.flap.layers : step.layers}
            prev={showFlap ? undefined : prev?.layers}
            full={{ mlp: run.shape.mlp_full, groups: run.shape.groups_full }}
            caption={showFlap ? "FLAP, same size" : undefined}
          /></Fallback>
        ) : (
          <>
            <ModelMap
              layers={showFlap && run.flap ? run.flap.layers : step.layers}
              full={{ mlp: run.shape.mlp_full, groups: run.shape.groups_full }}
              changed={showFlap ? undefined : changed}
              caption={showFlap ? "FLAP, same size" : at === 0 ? "Original model" : `After round ${at}`}
            />
            <MapLegend />
          </>
        )}
      </section>

      <section className="story">
        {round ? (
          <>
            <h2>
              Round {round.round}{sweepLayer(run, round.round) != null && <> · layer {sweepLayer(run, round.round)}</>}: <span className={`outcome ${round.outcome?.toLowerCase()}`}>{outcomeText[round.outcome ?? ""] ?? "no change"}</span>
            </h2>
            {round.candidates.length > 0 ? (
              <ol className="ideas">
                {round.candidates.map((c) => (
                  <li key={c.id} className={round.chosen === c.id ? "chosen" : ""}>
                    <div className="idea-what">
                      {round.chosen === c.id && <span className="badge kept">kept</span>}
                      Idea {c.id + 1}: {changesText(c.changes)}
                    </div>
                    <div className="idea-numbers">
                      {c.error != null ? <>saves {mb(c.freed_mb)} · quality {c.rise != null ? (c.rise <= 0 ? "no worse" : `${pct(c.rise, 1)} worse this step`) : ""}</> : "not tested"}
                      {c.rise != null && c.rise > 0.15 && <span className="warn"> · too much, over the 15% limit</span>}
                    </div>
                    <details>
                      <summary>AI reasoning</summary>
                      <p>{c.why}</p>
                    </details>
                  </li>
                ))}
              </ol>
            ) : round.outcome === "LEFT" ? (
              round.why ? (
                <details>
                  <summary>AI reasoning</summary>
                  <p>{round.why}</p>
                </details>
              ) : <p className="muted">No reason recorded</p>
            ) : (
              <p className="muted">{round.refused_reasons.join(" ") || "No ideas"}</p>
            )}
          </>
        ) : null}
      </section>

      <section>
        <h2>Steps by layer</h2>
        <StepGrid run={run} removal={removal} at={at} onPick={(r) => { setPlaying(false); setAt(r); }} />
      </section>

      <section>
        <h2>All tested ideas</h2>
        <PathChart run={run} upTo={at} />
      </section>

      {run.state === "finished" && <Evaluation runId={id} results={results} onResults={setResults} />}
      {run.state === "finished" && <YourModel run={run} onChange={load} />}
    </div>
  );
}

function Settings({ run, orig }: { run: RunDetail; orig: number }) {
  const s = run.settings;
  const rows: [string, string][] = [
    ["Model", modelName(run.model)],
    ["Strategy", run.search_label],
    ["Original size", mb(orig)],
    ["Target size", `under ${mb(run.limit_mb)}${run.cut != null ? ` (${pct(run.cut)} smaller)` : ""}`],
    ["Device", s.device.toUpperCase()],
    ["AI", s.agent],
    ["Ideas per round", String(s.ideas_per_round)],
    ["Max quality drop per step", pct(s.max_drop_per_step)],
    ["Max rounds", String(s.max_rounds)],
    ["Max AI calls", String(s.max_ai_calls)],
    ["Max layers removed", String(s.max_layers_removed)],
    ["Min layer width kept", pct(s.min_layer_width)],
    ["Sample text", s.sample_text],
  ];
  return (
    <dl className="settings">
      {rows.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ResultBars({ held }: { held: NonNullable<Results["heldout"]> }) {
  const base = held.unpruned.test_error;
  const rows = [
    { name: "PruneAgent", cls: "agent", loss: qualityLoss(held.agent.test_error, base), size: held.agent.size_mb },
    { name: "Standard method (FLAP)", cls: "flap", loss: qualityLoss(held["Baseline: FLAP"].test_error, base), size: held["Baseline: FLAP"].size_mb },
  ];
  const max = Math.max(...rows.map((r) => r.loss));
  return (
    <div className="bars">
      {rows.map((r) => (
        <div key={r.name} className="bar-row">
          <span className="bar-name">{r.name}</span>
          <span className="bar-track"><span className={`bar-fill ${r.cls}`} style={{ width: `${Math.max(1.5, (r.loss / max) * 100)}%` }} /></span>
          <span className="bar-val">{lossText(r.loss)}</span>
        </div>
      ))}
      <p className="note">Quality lost vs. the original · shorter is better</p>
    </div>
  );
}

// The 3D view needs WebGL; without it (some browsers, remote desktops) the page shows the flat map instead.
const CAN_3D = (() => {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") ?? c.getContext("webgl"));
  } catch {
    return false;
  }
})();

class Fallback extends Component<{ onError: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onError();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

const STAGES = [
  { key: "run", label: "Pruning" },
  { key: "heldout", label: "Perplexity check" },
  { key: "accuracy", label: "Accuracy check" },
  { key: "done", label: "Done" },
];

/** Progress at a glance for a run that is going, paused or being evaluated (docs/APP_DESIGN.md §2.2 step 5):
 * the stage, size against the target, quality so far, rounds, time left, and pause or continue. */
function RunProgress({ run, onChange }: { run: RunDetail; onChange: () => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [check, setCheck] = useState<Job | null>(null); // a test started on its own for this run
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    const read = () => api<Job[]>("/api/jobs").then((js) => {
      setJob(js.find((j) => j.run_id === run.id && (j.kind === "start" || j.kind === "resume")) ?? null);
      setCheck(js.find((j) => j.run_id === run.id && (j.kind === "heldout" || j.kind === "accuracy")
        && ["queued", "running"].includes(j.status)) ?? null);
    }).catch(() => undefined);
    read();
    const t = setInterval(read, 5000);
    return () => clearInterval(t);
  }, [run.id]);

  const active = (job && ["queued", "running", "stopped", "failed"].includes(job.status)) || !!check;
  if (run.state === "finished" && !active) return null;
  const orig = run.original_mb ?? 0;
  const now = run.now_mb ?? orig;
  const toCut = Math.max(1, orig - run.limit_mb);
  const share = Math.min(1, Math.max(0, (orig - now) / toCut));
  const stepName = run.state !== "finished" ? "run" : check ? check.kind
    : job && job.status !== "done" ? job.steps?.[job.step ?? 0]?.name ?? "done" : "done";
  const at = Math.max(0, STAGES.findIndex((x) => x.key === stepName));
  const paused = run.state === "paused" || run.state === "interrupted" || job?.status === "stopped" || job?.status === "failed";
  const status = job?.status === "queued" ? "Waiting for the Mac to be free"
    : job?.status === "failed" ? `Stopped: ${job.reason ?? "an error"}`
    : paused ? `Paused${run.reason || job?.reason ? `: ${job?.reason ?? run.reason}` : ""}`
    : run.state === "running" ? (run.time_left ? timeLeftText(run.time_left) : "Running")
    : check?.status === "queued" ? `${STAGES[at].label}: waiting for the Mac to be free`
    : `${STAGES[at].label}: running${testLeft(check?.output ?? job?.output ?? "")}`;
  const act = (what: "pause" | "resume") => {
    setBusy(true);
    api(`/api/runs/${run.id}/${what}`, { method: "POST" }).then(onChange).catch(() => undefined).finally(() => setBusy(false));
  };
  const loss = run.now_error && run.original_error ? qualityLoss(run.now_error, run.original_error) : null;
  return (
    <section className="progress-view">
      <StepBar steps={STAGES.map((x) => x.label)} at={at + 1} />
      <div className={`pv-grid ${run.state === "running" && run.now ? "with-now" : ""}`}>
        <div className="pv-size">
          <div className="g-label">Size</div>
          <div className="pv-big">{mb(now)} <span className="muted">→ under {mb(run.limit_mb)}</span></div>
          <div className="bar"><div className="fill" style={{ width: `${share * 100}%` }} /></div>
          <div className="g-note">{share >= 1 ? "Target met ✓" : `${mb(now - run.limit_mb)} still to cut · ${Math.round(share * 100)}% of the way`}</div>
        </div>
        <div>
          <div className="g-label">Quality so far</div>
          <div className="pv-big">{loss == null ? "—" : lossText(loss)}</div>
          <div className="g-note">vs. the original</div>
        </div>
        <div>
          {run.search === "layer_by_layer" ? (
            <>
              <div className="g-label">Layers decided</div>
              <div className="pv-big">{run.rounds.filter((r) => ["ACCEPT", "ROLLBACK", "LEFT"].includes(r.outcome ?? "")).length} <span className="muted">of {run.settings.max_rounds}</span></div>
            </>
          ) : (
            <>
              <div className="g-label">Rounds</div>
              <div className="pv-big">{run.round} <span className="muted">of up to {run.settings.max_rounds}</span></div>
            </>
          )}
          <div className="g-note">{run.kept_steps} steps kept</div>
        </div>
        {run.state === "running" && run.now && (
          <div>
            <div className="g-label">Now</div>
            <div className="pv-big">{run.now.layer != null ? `Layer ${run.now.layer}` : `Round ${run.now.round}`}</div>
            <div className="g-note">
              {run.now.phase === "measuring" ? `Measuring ${run.now.plans ?? ""} plan${run.now.plans === 1 ? "" : "s"}` : "AI proposing"}
            </div>
          </div>
        )}
      </div>
      <div className="pv-foot">
        <span className={paused ? "warn" : "muted"}>{status}</span>
        {check ? null : job?.status === "running" || run.state === "running"
          ? <button className="secondary" disabled={busy} onClick={() => act("pause")}>Pause</button>
          : paused ? <button className="primary" disabled={busy} onClick={() => act("resume")}>Continue</button> : null}
      </div>
    </section>
  );
}

/** "· this model: about 2 h 40 min left", from the test's own progress line (it tests the models one at a time). */
function testLeft(output: string): string {
  const m = [...output.matchAll(/(\d+)\/(\d+) \[[\d:]+<(\d+):(\d+)(?::(\d+))?/g)].pop();
  if (!m) return "";
  const [h, mm] = m[5] !== undefined ? [Number(m[3]), Number(m[4])] : [0, Number(m[3])];
  return ` · this model: about ${duration(h * 60 + mm)} left`;
}

/** Your model (docs/APP_DESIGN.md §2.2 step 7): the smaller model to download, built by pruning's last step. */
function YourModel({ run, onChange }: { run: RunDetail; onChange: () => void }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const building = jobs.some((j) => j.run_id === run.id && ["queued", "running"].includes(j.status)
    && (j.kind === "save" || j.steps?.[j.step ?? 0]?.name === "save"));
  const failed = !run.has_model && jobs.find((j) => j.run_id === run.id && j.kind === "save")?.status === "failed";
  useEffect(() => {
    const read = () => api<Job[]>("/api/jobs").then(setJobs).catch(() => undefined);
    read();
    if (run.has_model) return;
    const t = setInterval(() => { read(); onChange(); }, 5000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.id, run.has_model]);
  const build = () => api(`/api/runs/${run.id}/save`, { method: "POST" })
    .then(() => api<Job[]>("/api/jobs").then(setJobs)).catch(() => undefined);
  const name = modelName(run.model);
  return (
    <section className="your-model">
      <h2>Your model</h2>
      <div className="ym-card">
        <div>
          <div className="eval-title">{name}, {run.cut != null ? pct(run.cut) : ""} smaller</div>
          <div className="muted">{mb(run.now_mb ?? 0)} instead of {mb(run.original_mb ?? 0)}{run.now_mb != null && run.now_mb <= run.limit_mb ? " · fits the target ✓" : ""}</div>
        </div>
        {run.has_model ? (
          <a className="primary" href={`/api/runs/${run.id}/download/model`} download>
            Download{run.model_mb ? ` (${mb(run.model_mb)})` : ""}
          </a>
        ) : building ? (
          <span className="badge running">Building the download…</span>
        ) : (
          <button className="secondary" onClick={build}>{failed ? "Build it again" : "Build the download"}</button>
        )}
      </div>
      {run.has_model && (
        <details className="sets">
          <summary>Instructions</summary>
          <pre className="code">{`from pruneagent import loader
from transformers import AutoTokenizer

model = loader.load("${run.id}-model", device="mps")   # the unzipped folder; "cuda" or "cpu" elsewhere
tokenizer = AutoTokenizer.from_pretrained("${run.model}")`}</pre>
          <p className="muted">The folder also has the plan (which parts of each layer were kept) and a note on how it was made. <a href={`/api/runs/${run.id}/download/plan`}>Plan only (.json)</a></p>
        </details>
      )}
    </section>
  );
}

const SET_NAMES: Record<string, [string, string]> = {
  hellaswag: ["HellaSwag", "the most sensible ending for an everyday situation"],
  boolq: ["BoolQ", "yes/no questions about a passage"],
  arc_easy: ["ARC-Easy", "grade-school science"],
  winogrande: ["WinoGrande", "who “he” or “it” refers to"],
  arc_challenge: ["ARC-Challenge", "harder grade-school science"],
  openbookqa: ["OpenBookQA", "science facts plus common sense"],
  rte: ["RTE", "does one sentence imply another"],
};

type EvalState = "done" | "running" | "waiting" | "failed" | "none";

/** Evaluation, the step after the run (docs/APP_DESIGN.md §2.2): the unseen-text test (perplexity) and the quiz
 * questions (accuracy), each against the original model and the standard method at the same size. */
function Evaluation({ runId, results, onResults }: {
  runId: string;
  results: Results | null;
  onResults: (r: Results) => void;
}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const refresh = () => {
    api<Job[]>("/api/jobs").then(setJobs).catch(() => undefined);
    api<Results>(`/api/runs/${runId}/results`).then(onResults).catch(() => undefined);
  };
  useEffect(() => {
    api<Job[]>("/api/jobs").then(setJobs).catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);
  const stateOf = (kind: "heldout" | "accuracy", done: boolean): EvalState => {
    if (done) return "done";
    // a check of its own, or the matching step of this run's pruning workflow
    const j = jobs.find((x) => x.run_id === runId && (x.kind === kind
      || ((x.kind === "start" || x.kind === "resume") && x.steps?.some((st) => st.name === kind && st.state !== "done"))));
    if (!j) return "none";
    if (j.status === "running" && (j.kind === kind || j.steps?.[j.step ?? 0]?.name === kind)) return "running";
    if (j.status === "queued" || j.status === "running") return "waiting";
    if (j.status === "failed") return "failed";
    return "none";
  };
  const text = stateOf("heldout", !!results?.heldout);
  const quiz = stateOf("accuracy", !!results?.accuracy);
  const busy = [text, quiz].some((x) => x === "running" || x === "waiting");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, runId]);
  const start = (kind: string) => api(`/api/runs/${runId}/checks/${kind}`, { method: "POST" }).then(refresh).catch(() => undefined);
  return (
    <section className="evaluation">
      <h2>Evaluation</h2>
      <div className="eval-grid">
        <EvalCard title="Perplexity check" what="Unseen text" state={text} onRun={() => start("heldout")}>
          {results?.heldout && <ResultBars held={results.heldout} />}
        </EvalCard>
        <EvalCard title="Accuracy check" what="18,904 questions" state={quiz} onRun={() => start("accuracy")}>
          {results?.accuracy && <QuizResult acc={results.accuracy} />}
        </EvalCard>
      </div>
    </section>
  );
}

function EvalCard({ title, what, state, onRun, children }: {
  title: string;
  what: string;
  state: EvalState;
  onRun: () => void;
  children: React.ReactNode;
}) {
  const badge = { done: ["done", "Done ✓"], running: ["running", "Running"], waiting: ["", "Waiting for the Mac"],
                  failed: ["failed", "Didn't finish"], none: ["", "Not run"] }[state];
  return (
    <div className="eval-card">
      <div className="eval-head">
        <div>
          <div className="eval-title">{title}</div>
          <div className="muted">{what}</div>
        </div>
        <span className={`badge ${badge[0]}`}>{badge[1]}</span>
      </div>
      {state === "done" ? children
        : state === "running" ? <p className="muted">Running</p>
        : state === "waiting" ? <p className="muted">Starts when the Mac is free.</p>
        : <button className="secondary" onClick={onRun}>{state === "failed" ? "Run it again" : "Run it"}</button>}
    </div>
  );
}

function QuizResult({ acc }: { acc: NonNullable<Results["accuracy"]> }) {
  const rows = [
    { name: "Original", cls: "orig", r: acc.unpruned },
    { name: "PruneAgent", cls: "agent", r: acc.agent },
    { name: "Standard method (FLAP)", cls: "flap", r: acc["Baseline: FLAP"] },
  ].filter((x) => x.r?.all_questions);
  const sets = Object.keys(acc.agent?.sets ?? {});
  return (
    <div className="bars">
      {rows.map((x) => (
        <div key={x.name} className="bar-row">
          <span className="bar-name">{x.name}</span>
          <span className="bar-track"><span className={`bar-fill ${x.cls}`} style={{ width: `${x.r!.all_questions.acc * 100}%` }} /></span>
          <span className="bar-val">{pct(x.r!.all_questions.acc, 1)} right</span>
        </div>
      ))}
      <p className="note">Weighted by questions per set · longer is better</p>
      <details className="sets">
        <summary>Each question set</summary>
        <table>
          <thead><tr><th>Set</th>{rows.map((x) => <th key={x.name}>{x.name}</th>)}</tr></thead>
          <tbody>
            {sets.map((k) => (
              <tr key={k}>
                <td><b>{SET_NAMES[k]?.[0] ?? k}</b> <span className="muted">{SET_NAMES[k]?.[1] ?? ""}</span></td>
                {rows.map((x) => <td key={x.name}>{x.r!.sets[k] ? pct(x.r!.sets[k].acc, 1) : "—"}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
