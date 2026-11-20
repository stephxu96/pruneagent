import { useEffect, useMemo, useState } from "react";
import { MapLegend, ModelMap } from "./ModelMap";
import { Neurons3D } from "./Neurons3D";
import { PathChart } from "./PathChart";
import {
  api, changedLayers, changesText, lossText, mb, modelName, outcomeText, pct, qualityLoss, stateText, timeLeftText,
  type Results, type RunDetail,
} from "./lib";
import type { Job } from "./parts";

export function RunPage({ id, start = 0 }: { id: string; start?: number }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [at, setAt] = useState(start); // round being shown; 0 = the original model
  const [playing, setPlaying] = useState(false);
  const [showFlap, setShowFlap] = useState(false);
  const [follow, setFollow] = useState(true);
  const [threeD, setThreeD] = useState(true);

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
  useEffect(() => {
    if (!playing) return;
    if (at >= last) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setAt((a) => a + 1), 1100);
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
        <a href="#/" className="back">← All runs</a>
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
        <a href="#/" className="back">← All runs</a>
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

      <section className="player">
        <button className="primary" onClick={() => { if (at >= last) setAt(0); setPlaying(!playing); }}>
          {playing ? "Pause" : at >= last ? "Replay" : "Play"}
        </button>
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
                model + its notes while answering (MAC)
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
          <button className={threeD ? "on" : ""} onClick={() => setThreeD(true)}>3D neurons</button>
          <button className={threeD ? "" : "on"} onClick={() => setThreeD(false)}>Flat</button>
        </div>
        {threeD ? (
          <Neurons3D
            layers={showFlap && run.flap ? run.flap.layers : step.layers}
            prev={showFlap ? undefined : prev?.layers}
            full={{ mlp: run.shape.mlp_full, groups: run.shape.groups_full }}
            caption={showFlap ? "What the standard method (FLAP) cuts to reach the same size." : undefined}
          />
        ) : (
          <>
            <ModelMap
              layers={showFlap && run.flap ? run.flap.layers : step.layers}
              full={{ mlp: run.shape.mlp_full, groups: run.shape.groups_full }}
              changed={showFlap ? undefined : changed}
              caption={showFlap ? "What the standard method (FLAP) cuts to reach the same size: a bit from almost everywhere."
                : `${at === 0 ? "The original model" : `The model after round ${at}`}: ${run.shape.layers} layers (0 to ${run.shape.layers - 1}), left to right.`}
            />
            <MapLegend />
          </>
        )}
      </section>

      <section className="story">
        {round ? (
          <>
            <h2>
              Round {round.round}: <span className={`outcome ${round.outcome?.toLowerCase()}`}>{outcomeText[round.outcome ?? ""] ?? "no change"}</span>
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
            ) : (
              <p className="muted">{round.refused_reasons.join(" ") || "No ideas this round."}</p>
            )}
          </>
        ) : (
          <p className="muted">Press Play for the round-by-round replay.</p>
        )}
      </section>

      <section>
        <h2>All tested ideas</h2>
        <PathChart run={run} upTo={at} />
      </section>

      {run.state === "finished" && <Evaluation runId={id} results={results} onResults={setResults} />}
    </div>
  );
}

function Settings({ run, orig }: { run: RunDetail; orig: number }) {
  const s = run.settings;
  const rows: [string, string][] = [
    ["Model", modelName(run.model)],
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
      <p className="note">Quality lost compared with the original model; shorter is better. Both models are the same size.</p>
    </div>
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
      <p className="muted">Two tests of the finished model, each against the original and the standard method (FLAP) at the same size.</p>
      <div className="eval-grid">
        <EvalCard title="Unseen-text test" what="How well it predicts text it never saw (perplexity)" state={text} onRun={() => start("heldout")}>
          {results?.heldout && <ResultBars held={results.heldout} />}
        </EvalCard>
        <EvalCard title="Quiz questions" what="How many of 18,904 multiple-choice questions it gets right (accuracy)" state={quiz} onRun={() => start("accuracy")}>
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
        : state === "running" ? <p className="muted">Testing the original, this model and the standard method in turn. This page updates on its own.</p>
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
      <p className="note">Share of all questions answered right, each set counted by its number of questions; longer is better.</p>
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
