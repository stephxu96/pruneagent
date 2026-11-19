import { useEffect, useMemo, useState } from "react";
import { MapLegend, ModelMap } from "./ModelMap";
import { Neurons3D } from "./Neurons3D";
import { PathChart } from "./PathChart";
import {
  api, changedLayers, changesText, lossText, mb, modelName, outcomeText, pct, qualityLoss, stateText,
  type Results, type RunDetail,
} from "./lib";

export function RunPage({ id, start = 0 }: { id: string; start?: number }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [at, setAt] = useState(start); // round being shown; 0 = the original model
  const [playing, setPlaying] = useState(false);
  const [showFlap, setShowFlap] = useState(false);
  const [follow, setFollow] = useState(true);
  const [threeD, setThreeD] = useState(true);

  const load = () =>
    api<RunDetail>(`/api/runs/${id}`).then((r) => {
      setRun(r);
      if (follow && r.state === "running") setAt(r.rounds.length);
    }).catch((e) => setErr(String(e)));

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
  if (!run || !view) return <p className="muted">Loading…</p>;

  const base = run.original_error ?? 1;
  const orig = run.original_mb ?? 0;
  const { step, prev, round, changed } = view;
  const needed = orig - run.limit_mb;
  const savedNow = orig - step.size_mb;
  const progress = Math.min(1, savedNow / needed);
  const held = results?.heldout;

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
        <span className="where">{at === 0 ? "Start: the original model" : `Round ${at} of ${last}`}</span>
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
            <div className="g-note">{progress >= 1 ? "Fits the goal ✓" : `${mb(Math.max(0, step.size_mb - run.limit_mb))} still to cut`}</div>
          </div>
          <div className="gauge">
            <div className="g-label">Quality lost so far</div>
            <div className="g-value">{lossText(qualityLoss(step.error, base))}</div>
            <div className="g-note">than the original, at predicting text</div>
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
              <div className="g-label">Standard method (FLAP), cut to the goal</div>
              <div className="g-value">{lossText(qualityLoss(run.flap.error, base))}</div>
              <div className="g-note">
                <label><input type="checkbox" checked={showFlap} onChange={(e) => setShowFlap(e.target.checked)} /> show its cut</label>
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
                : `${at === 0 ? "The original model" : `The model after round ${at}`}: ${run.shape.layers} layers, left to right.`}
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
                      <summary>The AI's reason</summary>
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
          <p className="muted">Press Play to watch the AI cut the model down, round by round.</p>
        )}
      </section>

      <section>
        <h2>Every idea tested</h2>
        <PathChart run={run} upTo={at} />
      </section>

      {held && (
        <section>
          <h2>Final check on new text</h2>
          <p className="muted">After the run, both cut models were tested on text neither had seen.</p>
          <ResultBars held={held} />
          {results?.accuracy && <QuizLine acc={results.accuracy} />}
        </section>
      )}
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

function QuizLine({ acc }: { acc: NonNullable<Results["accuracy"]> }) {
  const a = acc.agent?.all_questions;
  const f = acc["Baseline: FLAP"]?.all_questions;
  if (!a || !f) return null;
  return (
    <p className="quiz">
      Quiz questions ({a.questions.toLocaleString()}): PruneAgent got <b>{pct(a.acc, 1)}</b> right, the standard method{" "}
      <b>{pct(f.acc, 1)}</b>.
    </p>
  );
}
