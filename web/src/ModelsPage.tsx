import { useEffect, useState } from "react";
import { api, duration, durationRange, mb, modelName } from "./lib";
import { StepBar } from "./parts";

// Model setup (docs/APP_DESIGN.md §2.1, FR17): put a model on this computer, ready to prune. Name it, check it, set it
// up (load the model, generate the assets pruning needs), ready. Pruning is a separate story (PrunePage). The flow
// follows the workflow service's state, so leaving and coming back shows where things stand.

type Check =
  | { id: string; ok: false; reason: string }
  | { id: string; ok: true; family: string; size_mb: number; layers: number; prepare_minutes: [number, number]; prepared: boolean; added: boolean };

type Progress = { step: number; of: number; label: string; done: number; total: number; share: number };
type PrepJob = {
  id: number; kind: string; status: string; started: string | null; output: string;
  params: { model?: string }; progress?: Progress;
};
type ModelRow = { id: string; ready: boolean; built_in: boolean; size_mb: number };

const ACTIVE = ["queued", "running"];
const STEPS = ["Name", "Check", "Set up", "Ready"];
// setup's three parts, in the order the tool runs them (text; scores and quick tests; damage table)
// setup's two parts and their steps, in the order the tools run them (`pruneagent setup`, steps 1-4)
const PARTS = [
  { group: "Load the model", title: "Model", text: "Download the model from Hugging Face onto this computer." },
  { group: "Generate assets", title: "Sample text", text: "Download text samples and split them into pieces the way this model reads." },
  { group: "Generate assets", title: "Part usage", text: "Run the samples through the model and see how much it uses each part." },
  { group: "Generate assets", title: "Cut tests", text: "Try each kind of cut on each layer and measure how much quality drops." },
];

export function ModelsPage() {
  const asked = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("check") ?? "";
  const [name, setName] = useState(asked);
  const [selected, setSelected] = useState<string | null>(null);
  const [check, setCheck] = useState<Check | null>(null);
  const [checking, setChecking] = useState(false);
  const [rows, setRows] = useState<ModelRow[]>([]);
  const [jobs, setJobs] = useState<PrepJob[] | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = () => {
    api<{ models: ModelRow[] }>("/api/machine").then((x) => setRows(x.models)).catch(() => undefined);
    api<PrepJob[]>("/api/jobs").then((js) => setJobs(js.filter((j) => j.kind === "prepare"))).catch(() => undefined);
  };
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  const choose = (id: string) => {
    id = id.trim();
    if (!id) return;
    setName(id);
    setSelected(id);
    setChecking(true);
    setCheck(null);
    setMsg(null);
    api<Check>(`/api/models/check?id=${encodeURIComponent(id)}`).then(setCheck).catch((e) => setMsg(String(e)))
      .finally(() => setChecking(false));
  };
  const restart = () => {
    setSelected(null);
    setCheck(null);
    setMsg(null);
  };

  // the latest setup for each model (jobs come newest first)
  const latest = new Map<string, PrepJob>();
  for (const j of jobs ?? []) if (j.params.model && !latest.has(j.params.model)) latest.set(j.params.model, j);

  // open on what's happening: the model named in the address, else a setup under way
  useEffect(() => {
    if (jobs === null || selected) return;
    if (asked) choose(asked);
    else {
      const running = (jobs ?? []).find((j) => ACTIVE.includes(j.status) && j.params.model);
      if (running?.params.model) choose(running.params.model);
    }
  }, [jobs]);

  const prepare = (id: string) =>
    api<PrepJob>("/api/models/prepare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: id }) })
      .then(() => { setMsg(null); refresh(); })
      .catch((e) => setMsg(String(e)));
  const pause = (job: PrepJob) => api(`/api/jobs/${job.id}/stop`, { method: "POST" }).then(refresh);
  const resume = (job: PrepJob) => api(`/api/jobs/${job.id}/continue`, { method: "POST" }).then(refresh);

  const row = rows.find((r) => r.id === selected);
  const job = selected ? latest.get(selected) : undefined;
  const ready = !!(row?.ready || (check?.ok && check.prepared));
  const step = !selected ? 1 : ready ? 4 : job ? 3 : 2;

  return (
    <div className="models-page">
      <a href="#/" className="back">← All runs</a>
      <h1>Models</h1>
      <ModelList rows={rows} latest={latest} onOpen={choose} />
      <h2>Set up a model</h2>

      <StepBar steps={STEPS} at={step} />

      <section className="wizard">
        {step === 1 && (
          <ChooseStep name={name} setName={setName} onChoose={choose} />
        )}
        {step === 2 && (
          checking || !check ? <p className="muted">Checking {selected}…</p>
            : <CheckStep check={check} onPrepare={() => prepare(check.id)} onBack={restart} />
        )}
        {step === 3 && job && selected && (
          <PrepareStep id={selected} job={job} onPause={() => pause(job)} onContinue={() => resume(job)} />
        )}
        {step === 4 && selected && (
          <div className="ready-step">
            <p className="ready-big">✓ {modelName(selected)} is ready for pruning.</p>
            <div className="actions">
              <a className="primary" href={`#/new?model=${encodeURIComponent(selected)}`}>Prune it →</a>
              <button className="secondary" onClick={restart}>Set up another model</button>
            </div>
          </div>
        )}
        {msg && <p className="warn">{msg}</p>}
      </section>
    </div>
  );
}

function ChooseStep({ name, setName, onChoose }: {
  name: string;
  setName: (s: string) => void;
  onChoose: (id: string) => void;
}) {
  return (
    <div className="choose-step">
      <label>
        <span className="m-label">Model name on Hugging Face</span>
        <div className="add-row">
          <input type="text" value={name} placeholder="e.g. HuggingFaceTB/SmolLM2-360M" spellCheck={false}
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") onChoose(name); }} />
          <button className="primary" onClick={() => onChoose(name)} disabled={!name.trim()}>Check →</button>
        </div>
      </label>
      <p className="muted">Works with Qwen3, Qwen2, Llama and Mistral models.</p>
    </div>
  );
}

/** The models on this computer: ready ones can be pruned; others show their setup. */
function ModelList({ rows, latest, onOpen }: { rows: ModelRow[]; latest: Map<string, PrepJob>; onOpen: (id: string) => void }) {
  return (
    <div className="model-list">
      {rows.map((r) => {
        const job = latest.get(r.id);
        const active = job && ACTIVE.includes(job.status);
        const paused = job && ["stopped", "failed"].includes(job.status);
        return (
          <div key={r.id} className="model-line">
            <span className="ml-name">{modelName(r.id)}</span>
            <span className="muted">{mb(r.size_mb)}{r.built_in ? "" : " · added by you"}</span>
            <span className="ml-state">
              {r.ready ? (
                <><span className="badge done">Ready ✓</span><a href={`#/new?model=${encodeURIComponent(r.id)}`}>Prune it →</a></>
              ) : active ? (
                <><span className="badge running">Setting up · {Math.round((job.progress?.share ?? 0) * 100)}%</span>
                  <button className="small" onClick={() => onOpen(r.id)}>Show</button></>
              ) : (
                <><span className="badge stopped">{paused ? "Setup paused" : "Not set up"}</span>
                  <button className="small" onClick={() => onOpen(r.id)}>{paused ? "Continue setup" : "Set up"}</button></>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function CheckStep({ check, onPrepare, onBack }: { check: Check; onPrepare: () => void; onBack: () => void }) {
  if (!check.ok) {
    return (
      <div className="feasible no">
        <div className="f-top">
          <span className="f-badge">✕ Can't be used</span>
          <span className="f-title">{check.id}</span>
        </div>
        <div className="f-reason">{check.reason}</div>
        <div className="actions"><button className="secondary" onClick={onBack}>← Choose another model</button></div>
      </div>
    );
  }
  const [lo, hi] = check.prepare_minutes;
  return (
    <div className="feasible yes">
      <div className="f-top">
        <span className="f-badge">✓ Can be used</span>
        <span className="f-title">{check.id}</span>
      </div>
      <div className="f-tiles">
        <div className="f-tile">
          <div className="f-label">Kind</div>
          <div className="f-big">{check.family}</div>
        </div>
        <div className="f-tile">
          <div className="f-label">Size</div>
          <div className="f-big">{mb(check.size_mb)}</div>
        </div>
        <div className="f-tile">
          <div className="f-label">Setup time</div>
          <div className="f-big">{durationRange(lo, hi)}</div>
          <div className="f-sub">Once, on this computer</div>
        </div>
      </div>
      <div className="actions">
        <button className="primary" onClick={onPrepare}>Set up this model</button>
        <button className="secondary" onClick={onBack}>← Choose another model</button>
      </div>
    </div>
  );
}

function timeLeft(job: PrepJob): string | null {
  const p = job.progress;
  if (!p || !job.started || p.share < 0.12) return null; // too early to tell
  const elapsed = (Date.now() - new Date(job.started).getTime()) / 60000;
  return `about ${duration((elapsed * (1 - p.share)) / p.share)} left`;
}

function PrepareStep({ id, job, onPause, onContinue }: {
  id: string;
  job: PrepJob;
  onPause: () => void;
  onContinue: () => void;
}) {
  const p = job.progress;
  const active = ACTIVE.includes(job.status);
  const at = p?.step ?? 0; // the part under way (1-3); 0 = not started
  const left = active ? timeLeft(job) : null;
  return (
    <div className="prepare-step">
      <p className="prep-head">Setting up <b>{modelName(id)}</b></p>
      {["Load the model", "Generate assets"].map((group) => (
        <div key={group} className="check-group">
          <div className="m-label">{group}</div>
          <ol className="checklist">
            {PARTS.map((part, i) => ({ part, n: i + 1 })).filter(({ part }) => part.group === group).map(({ part, n }) => {
              const state = n < at || (n === at && !active && job.status === "done") ? "done" : n === at ? "current" : "todo";
              return (
                <li key={part.title} className={state}>
                  <span className="mark">{state === "done" ? "✓" : n}</span>
                  <div>
                    <div className="c-title">{part.title}</div>
                    <div className="c-text">{part.text}</div>
                    {state === "current" && n === PARTS.length && p && p.total > 0 && (
                      <div className="c-count">{p.done} of {p.total} cuts tested</div>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        </div>
      ))}
      <div className="prep-foot">
        <div className="bar"><div className="fill" style={{ width: `${Math.round((p?.share ?? 0) * 100)}%` }} /></div>
        <div className="prep-status">
          <span className="muted">
            {job.status === "queued" ? "Waiting for the computer's graphics chip to be free"
              : active ? `${Math.round((p?.share ?? 0) * 100)}% done${left ? ` · ${left}` : ""}`
              : job.status === "failed" ? "Stopped with an error. See the system logs below."
              : "Paused. Continuing picks up where it stopped."}
          </span>
          {active
            ? <button className="secondary" onClick={onPause}>{job.status === "running" ? "Pause" : "Cancel"}</button>
            : <button className="primary" onClick={onContinue}>Continue setup</button>}
        </div>
      </div>
      <details className="raw">
        <summary>System logs</summary>
        <pre>{job.output || "(nothing yet)"}</pre>
      </details>
    </div>
  );
}
