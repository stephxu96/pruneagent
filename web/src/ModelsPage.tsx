import { useEffect, useState } from "react";
import { api, duration, durationRange, mb, modelName } from "./lib";

// Bring your own model (FR17), as a guided flow: choose a model, check it, prepare it on this computer, then
// prune it. The flow follows the server's state, so leaving and coming back shows where things stand.

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
const STEPS = ["Choose", "Check", "Prepare", "Ready"];
// the preparation's three parts, in the order the tool runs them (text; scores and quick tests; damage table)
const PARTS = [
  { title: "Sample text", text: "Download text samples and split them into pieces the way this model reads." },
  { title: "Part usage", text: "Run the samples through the model and see how much it uses each part." },
  { title: "Cut tests", text: "Try each kind of cut on each layer and measure how much quality drops." },
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

  // the latest preparation job for each model (jobs come newest first)
  const latest = new Map<string, PrepJob>();
  for (const j of jobs ?? []) if (j.params.model && !latest.has(j.params.model)) latest.set(j.params.model, j);

  // open on what's happening: the model named in the address, else a preparation under way
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

  const row = rows.find((r) => r.id === selected);
  const job = selected ? latest.get(selected) : undefined;
  const ready = !!(row?.ready || (check?.ok && check.prepared));
  const step = !selected ? 1 : ready ? 4 : job ? 3 : 2;

  return (
    <div className="models-page">
      <a href="#/" className="back">← All runs</a>
      <h1>Select your model</h1>

      <ol className="stepper">
        {STEPS.map((s, i) => (
          <li key={s} className={i + 1 < step ? "done" : i + 1 === step ? "current" : ""}>
            <span className="dot">{i + 1 < step ? "✓" : i + 1}</span>
            <span>{s}</span>
          </li>
        ))}
      </ol>

      <section className="wizard">
        {step === 1 && (
          <ChooseStep name={name} setName={setName} rows={rows} onChoose={choose} />
        )}
        {step === 2 && (
          checking || !check ? <p className="muted">Checking {selected}…</p>
            : <CheckStep check={check} onPrepare={() => prepare(check.id)} onBack={restart} />
        )}
        {step === 3 && job && selected && (
          <PrepareStep id={selected} job={job} onPause={() => pause(job)} onContinue={() => prepare(selected)} />
        )}
        {step === 4 && selected && (
          <div className="ready-step">
            <p className="ready-big">✓ {modelName(selected)} is ready for pruning.</p>
            <div className="actions">
              <a className="primary" href={`#/new?model=${encodeURIComponent(selected)}`}>Make it smaller →</a>
              <button className="secondary" onClick={restart}>Choose another model</button>
            </div>
          </div>
        )}
        {msg && <p className="warn">{msg}</p>}
      </section>
    </div>
  );
}

function ChooseStep({ name, setName, rows, onChoose }: {
  name: string;
  setName: (s: string) => void;
  rows: ModelRow[];
  onChoose: (id: string) => void;
}) {
  return (
    <div className="choose-step">
      <label>
        <span className="m-label">Model name on Hugging Face</span>
        <div className="add-row">
          <input type="text" value={name} placeholder="e.g. HuggingFaceTB/SmolLM2-360M" spellCheck={false} autoFocus
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") onChoose(name); }} />
          <button className="primary" onClick={() => onChoose(name)} disabled={!name.trim()}>Check →</button>
        </div>
      </label>
      <p className="muted">Works with Qwen3, Qwen2, Llama and Mistral models.</p>
      {rows.length > 0 && (
        <div className="on-computer">
          <span className="m-label">Or one already on this computer</span>
          <div className="chips">
            {rows.map((r) => (
              <button key={r.id} className="chip" onClick={() => onChoose(r.id)}>
                {modelName(r.id)} <span className="muted">· {mb(r.size_mb)}{r.ready ? " · ready ✓" : ""}</span>
              </button>
            ))}
          </div>
        </div>
      )}
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
          <div className="f-label">Preparation time</div>
          <div className="f-big">{durationRange(lo, hi)}</div>
          <div className="f-sub">Once, on this computer</div>
        </div>
      </div>
      <div className="actions">
        <button className="primary" onClick={onPrepare}>Start preparing</button>
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
      <p className="prep-head">Preparing <b>{modelName(id)}</b> for pruning</p>
      <ol className="checklist">
        {PARTS.map((part, i) => {
          const n = i + 1;
          const state = n < at || (n === at && !active && job.status === "done") ? "done" : n === at ? "current" : "todo";
          return (
            <li key={part.title} className={state}>
              <span className="mark">{state === "done" ? "✓" : n}</span>
              <div>
                <div className="c-title">{part.title}</div>
                <div className="c-text">{part.text}</div>
                {state === "current" && n === 3 && p && p.total > 0 && (
                  <div className="c-count">{p.done} of {p.total} cuts tested</div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
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
            : <button className="primary" onClick={onContinue}>Continue preparing</button>}
        </div>
      </div>
      <details className="raw">
        <summary>System logs</summary>
        <pre>{job.output || "(nothing yet)"}</pre>
      </details>
    </div>
  );
}
