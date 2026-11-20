import { useEffect, useState } from "react";
import { api, duration, durationRange, mb, modelName } from "./lib";

// Bring your own model (FR17): check a Hugging Face model, prepare it once on this computer, then run it.

type Check =
  | { id: string; ok: false; reason: string }
  | { id: string; ok: true; family: string; size_mb: number; layers: number; prepare_minutes: [number, number]; prepared: boolean; added: boolean };

type Progress = { step: number; of: number; label: string; done: number; total: number; share: number };
type PrepJob = { id: number; kind: string; status: string; started: string | null; params: { model?: string }; progress?: Progress };
type ModelRow = { id: string; ready: boolean; built_in: boolean; size_mb: number };

const ACTIVE = ["queued", "running"];
// the preparation's steps in plain words (the tool's own names: text, scores and quick tests, damage table)
const STEP_TEXT: Record<number, string> = {
  1: "Getting the text ready",
  2: "Measuring how much each part is used",
  3: "Testing each cut",
};

export function ModelsPage() {
  const asked = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("check") ?? "";
  const [name, setName] = useState(asked);
  const [check, setCheck] = useState<Check | null>(null);
  const [checking, setChecking] = useState(false);
  const [rows, setRows] = useState<ModelRow[]>([]);
  const [jobs, setJobs] = useState<PrepJob[]>([]);
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

  const runCheck = (given?: string) => {
    const id = (given ?? name).trim();
    if (!id) return;
    setChecking(true);
    setCheck(null);
    setMsg(null);
    api<Check>(`/api/models/check?id=${encodeURIComponent(id)}`).then(setCheck).catch((e) => setMsg(String(e)))
      .finally(() => setChecking(false));
  };
  useEffect(() => {
    if (asked) runCheck(asked);
  }, []);
  const prepare = (id: string) =>
    api<PrepJob>("/api/models/prepare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: id }) })
      .then(() => { setMsg(null); refresh(); })
      .catch((e) => setMsg(String(e)));
  const pause = (job: PrepJob) => api(`/api/jobs/${job.id}/stop`, { method: "POST" }).then(refresh);

  // the latest preparation job for each model (jobs come newest first)
  const latest = new Map<string, PrepJob>();
  for (const j of jobs) if (j.params.model && !latest.has(j.params.model)) latest.set(j.params.model, j);
  const busy = (id: string) => ACTIVE.includes(latest.get(id)?.status ?? "");

  return (
    <div className="models-page">
      <a href="#/" className="back">← All runs</a>
      <h1>Select your model</h1>
      <p className="lead">Use a model from Hugging Face. It's measured once on this computer before its first run.</p>

      <section className="add-model">
        <label>
          <span className="m-label">Model name on Hugging Face</span>
          <div className="add-row">
            <input type="text" value={name} placeholder="e.g. HuggingFaceTB/SmolLM2-360M" spellCheck={false}
              onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") runCheck(); }} />
            <button className="secondary" onClick={() => runCheck()} disabled={checking || !name.trim()}>
              {checking ? "Checking…" : "Check"}
            </button>
          </div>
        </label>
        <p className="muted">Works with Qwen3, Qwen2, Llama and Mistral models.</p>
        {check && <CheckResult check={check} busy={busy(check.id)} onPrepare={() => prepare(check.id)} />}
        {msg && <p className="warn">{msg}</p>}
      </section>

      <h2>Models on this computer</h2>
      <div className="model-list">
        {rows.map((r) => (
          <ModelLine key={r.id} row={r} job={latest.get(r.id)} onPrepare={() => prepare(r.id)} onPause={pause} />
        ))}
      </div>
    </div>
  );
}

function CheckResult({ check, busy, onPrepare }: { check: Check; busy: boolean; onPrepare: () => void }) {
  if (!check.ok) {
    return (
      <div className="feasible no">
        <div className="f-top">
          <span className="f-badge">✕ Can't be used</span>
          <span className="f-title">{check.id}</span>
        </div>
        <div className="f-reason">{check.reason}</div>
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
          <div className="f-sub">{check.layers} layers</div>
        </div>
        <div className="f-tile">
          <div className="f-label">Preparation</div>
          <div className="f-big">{check.prepared ? "Done ✓" : durationRange(lo, hi)}</div>
          <div className="f-sub">{check.prepared ? "Ready for runs" : "Once, on this computer"}</div>
        </div>
      </div>
      <div className="actions">
        {check.prepared ? (
          <a className="primary" href={`#/new?model=${encodeURIComponent(check.id)}`}>Make it smaller →</a>
        ) : busy ? (
          <span className="muted">Preparing now. See below.</span>
        ) : (
          <button className="primary" onClick={onPrepare}>Prepare this model</button>
        )}
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

function ModelLine({ row, job, onPrepare, onPause }: {
  row: ModelRow;
  job: PrepJob | undefined;
  onPrepare: () => void;
  onPause: (j: PrepJob) => void;
}) {
  const active = job && ACTIVE.includes(job.status);
  const p = job?.progress;
  let state;
  if (row.ready) {
    state = (
      <>
        <span className="badge done">Ready ✓</span>
        <a href={`#/new?model=${encodeURIComponent(row.id)}`}>Make it smaller →</a>
      </>
    );
  } else if (active && job) {
    const left = timeLeft(job);
    state = (
      <>
        <div className="prep">
          <div className="bar"><div className="fill" style={{ width: `${Math.round((p?.share ?? 0) * 100)}%` }} /></div>
          <span className="muted">
            {job.status === "queued" ? "Waiting for the Mac to be free" : p && p.step > 0 ? `Step ${p.step} of ${p.of}: ${STEP_TEXT[p.step] ?? p.label}` : "Starting"}
            {p && p.total > 0 && ` · ${p.done} of ${p.total} tests`}
            {left && ` · ${left}`}
          </span>
        </div>
        <button className="small" onClick={() => onPause(job)}>{job.status === "running" ? "Pause" : "Cancel"}</button>
      </>
    );
  } else {
    const stopped = job && ["stopped", "lost", "failed"].includes(job.status);
    state = (
      <>
        <span className="badge">{job?.status === "failed" ? "Preparation didn't finish" : stopped ? "Paused" : "Not prepared"}</span>
        <button className="small" onClick={onPrepare}>{stopped ? "Continue preparing" : "Prepare"}</button>
      </>
    );
  }
  return (
    <div className="model-line">
      <span className="ml-name">{modelName(row.id)}</span>
      <span className="muted">{mb(row.size_mb)}{row.built_in ? "" : " · added by you"}</span>
      <span className="ml-state">{state}</span>
    </div>
  );
}
