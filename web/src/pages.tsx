import { useEffect, useState } from "react";
import { ResultBars } from "./RunPage";
import { StartDialog } from "./StartDialog";
import { api, duration, durationRange, FINAL_CHECK_MIN, lossText, mb, modelName, pct, qualityLoss, runLabel, stateText, type Compare, type RunSummary } from "./lib";

// ---------------------------------------------------------------- home: every run

export function HomePage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api<RunSummary[]>("/api/runs").then((r) => setRuns([...r].sort((a, b) => (a.cut ?? 0) - (b.cut ?? 0)))).catch((e) => setErr(String(e)));
  }, []);
  return (
    <div>
      <section className="hero">
        <h1>Shrink an AI model to fit your computer.</h1>
      </section>
      <MachinePanel />
      <h2>Runs</h2>
      {err && <p className="error">{err}</p>}
      {runs && runs.length === 0 && <p className="muted">No runs yet.</p>}
      <div className="cards">
        {runs?.map((r) => (
          <a key={r.id} className="card" href={`#/run/${r.id}`}>
            <div className="card-title">{runLabel(r)}</div>
            <div className={`badge ${r.state}`}>{stateText[r.state] ?? r.state}</div>
            <dl>
              <dt>Size</dt>
              <dd>{r.original_mb && r.now_mb ? `${mb(r.original_mb)} → ${mb(r.now_mb)}` : "—"}</dd>
              <dt>Quality lost</dt>
              <dd>{r.now_error && r.original_error ? lossText(qualityLoss(r.now_error, r.original_error)) : "—"}</dd>
              <dt>Rounds</dt>
              <dd>{r.round}</dd>
            </dl>
            <span className="go">Replay →</span>
          </a>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- compare: PruneAgent vs FLAP at each size

export function ComparePage() {
  const [data, setData] = useState<Compare | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api<Compare>("/api/compare").then(setData).catch((e) => setErr(String(e)));
  }, []);
  if (err) return <p className="error">{err}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  return (
    <div>
      <a href="#/" className="back">← All runs</a>
      <h1>Quality at the same size</h1>
      <p className="lede">
        Each pair was cut to exactly the same size, then tested on text neither had seen. The standard method (FLAP)
        applies one fixed rule; PruneAgent's AI tries ideas and keeps what works.
      </p>
      <div className="cards wide">
        {data.runs.map((r) => (
          <div key={r.id} className="card static">
            <div className="card-title">
              {modelName(r.model)}, {pct(r.cut)} smaller <span className="muted">({mb(r.agent.size_mb)})</span>
            </div>
            <ResultBars held={{ unpruned: r.original, agent: r.agent, "Baseline: FLAP": r.flap }} />
            <a href={`#/run/${r.id}`}>Replay →</a>
          </div>
        ))}
      </div>
      {data.flap_curve.length > 0 && <CurveChart data={data} />}
    </div>
  );
}

function CurveChart({ data }: { data: Compare }) {
  const W = 640, H = 300, M = { l: 78, r: 24, t: 20, b: 44 };
  const base = data.runs[0]?.original.test_error ?? 1;
  const times = (e: number) => e / base; // 1 = as good as the original
  const xs = [0, ...data.flap_curve.map((p) => p.cut)];
  const xMax = Math.max(...xs, ...data.runs.map((r) => r.cut)) * 1.05;
  const yMax = Math.max(...data.flap_curve.map((p) => times(p.test_error)), 2) * 1.2;
  const sx = (x: number) => M.l + (x / xMax) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - (Math.log(y) / Math.log(yMax)) * (H - M.t - M.b);
  const yTicks = [1, 2, 5, 10, 20, 50].filter((t) => t <= yMax);
  return (
    <section>
      <h2>Quality by size cut</h2>
      <p className="muted">How many times worse than the original model, at each size cut (lower is better).</p>
      <figure className="chart">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Quality lost by size cut, both methods">
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} className="grid" />
              <text x={M.l - 8} y={sy(t) + 4} textAnchor="end" className="tick">{t === 1 ? "original" : `${t}×`}</text>
            </g>
          ))}
          {data.flap_curve.map((p) => (
            <text key={p.cut} x={sx(p.cut)} y={H - M.b + 18} textAnchor="middle" className="tick">{pct(p.cut, p.cut * 100 % 1 ? 1 : 0)}</text>
          ))}
          <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} className="axis" />
          <line x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} className="axis" />
          <text x={(M.l + W - M.r) / 2} y={H - 6} textAnchor="middle" className="axis-title">Size cut →</text>
          <polyline className="flap-line" points={[{ cut: 0, test_error: base }, ...data.flap_curve].map((p) => `${sx(p.cut)},${sy(times(p.test_error))}`).join(" ")} />
          {data.flap_curve.map((p) => (
            <circle key={p.cut} cx={sx(p.cut)} cy={sy(times(p.test_error))} r={4} className="flap-dot">
              <title>{`Standard method, ${pct(p.cut, 1)} smaller: ${times(p.test_error).toFixed(1)}× worse`}</title>
            </circle>
          ))}
          <polyline className="agent-line" points={[{ cut: 0, e: base }, ...data.runs.map((r) => ({ cut: r.cut, e: r.agent.test_error }))].map((p) => `${sx(p.cut)},${sy(times(p.e))}`).join(" ")} />
          {data.runs.map((r) => (
            <circle key={r.id} cx={sx(r.cut)} cy={sy(times(r.agent.test_error))} r={6} className="agent-dot">
              <title>{`PruneAgent, ${pct(r.cut)} smaller: ${times(r.agent.test_error).toFixed(1)}× worse`}</title>
            </circle>
          ))}
        </svg>
        <div className="legend">
          <span><i className="sw agent" /> PruneAgent</span>
          <span><i className="sw flap" /> Standard method (FLAP)</span>
        </div>
      </figure>
      <p className="note">The standard method breaks down past about 15% smaller. PruneAgent keeps working.</p>
    </section>
  );
}

// ---------------------------------------------------------------- new run

type Feasibility = {
  feasible: boolean;
  stop_reason: string | null;
  warnings: string[];
  usage: {
    calls: number; calls_fit: number | null; fits: boolean | null;
    windows?: { name: string; used_pct: number; pct_per_call: number; resets_at: string | null }[];
  } | null;
  limit_mb: number;
  original_mb?: number;
  smallest_mb?: number;
  time?: { minutes: [number, number]; rounds: [number, number]; based_on_runs: number; worst_minutes?: number; max_rounds?: number } | null;
};
type Job = {
  id: number; kind: string; reason?: string | null; params: { model?: string; cut?: number; run_id?: string }; run_id: string | null; status: string;
  created: string; output: string; result: Feasibility | null;
  progress?: { step: number; of: number; label: string; done: number; total: number; share: number };
};

const ADD_MODEL = "__add__"; // the model pickers' last option: go to "Select your model"

export function NewRunPage() {
  const q = new URLSearchParams(window.location.hash.split("?")[1] ?? "");
  const [model, setModel] = useState(q.get("model") ?? "Qwen/Qwen3-0.6B-Base");
  const [cut, setCut] = useState(Math.round(Number(q.get("cut") ?? 0.1) * 100));
  const [jobs, setJobs] = useState<Job[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [check, setCheck] = useState<Job | null>(null);
  const [asking, setAsking] = useState(false);
  useEffect(() => {
    if (!check || !["queued", "running"].includes(check.status)) return;
    const t = setTimeout(() => api<Job>(`/api/jobs/${check.id}`).then(setCheck).catch(() => undefined), 1500);
    return () => clearTimeout(t);
  }, [check]);
  const [models, setModels] = useState<Machine["models"]>([]);
  const refresh = () => api<Job[]>("/api/jobs").then(setJobs).catch(() => undefined);
  useEffect(() => {
    api<Machine>("/api/machine").then((x) => setModels(x.models)).catch(() => undefined);
  }, []);
  const info = models.find((x) => x.id === model);
  const ready = info?.ready ?? true;
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);
  const body = JSON.stringify({ model, cut: cut / 100, backend: "codex" });
  const post = (path: string) =>
    api<Job>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body })
      .then((j) => { setMsg(`Queued (job ${j.id}).`); refresh(); })
      .catch((e) => setMsg(String(e)));
  const orig = info?.size_mb;
  const sameRun = (j: Job) => j.kind === "precheck" && j.status === "done" && !!j.result && j.params?.model === model
    && Math.abs((j.params?.cut ?? -1) - cut / 100) < 1e-6;
  const lastCheck = [check, ...jobs].find((j): j is Job => !!j && sameRun(j))?.result ?? null;
  const runCheck = () => {
    setMsg(null);
    api<Job>("/api/precheck", { method: "POST", headers: { "Content-Type": "application/json" }, body })
      .then(setCheck).catch((e) => setMsg(String(e)));
  };
  return (
    <div>
      <a href="#/" className="back">← All runs</a>
      <h1>New run</h1>
      <div className="form">
        <label>
          Model
          <select value={model} onChange={(e) => {
            if (e.target.value === ADD_MODEL) window.location.hash = "#/models";
            else setModel(e.target.value);
          }}>
            {(models.length ? models : [{ id: model, ready: true, built_in: true, size_mb: 0, working_mb: 0 }]).map((x) => (
              <option key={x.id} value={x.id}>{modelName(x.id)}{x.ready ? "" : " (not prepared)"}</option>
            ))}
            <option value={ADD_MODEL}>Add your own model…</option>
          </select>
        </label>
        <label>
          How much smaller? <b>{cut}%</b>
          <input type="range" min={1} max={60} step={1} value={cut} onChange={(e) => setCut(Number(e.target.value))} />
          {orig && <span className="muted">{mb(orig)} → under {mb(orig * (1 - cut / 100))}</span>}
        </label>
        <div className="actions">
          <button className="secondary" onClick={runCheck} title="Checks that this size can be reached, without starting anything" disabled={!ready}>Feasibility check</button>
          <button className="primary" disabled={!ready} onClick={() => setAsking(true)}>Start run</button>
        </div>
        {!ready && (
          <p className="warn">
            {modelName(model)} isn't prepared yet. Each model is measured once on this computer before its first
            run. <a href="#/models">Prepare it on Select your model →</a>
          </p>
        )}
        {check && <FeasibilityResult job={check} model={model} />}
        <StartDialog open={asking} model={model} cut={cut / 100} originalMb={orig} check={lastCheck}
          onStart={() => post("/api/runs")} onCheck={runCheck} onClose={() => setAsking(false)} />
        {msg && <p className="note">{msg}</p>}
      </div>
      <h2>History</h2>
      {jobs.length === 0 && <p className="muted">Nothing yet.</p>}
      {jobs.map((j) => (
        <details key={j.id} className="job">
          <summary>
            <span className="job-what">{jobTitle(j)}</span>
            <span className={`badge ${j.status}`}>{jobOutcome(j)}</span>
            <span className="muted">{new Date(j.created).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</span>
            {(j.status === "running" || j.status === "queued") && (
              <button className="small" onClick={(e) => { e.preventDefault(); api(`/api/jobs/${j.id}/stop`, { method: "POST" }).then(refresh); }}>
                {j.status === "running" ? "Pause" : "Cancel"}
              </button>
            )}
            {(j.status === "stopped" || j.status === "failed") && j.kind !== "precheck" && (
              <button className="small" onClick={(e) => { e.preventDefault(); api(`/api/jobs/${j.id}/continue`, { method: "POST" }).then(refresh); }}>
                Continue
              </button>
            )}
          </summary>
          <div className="job-details">
            {j.kind === "precheck" && j.result && <FeasibilityResult job={j} model={j.params?.model ?? model} />}
            <details className="raw">
              <summary>System logs</summary>
              <pre>{j.output || "(nothing yet)"}</pre>
            </details>
          </div>
        </details>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- "Your computer" panel

type Machine = {
  machine: { name: string; chip: string; memory_gb: number | null };
  models: { id: string; ready: boolean; built_in: boolean; size_mb: number; working_mb: number }[];
  max_cut: number;
  free_share: number;
};

// other devices are rough examples: memory only, as sold
const DEVICES: { id: string; label: string; memory_gb: number }[] = [
  { id: "mac-mini", label: "Mac mini · 16 GB", memory_gb: 16 },
  { id: "laptop-8", label: "Laptop · 8 GB", memory_gb: 8 },
  { id: "phone", label: "Phone · 8 GB", memory_gb: 8 },
  { id: "pi", label: "Raspberry Pi · 4 GB", memory_gb: 4 },
  { id: "pi2", label: "Small board · 2 GB", memory_gb: 2 },
];

type Live = { memory_used_pct: number | null; gpu_busy_pct: number | null; gpu_memory_mb: number | null };

const GB = 1073.741824; // MB in one GB of memory

function MachinePanel() {
  const [m, setM] = useState<Machine | null>(null);
  const [device, setDevice] = useState("this");
  const [memGb, setMemGb] = useState<number | null>(null);
  const [free, setFree] = useState<number | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [now, setNow] = useState<Live | null>(null);
  useEffect(() => {
    if (device !== "this") return;
    const read = () => api<Live>("/api/machine/live").then(setNow).catch(() => undefined);
    read();
    const t = setInterval(read, 3000);
    return () => clearInterval(t);
  }, [device]);
  useEffect(() => {
    api<Machine>("/api/machine").then((x) => {
      setM(x);
      setFree(Math.round(x.free_share * 100));
      setMemGb(Math.round(x.machine.memory_gb ?? 8));
      setModel(x.models[x.models.length - 1]?.id ?? null);
    }).catch(() => undefined);
  }, []);
  if (!m || free == null || memGb == null) return <div className="machine muted">Loading…</div>;
  const sel = m.models.find((x) => x.id === model) ?? m.models[0];
  if (!sel) return null;
  const pick = (id: string) => {
    setDevice(id);
    if (id === "this") setMemGb(Math.round(m.machine.memory_gb ?? 8));
    else if (id !== "custom") setMemGb(DEVICES.find((d) => d.id === id)!.memory_gb);
  };
  const total = memGb * GB;
  const others = total * (free / 100);
  const working = sel.working_mb;
  const room = Math.max(0, total - others - working);
  const fits = sel.size_mb <= room;
  const cut = 1 - room / sel.size_mb;
  const reachable = room > 0 && cut <= m.max_cut;
  const parts = [
    { cls: "others", label: "Other apps", mb: others },
    { cls: "working", label: "Working memory", mb: working },
    { cls: "room", label: "Model space", mb: room },
  ];
  return (
    <section className="machine">
      <div className="m-grid">
        <label>
          <span className="m-label">Preset device</span>
          <select value={device} onChange={(e) => pick(e.target.value)}>
            <option value="this">This computer: {m.machine.name} · {m.machine.chip.replace("Apple ", "")}</option>
            {DEVICES.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
            <option value="custom">Custom</option>
          </select>
        </label>
        <label>
          <span className="m-label">Model</span>
          <select value={sel.id} onChange={(e) => {
            if (e.target.value === ADD_MODEL) window.location.hash = "#/models";
            else setModel(e.target.value);
          }}>
            {m.models.map((x) => <option key={x.id} value={x.id}>{modelName(x.id)} · {mb(x.size_mb)}</option>)}
            <option value={ADD_MODEL}>Add your own model…</option>
          </select>
        </label>
        <label>
          <span className="m-label">Memory (GB)</span>
          <NumberField min={1} max={512} step={1} value={memGb} disabled={device !== "custom"} onCommit={setMemGb} />
        </label>
        <label>
          <span className="m-label">Reserved for other apps (%)</span>
          <NumberField min={0} max={95} step={1} value={free} onCommit={setFree} />
        </label>
      </div>

      {device === "this" && now && (
        <div className="live">
          <span className="dot-live" /> Current state:
          {now.memory_used_pct != null && <span>memory <b>{now.memory_used_pct}%</b> in use</span>}
          {now.gpu_busy_pct != null && <span>GPU <b>{now.gpu_busy_pct}%</b> busy</span>}
          {now.gpu_memory_mb != null && <span>GPU using <b>{mb(now.gpu_memory_mb)}</b></span>}
        </div>
      )}

      <div>
        <div className="m-label">Memory breakdown ({memGb} GB)</div>
        <div className="stack" role="img" aria-label="How the memory is shared">
          {parts.map((p) => <span key={p.cls} className={`seg ${p.cls}`} style={{ width: `${(p.mb / total) * 100}%` }} title={`${p.label}: ${mb(p.mb)}`} />)}
        </div>
        <div className="legend">
          {parts.map((p) => <span key={p.cls}><i className={`sw ${p.cls}`} /> {p.label}: {mb(p.mb)}</span>)}
        </div>
      </div>

      <div className="m-verdict">
        {fits ? (
          <span className="ok-text">{modelName(sel.id)} fits as it is ({mb(sel.size_mb)}). No cutting needed.</span>
        ) : reachable ? (
          <>
            <span>
              Suggested: cut {modelName(sel.id)} from {mb(sel.size_mb)} to <b>{mb(room)}</b> ({pct(cut)} smaller)
            </span>
            <a className="primary" href={`#/new?model=${encodeURIComponent(sel.id)}&cut=${Math.ceil(cut * 100) / 100}`}>Make it fit →</a>
          </>
        ) : (
          <span className="warn">{modelName(sel.id)} would need to be {room > 0 ? pct(cut) : "100%"} smaller. Too much to cut; pick a smaller model.</span>
        )}
      </div>

    </section>
  );
}

/** A number box that lets people type freely: in-range values apply as typed; anything else is fixed on leaving
 * the box (or Enter), instead of being overwritten on every key. */
function NumberField({ value, min, max, step, disabled, onCommit }: {
  value: number; min: number; max: number; step: number; disabled?: boolean; onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const settle = () => {
    const n = Number(text);
    const v = text.trim() === "" || Number.isNaN(n) ? value : Math.min(max, Math.max(min, n));
    setText(String(v));
    onCommit(v);
  };
  return (
    <input type="number" inputMode="numeric" min={min} max={max} step={step} value={text} disabled={disabled}
      onChange={(e) => {
        setText(e.target.value);
        const n = Number(e.target.value);
        if (e.target.value.trim() !== "" && !Number.isNaN(n) && n >= min && n <= max) onCommit(n);
      }}
      onBlur={settle}
      onKeyDown={(e) => { if (e.key === "Enter") settle(); }} />
  );
}

function FeasibilityResult({ job, model }: { job: Job; model: string }) {
  if (job.status === "queued" || job.status === "running") return <div className="feasible checking">Checking… about half a minute</div>;
  const r = job.result;
  if (!r) return <p className="warn">Check failed. See the system logs.</p>;
  const lo = r.smallest_mb, hi = r.original_mb;
  const sizeOk = lo != null && hi != null && r.limit_mb >= lo && r.limit_mb < hi;
  const calls = r.time ? Math.ceil(r.time.rounds[1]) : r.usage?.calls;
  const t = r.time ? durationRange(r.time.minutes[0], r.time.minutes[1]) : null;
  return (
    <div className={`feasible ${r.feasible ? "yes" : "no"}`}>
      <div className="f-top">
        <span className="f-badge">{r.feasible ? "✓ Feasible" : "✕ Not feasible"}</span>
        <span className="f-title">{modelName(model)} → under {mb(r.limit_mb)}</span>
      </div>
      {!r.feasible && r.stop_reason && <div className="f-reason">{r.stop_reason}</div>}
      <div className="f-tiles">
        {lo != null && hi != null && (
          <div className="f-tile">
            <div className="f-label">Size</div>
            <div className="f-big">{sizeOk ? "Within reach" : "Out of reach"}</div>
            <div className="f-range">
              <span className="f-track"><span className="f-mark" style={{ left: `${Math.min(100, Math.max(0, ((r.limit_mb - lo) / (hi - lo)) * 100))}%` }} /></span>
              <span className="f-ends"><span>{mb(lo)} smallest</span><span>{mb(hi)} now</span></span>
            </div>
          </div>
        )}
        {r.usage && (
          <div className="f-tile">
            <div className="f-label">Usage limits</div>
            {(r.usage.windows ?? []).map((w) => {
              const cost = w.pct_per_call * (calls ?? r.usage!.calls);
              return (
                <div key={w.name} className="f-limit">
                  <span>{w.name === "weekly" ? "Weekly" : w.name} limit</span>
                  <span className="f-limit-bar"><span className="add" style={{ width: `${Math.min(100, cost)}%` }} /></span>
                  <b>~{cost < 1 ? cost.toFixed(1) : Math.ceil(cost)}%</b>
                </div>
              );
            })}
            {!!r.usage.windows?.length && <div className="f-sub">How much of each limit this run uses</div>}
            {!r.usage.windows?.length && <div className="f-sub">Not recorded for this older check.</div>}
            {r.usage.fits === false && <div className="f-sub warn">Over the limit ✕</div>}
          </div>
        )}
        {t && r.time && (
          <div className="f-tile">
            <div className="f-label">Time</div>
            <div className="f-big">{durationRange(r.time.minutes[0] + FINAL_CHECK_MIN, r.time.minutes[1] + FINAL_CHECK_MIN)}</div>
            <div className="f-sub">Pruning {t} + about {FINAL_CHECK_MIN} min final check</div>
            {r.time.worst_minutes != null && (
              <div className="f-sub">Up to {duration(r.time.worst_minutes + FINAL_CHECK_MIN)} if it uses all {r.time.max_rounds} rounds</div>
            )}
          </div>
        )}
      </div>
      {r.time && <div className="f-foot">Estimates from {r.time.based_on_runs} earlier runs.</div>}
      {r.warnings.map((w) => <div key={w} className="f-reason">{w}</div>)}
    </div>
  );
}

const JOB_KIND: Record<string, string> = {
  precheck: "Feasibility check",
  start: "Run",
  resume: "Run resume",
  heldout: "Unseen-text test",
  accuracy: "Quiz check",
  prepare: "Model preparation",
};

function jobTitle(j: Job): string {
  const what = JOB_KIND[j.kind] ?? j.kind;
  const p = j.params ?? {};
  if (p.model && p.cut != null) return `${what}: ${modelName(p.model)}, ${pct(p.cut)} smaller`;
  if (p.model) return `${what}: ${modelName(p.model)}`;
  if (j.run_id) return `${what}: run ${j.run_id}`;
  return what;
}

function jobOutcome(j: Job): string {
  if (j.kind === "precheck" && j.result) return j.result.feasible ? "Feasible ✓" : "Not feasible";
  if (j.kind === "prepare" && j.status === "running" && j.progress) return `Running · ${Math.round(j.progress.share * 100)}%`;
  return { queued: "Waiting", running: "Running", done: "Finished", failed: "Didn't finish", stopped: "Paused",
           cancelled: "Cancelled", lost: "Interrupted" }[j.status] ?? j.status;
}
