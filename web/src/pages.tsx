import { useEffect, useState } from "react";
import { FeasibilityResult, jobOutcome, jobTitle, type Job } from "./parts";
import { ResultBars } from "./RunPage";
import { api, lossText, mb, modelName, pct, qualityLoss, runLabel, stateText, type Compare, type RunSummary } from "./lib";

// ---------------------------------------------------------------- home: every run

export function HomePage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    const read = () => {
      api<RunSummary[]>("/api/runs").then((r) => setRuns([...r].sort((a, b) => b.created.localeCompare(a.created))))
        .catch((e) => setErr(String(e)));
      api<Job[]>("/api/jobs").then(setJobs).catch(() => undefined);
    };
    read();
    const t = setInterval(read, 5000);
    return () => clearInterval(t);
  }, []);
  // pruning workflows that are under way, newest first; their run may not exist yet (waiting for the Mac)
  const active = jobs.filter((j) => (j.kind === "start" || j.kind === "resume") && ["queued", "running", "stopped"].includes(j.status));
  const activeIds = new Set(active.map((j) => j.run_id));
  const finished = (runs ?? []).filter((r) => !activeIds.has(r.id));
  return (
    <div>
      <section className="hero">
        <h1>Shrink an AI model to fit your computer.</h1>
        <a className="primary" href="#/new">Prune a model →</a>
      </section>
      {active.length > 0 && (
        <>
          <h2>Under way</h2>
          <div className="cards">
            {active.map((j) => {
              const r = runs?.find((x) => x.id === j.run_id);
              return <ActiveCard key={j.id} job={j} run={r} />;
            })}
          </div>
        </>
      )}
      <h2>Runs</h2>
      {err && <p className="error">{err}</p>}
      {runs && finished.length === 0 && <p className="muted">No runs yet.</p>}
      <div className="cards">
        {finished.map((r) => (
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
            <span className="go">Open →</span>
          </a>
        ))}
      </div>
    </div>
  );
}

const STEP_NAMES: Record<string, string> = { run: "Pruning", heldout: "Evaluating: unseen text", accuracy: "Evaluating: question sets" };

function ActiveCard({ job, run }: { job: Job; run?: RunSummary }) {
  const toCut = run?.original_mb && run.now_mb != null ? run.original_mb - run.limit_mb : null;
  const share = run && toCut ? Math.min(1, (run.original_mb! - run.now_mb!) / toCut) : 0;
  const step = job.steps?.[job.step ?? 0]?.name ?? "run";
  const status = job.status === "queued" ? "Waiting for the Mac to be free"
    : job.status === "stopped" ? `Paused${job.reason ? `: ${job.reason}` : ""}` : STEP_NAMES[step] ?? "Running";
  return (
    <a className="card active" href={job.run_id ? `#/run/${job.run_id}` : "#/activity"}>
      <div className="card-title">{modelName(job.params.model ?? run?.model ?? "")}{job.params.cut != null ? `, ${pct(job.params.cut)} smaller` : run?.cut != null ? `, ${pct(run.cut)} smaller` : ""}</div>
      <div className={`badge ${job.status === "stopped" ? "paused" : job.status === "queued" ? "" : "running"}`}>{status}</div>
      <div className="bar"><div className="fill" style={{ width: `${Math.round((step === "run" ? share : 1) * 100)}%` }} /></div>
      <div className="muted">
        {run?.now_mb != null ? `${mb(run.now_mb)} now · target under ${mb(run.limit_mb)}` : "Not started yet"}
      </div>
      <span className="go">Watch →</span>
    </a>
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

// ---------------------------------------------------------------- activity: every workflow, for fixing problems

export function ActivityPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const refresh = () => api<Job[]>("/api/jobs").then(setJobs).catch(() => undefined);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);
  return (
    <div>
      <a href="#/" className="back">← All runs</a>
      <h1>Activity</h1>
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
            {j.reason && <p className="muted">{j.reason}</p>}
            {j.kind === "precheck" && j.result && <FeasibilityResult job={j} model={j.params?.model ?? ""} />}
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
