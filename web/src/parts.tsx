import { useEffect, useState } from "react";
import { duration, durationRange, checksMinutes, mb, modelName, pct } from "./lib";

// Shared parts (docs/APP_DESIGN.md §5.1): the types the pages read, and the pieces every page is built from.

export type Feasibility = {
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
  time_by_search?: Record<string, { minutes: [number, number]; rounds: [number, number]; based_on_runs: number; worst_minutes?: number; max_rounds?: number } | null>;
};
export type Job = {
  id: number; kind: string; reason?: string | null; params: { model?: string; cut?: number; run_id?: string }; run_id: string | null; status: string;
  created: string; output: string; result: Feasibility | null;
  progress?: { step: number; of: number; label: string; done: number; total: number; share: number };
  steps?: { name: string; state: string }[]; step?: number; release?: string | null; started?: string | null;
};


export type Machine = {
  machine: { name: string; chip: string; memory_gb: number | null };
  models: { id: string; ready: boolean; built_in: boolean; size_mb: number; working_mb: number; layers?: number }[];
  max_cut: number;
  free_share: number;
};

// other devices are rough examples: memory only, as sold
export const DEVICES: { id: string; label: string; memory_gb: number }[] = [
  { id: "mac-mini", label: "Mac mini · 16 GB", memory_gb: 16 },
  { id: "laptop-8", label: "Laptop · 8 GB", memory_gb: 8 },
  { id: "phone", label: "Phone · 8 GB", memory_gb: 8 },
  { id: "pi", label: "Raspberry Pi · 4 GB", memory_gb: 4 },
  { id: "pi2", label: "Small board · 2 GB", memory_gb: 2 },
];

export type Live = { memory_used_pct: number | null; gpu_busy_pct: number | null; gpu_memory_mb: number | null };

export const GB = 1073.741824; // MB in one GB of memory


/** The step bar of a guided flow: done steps get ✓, the current one is highlighted. */
export function StepBar({ steps, at }: { steps: string[]; at: number }) {
  return (
    <ol className="stepper">
      {steps.map((s, i) => (
        <li key={s} className={i + 1 < at ? "done" : i + 1 === at ? "current" : ""}>
          <span className="dot">{i + 1 < at ? "✓" : i + 1}</span>
          <span>{s}</span>
        </li>
      ))}
    </ol>
  );
}

export /** A number box that lets people type freely: in-range values apply as typed; anything else is fixed on leaving
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

export function FeasibilityResult({ job, model }: { job: Job; model: string }) {
  if (job.status === "queued" || job.status === "running") return <div className="feasible checking">Checking… about half a minute</div>;
  const r = job.result;
  if (!r) return <p className="warn">Check failed. See the system logs.</p>;
  const lo = r.smallest_mb, hi = r.original_mb;
  const ck = checksMinutes(r.original_mb);
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
            <div className="f-big">{durationRange(r.time.minutes[0] + ck[0], r.time.minutes[1] + ck[1])}</div>
            <div className="f-sub">Pruning {t} + checks {durationRange(ck[0], ck[1])}</div>
            {r.time.worst_minutes != null && (
              <div className="f-sub">Up to {duration(r.time.worst_minutes + ck[1])} if it uses all {r.time.max_rounds} rounds</div>
            )}
          </div>
        )}
      </div>
      {r.time && <div className="f-foot">Estimates from {r.time.based_on_runs} earlier runs.</div>}
      {r.warnings.map((w) => <div key={w} className="f-reason">{w}</div>)}
    </div>
  );
}

export const JOB_KIND: Record<string, string> = {
  precheck: "Feasibility check",
  start: "Run",
  resume: "Run resume",
  heldout: "Perplexity check",
  accuracy: "Accuracy check",
  prepare: "Model setup",
};

export function jobTitle(j: Job): string {
  const what = JOB_KIND[j.kind] ?? j.kind;
  const p = j.params ?? {};
  if (p.model && p.cut != null) return `${what}: ${modelName(p.model)}, ${pct(p.cut)} smaller`;
  if (p.model) return `${what}: ${modelName(p.model)}`;
  if (j.run_id) return `${what}: run ${j.run_id}`;
  return what;
}

export function jobOutcome(j: Job): string {
  if (j.kind === "precheck" && j.result) return j.result.feasible ? "Feasible ✓" : "Not feasible";
  if (j.kind === "prepare" && j.status === "running" && j.progress) return `Running · ${Math.round(j.progress.share * 100)}%`;
  return { queued: "Waiting", running: "Running", done: "Finished", failed: "Didn't finish", stopped: "Paused",
           cancelled: "Cancelled", lost: "Interrupted" }[j.status] ?? j.status;
}
