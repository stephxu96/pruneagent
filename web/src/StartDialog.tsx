import { useEffect, useRef } from "react";
import { checksMinutes, duration, durationRange, mb, modelName, SEARCHES } from "./lib";

// The confirmation before a real run: what it will do, how long, how much of the usage limits, and what to keep in
// mind while it runs. Built on <dialog>, so Escape, focus and the backdrop behave as the browser's own.

export type CheckResult = {
  feasible: boolean;
  stop_reason: string | null;
  limit_mb: number;
  usage: { windows?: { name: string; pct_per_call: number }[]; calls: number } | null;
  time?: { minutes: [number, number]; rounds: [number, number]; worst_minutes?: number; max_rounds?: number } | null;
};

export function StartDialog({ open, model, cut, originalMb, check, search, onSearch, onStart, onCheck, onClose }: {
  open: boolean;
  model: string;
  cut: number; // 0.10 = 10% smaller
  originalMb: number | undefined;
  check: CheckResult | null; // the latest feasibility check for this model and size, if any
  search: string;
  onSearch: (id: string) => void;
  onStart: () => void;
  onCheck: () => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const calls = check?.time ? Math.ceil(check.time.rounds[1]) : check?.usage?.calls;
  const blocked = check != null && !check.feasible;
  const ck = checksMinutes(originalMb);
  return (
    <dialog ref={ref} className="start-dialog" onClose={onClose}
      onClick={(e) => { if (e.target === ref.current) onClose(); }}>
      <div className="sd-body">
        <h2>Start this run?</h2>
        <dl className="sd-facts">
          <dt>Model</dt>
          <dd>{modelName(model)}</dd>
          <dt>Strategy</dt>
          <dd>
            <div className="sd-choice" role="radiogroup" aria-label="Strategy">
              {SEARCHES.map((s) => (
                <label key={s.id} className={search === s.id ? "on" : ""}>
                  <input type="radio" name="search" checked={search === s.id} onChange={() => onSearch(s.id)} />
                  <span title={s.text}>{s.label}</span>
                </label>
              ))}
            </div>
          </dd>
          <dt>Target size</dt>
          <dd>
            {originalMb ? <>{mb(originalMb)} → under {mb(originalMb * (1 - cut))}</> : <>under {check ? mb(check.limit_mb) : "?"}</>}
            <span className="muted"> · {Math.round(cut * 100)}% smaller</span>
          </dd>
          <dt>Time</dt>
          <dd>
            {check?.time ? (
              <>
                {durationRange(check.time.minutes[0] + ck[0], check.time.minutes[1] + ck[1])}
                {check.time.worst_minutes != null && check.time.worst_minutes > check.time.minutes[1] + 1 && (
                  <span className="muted"> · up to {duration(check.time.worst_minutes + ck[1])}</span>
                )}
              </>
            ) : <span className="muted">Known after the feasibility check</span>}
          </dd>
          <dt>Usage limits</dt>
          <dd>
            {check?.usage?.windows?.length && calls ? (
              check.usage.windows.map((w, i) => {
                const pct = w.pct_per_call * calls;
                return (
                  <span key={w.name}>
                    {i > 0 && " · "}
                    {w.name === "weekly" ? "weekly" : w.name} ~{pct < 1 ? pct.toFixed(1) : Math.ceil(pct)}%
                  </span>
                );
              })
            ) : <span className="muted">Known after the feasibility check</span>}
          </dd>
        </dl>

        {blocked && (
          <div className="sd-note bad">
            <b>Not feasible at this size.</b> {check?.stop_reason}
          </div>
        )}
        {!blocked && (
          <ul className="sd-keep">
            <li>Lid open until it finishes. Closing it stops the run; it can be continued later.</li>
            <li>Other heavy apps slow the run down.</li>
          </ul>
        )}

        <div className="sd-actions">
          <button className="secondary" onClick={onClose}>Cancel</button>
          {!check && <button className="secondary" onClick={() => { onClose(); onCheck(); }}>Feasibility check first</button>}
          <button className="primary" disabled={blocked} onClick={() => { onClose(); onStart(); }}>
            Start run
          </button>
        </div>
      </div>
    </dialog>
  );
}
