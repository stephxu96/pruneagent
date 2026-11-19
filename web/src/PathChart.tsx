import { useState } from "react";
import { changesText, lossText, mb, qualityLoss, type RunDetail } from "./lib";

// Every idea the AI tested, placed by how much memory it would save (across) and how much quality it would
// cost (up). The kept ideas form the path toward the goal line. FLAP's single cut at the same size is marked.

const W = 640;
const H = 300;
const M = { l: 78, r: 20, t: 16, b: 44 };

type Pt = { x: number; y: number; label: string; kept: boolean; round: number };

export function PathChart({ run, upTo }: { run: RunDetail; upTo: number }) {
  const [hover, setHover] = useState<Pt | null>(null);
  const orig = run.original_mb ?? 0;
  const base = run.original_error ?? 1;
  const saved = (size: number) => orig - size;
  const all: Pt[] = [];
  for (const r of run.rounds) {
    for (const c of r.candidates) {
      if (c.error == null || c.size_mb == null) continue;
      all.push({
        x: saved(c.size_mb),
        y: qualityLoss(c.error, base),
        label: `Round ${r.round}: ${changesText(c.changes)}. Would save ${mb(saved(c.size_mb))} in all, quality ${lossText(qualityLoss(c.error, base))}.`,
        kept: r.chosen === c.id,
        round: r.round,
      });
    }
  }
  const pts = all.filter((p) => p.round <= upTo);
  const fullPath = run.steps.map((s) => ({ x: saved(s.size_mb), y: qualityLoss(s.error, base), round: s.round }));
  const path = fullPath.filter((p) => p.round <= upTo);
  const goal = orig - run.limit_mb;
  const flap = run.flap?.error ? { x: saved(run.flap.size_mb), y: qualityLoss(run.flap.error, base) } : null;
  // fixed scale for the whole run, so the chart does not jump while playing; FLAP is shown only if it fits
  const yTop = Math.max(...all.map((p) => p.y), ...fullPath.map((p) => p.y), 0.1) * 1.1;
  const xMax = goal * 1.08;
  const sx = (x: number) => M.l + (Math.max(0, x) / xMax) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - (Math.min(Math.max(y, 0), yTop) / yTop) * (H - M.t - M.b);
  const yTicks = niceTicks(yTop);
  const xTicks = niceTicks(xMax);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Ideas tested: memory saved against quality lost">
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={M.l - 8} y={sy(t) + 4} textAnchor="end" className="tick">{`+${Math.round(t * 100)}%`}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={H - M.b + 18} textAnchor="middle" className="tick">{`${Math.round(t)} MB`}</text>
        ))}
        <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} className="axis" />
        <line x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} className="axis" />
        <line x1={sx(goal)} x2={sx(goal)} y1={M.t} y2={H - M.b} className="goal" />
        <text x={sx(goal) - 6} y={M.t + 12} textAnchor="end" className="goal-label">Goal</text>
        <text x={(M.l + W - M.r) / 2} y={H - 6} textAnchor="middle" className="axis-title">Memory saved →</text>
        <text transform={`translate(14 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis-title">
          Quality lost ↑
        </text>
        {pts.filter((p) => !p.kept).map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={4} className="idea"
            onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)} />
        ))}
        <polyline points={path.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} className="path" />
        {path.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={5} className="kept" />
        ))}
        {pts.filter((p) => p.kept).map((p, i) => (
          <circle key={`h${i}`} cx={sx(p.x)} cy={sy(p.y)} r={10} fill="transparent"
            onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)} />
        ))}
        {flap && flap.y <= yTop * 1.0001 && (
          <g>
            <rect x={sx(flap.x) - 6} y={sy(flap.y) - 6} width={12} height={12} className="flap" transform={`rotate(45 ${sx(flap.x)} ${sy(flap.y)})`} />
            <text x={sx(flap.x) - 12} y={sy(flap.y) + 4} textAnchor="end" className="flap-label">FLAP</text>
          </g>
        )}
      </svg>
      {flap && flap.y > yTop && (
        <p className="note">The standard method (FLAP) at this size is off the top of the chart: {lossText(flap.y)}.</p>
      )}
      <div className="legend">
        <span><i className="sw kept" /> Kept ideas</span>
        <span><i className="sw idea" /> Rejected ideas</span>
        <span><i className="sw flap" /> FLAP, same size</span>
      </div>
      <p className="tooltip" aria-live="polite">{hover ? hover.label : "Hover a dot for details."}</p>
    </figure>
  );
}

function niceTicks(max: number): number[] {
  const raw = max / 4;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let t = 0; t <= max + 1e-9; t += step) out.push(t);
  return out;
}
