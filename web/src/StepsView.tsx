import { useEffect, useRef } from "react";
import type { Layer, RunDetail } from "./lib";

// What each step of a run removed, exactly (GET /api/runs/{id}/removal). In every layer the pruner keeps the
// highest-scoring neurons and attention units, so each part has a fixed place in its layer's removal order
// (its rank): rank r is gone once the layer keeps r or fewer. With each step's plan that gives every removed part.

export type Removal = {
  mlp_rank: number[][]; // [layer][neuron] -> place in the removal order (0 = kept longest)
  attn_rank: number[][]; // [layer][attention unit] -> same
  weights: { per_neuron: number; per_group: number } | null;
};

type State = "kept" | "now" | "earlier";
const state = (rank: number, keepNow: number, keepBefore: number): State =>
  rank < keepNow ? "kept" : rank < keepBefore ? "now" : "earlier";
const kept = (l: Layer, full: number, of: "mlp" | "groups") => (l.dropped ? 0 : Math.min(l[of], full));

/** Share of each layer's weights a step removed (0-1), from the plans before and after it. */
export function removedShare(prev: Layer[], cur: Layer[], full: { mlp: number; groups: number }, w: Removal["weights"]) {
  const pn = w?.per_neuron ?? 1;
  const pg = w?.per_group ?? full.mlp / Math.max(1, full.groups); // without sizes: attention as a share of the layer
  const total = full.mlp * pn + full.groups * pg;
  return cur.map((c, i) => {
    const p = prev[i];
    const m = kept(p, full.mlp, "mlp") - kept(c, full.mlp, "mlp");
    const g = kept(p, full.groups, "groups") - kept(c, full.groups, "groups");
    return Math.max(0, (m * pn + g * pg) / total);
  });
}

function colors(el: Element) {
  const css = getComputedStyle(el);
  const v = (n: string, d: string) => css.getPropertyValue(n).trim() || d;
  return { kept: v("--part", "#aab4c0"), now: v("--agent", "#d95926"), earlier: v("--line", "#34332f"), ink: v("--ink-3", "#888") };
}

/** A CSS colour as [r, g, b], read back through the canvas (it normalises any colour to #rrggbb). */
function toRgb(ctx: CanvasRenderingContext2D, color: string): [number, number, number] {
  ctx.fillStyle = "#000";
  ctx.fillStyle = color;
  const h = String(ctx.fillStyle);
  if (h.startsWith("#") && h.length === 7) return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
  const m = h.match(/\d+(\.\d+)?/g)?.map(Number) ?? [128, 128, 128];
  return [m[0], m[1], m[2]];
}

function mix(parts: [[number, number, number], number][]): string {
  const n = parts.reduce((a, [, w]) => a + w, 0) || 1;
  const c = [0, 1, 2].map((i) => Math.round(parts.reduce((a, [col, w]) => a + col[i] * w, 0) / n));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const LEFT = 92, COL = 26, GAP = 7, SQ = 5.5, NEURON_H = 240;

/** Every neuron of every layer at its real position, for one step: kept, removed in this step, removed earlier. */
export function NeuronMap({ removal, prev, cur, full, caption }: {
  removal: Removal;
  prev: Layer[]; // the plan before the step (the original model for FLAP's cut)
  cur: Layer[];
  full: { mlp: number; groups: number };
  caption: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const n = cur.length;
  const width = LEFT + n * (COL + GAP) - GAP + 8;
  const attnTop = 6, neuronTop = attnTop + 2 * (SQ + 1.5) + 10;
  const height = neuronTop + NEURON_H + 22;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = width * dpr;
    cv.height = height * dpr;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const c = colors(cv);
    const fill = { kept: c.kept, now: c.now, earlier: c.earlier };
    const rgb = { kept: toRgb(ctx, c.kept), now: toRgb(ctx, c.now), earlier: toRgb(ctx, c.earlier) };
    ctx.font = "9.5px system-ui, sans-serif";
    ctx.fillStyle = c.ink;
    ctx.textAlign = "right";
    ctx.fillText("Attention units", LEFT - 10, attnTop + SQ + 3);
    ctx.fillText("Knowledge part:", LEFT - 10, neuronTop + 10);
    ctx.fillText("every neuron,", LEFT - 10, neuronTop + 23);
    ctx.fillText("in its real place", LEFT - 10, neuronTop + 36);
    ctx.fillText("Layer →", LEFT - 10, height - 6);
    ctx.textAlign = "center";
    for (let i = 0; i < n; i++) {
      const x = LEFT + 4 + i * (COL + GAP);
      const p = prev[i], q = cur[i];
      // attention units
      const gBefore = kept(p, full.groups, "groups"), gNow = kept(q, full.groups, "groups");
      removal.attn_rank[i].forEach((r, g) => {
        ctx.fillStyle = fill[state(r, gNow, gBefore)];
        ctx.fillRect(x + 1 + (g % 4) * (SQ + 1.5), attnTop + Math.floor(g / 4) * (SQ + 1.5), SQ, SQ);
      });
      // neurons: one pixel row covers several neighbouring neurons; its colour mixes theirs
      const ranks = removal.mlp_rank[i];
      const mBefore = kept(p, full.mlp, "mlp"), mNow = kept(q, full.mlp, "mlp");
      const per = ranks.length / NEURON_H;
      for (let y = 0; y < NEURON_H; y++) {
        let now = 0, earlier = 0;
        const a = Math.floor(y * per), b = Math.max(a + 1, Math.floor((y + 1) * per));
        for (let j = a; j < b; j++) {
          const s = state(ranks[j], mNow, mBefore);
          if (s === "now") now++;
          else if (s === "earlier") earlier++;
        }
        const k = b - a;
        // the row's colour is its neurons' colours mixed by count, so a 10% cut looks like 10%
        ctx.fillStyle = mix([[rgb.kept, k - now - earlier], [rgb.now, now], [rgb.earlier, earlier]]);
        ctx.fillRect(x, neuronTop + y, COL, 1);
      }
      ctx.fillStyle = c.ink;
      ctx.fillText(String(i), x + COL / 2, height - 6);
    }
  }, [removal, prev, cur, full, n, width, height]);

  return (
    <figure className="map">
      <canvas ref={ref} style={{ width: "100%", aspectRatio: `${width} / ${height}` }}
        role="img" aria-label={caption} />
      <figcaption>{caption}</figcaption>
      <div className="legend">
        <span><i className="sw part" /> Kept</span>
        <span><i className="sw hot" /> Removed in this step</span>
        <span><i className="sw earlier" /> Removed earlier</span>
        <span className="muted">Each column is one layer.</span>
      </div>
    </figure>
  );
}

/** Every kept step at once: a row per step, a column per layer, darker where the step removed more of that layer. */
export function StepGrid({ run, removal, at, onPick }: {
  run: RunDetail;
  removal: Removal | null;
  at: number;
  onPick: (round: number) => void;
}) {
  const full = { mlp: run.shape.mlp_full, groups: run.shape.groups_full };
  const steps = run.steps;
  const rows = steps.slice(1).map((s, k) => ({ round: s.round, share: removedShare(steps[k].layers, s.layers, full, removal?.weights ?? null) }));
  if (!rows.length) return null;
  const total = removedShare(steps[0].layers, steps[steps.length - 1].layers, full, removal?.weights ?? null);
  const max = Math.max(...rows.flatMap((r) => r.share), 0.0001);
  const n = run.shape.layers;
  const CW = 22, CH = 15, G = 2, L = 78, T = 22;
  const W = L + n * (CW + G) + 4;
  const H = T + (rows.length + 1) * (CH + G) + 10;
  const current = [...rows].reverse().find((r) => r.round <= at)?.round;
  const cellX = (i: number) => L + i * (CW + G);
  const rowY = (k: number) => T + k * (CH + G);
  const alpha = (v: number) => (v <= 0 ? 0 : 0.18 + 0.82 * Math.sqrt(v / max));
  return (
    <figure className="chart step-grid">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="How much of each layer every step removed">
        {Array.from({ length: n }, (_, i) => (
          <text key={i} x={cellX(i) + CW / 2} y={T - 7} textAnchor="middle" className="tick">{i}</text>
        ))}
        <text x={L - 8} y={T - 7} textAnchor="end" className="tick">Layer →</text>
        {rows.map((r, k) => (
          <g key={r.round} className={`grid-row ${r.round === current ? "on" : ""}`} onClick={() => onPick(r.round)}>
            <rect x={0} y={rowY(k) - 1} width={W} height={CH + 2} className="row-hit" />
            <text x={L - 8} y={rowY(k) + CH - 4} textAnchor="end" className="tick">Round {r.round}</text>
            {r.share.map((v, i) => (
              <rect key={i} x={cellX(i)} y={rowY(k)} width={CW} height={CH} rx={2}
                className={v > 0 ? "cell hit" : "cell"} style={v > 0 ? { fillOpacity: alpha(v) } : undefined}>
                <title>{`Round ${r.round}, layer ${i}: ${v > 0 ? `${(v * 100).toFixed(1)}% of the layer removed` : "nothing removed"}`}</title>
              </rect>
            ))}
          </g>
        ))}
        <g>
          <text x={L - 8} y={rowY(rows.length) + CH - 4} textAnchor="end" className="tick total">In total</text>
          {total.map((v, i) => (
            <rect key={i} x={cellX(i)} y={rowY(rows.length) + 4} width={CW} height={CH} rx={2}
              className={v > 0 ? "cell total hit" : "cell"} style={v > 0 ? { fillOpacity: 0.18 + 0.82 * Math.min(1, v) } : undefined}>
              <title>{`Layer ${i}: ${(v * 100).toFixed(1)}% removed by the end`}</title>
            </rect>
          ))}
        </g>
      </svg>
      <div className="legend">
        <span><i className="sw hot" /> Darker: more of that layer removed in that step</span>
        <span className="muted">Click a round to show it above.</span>
      </div>
    </figure>
  );
}
