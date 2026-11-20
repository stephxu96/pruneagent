import type { Layer } from "./lib";

// The model drawn as its layers, side by side. Each layer: 8 attention units (squares) on top, and its
// "knowledge" part below as a bar whose height is how much of it is left. Removed parts fade to outlines;
// what changed in the current step is highlighted.

const COL = 26;
const GAP = 7;
const SQ = 5.5;
const BAR_H = 120;
const TOP = 18;
const LEFT = 92; // row labels: what each row of a column is

export function ModelMap({
  layers,
  full,
  changed = new Set<number>(),
  highlight = "var(--agent)",
  caption,
}: {
  layers: Layer[];
  full: { mlp: number; groups: number };
  changed?: Set<number>;
  highlight?: string;
  caption?: string;
}) {
  const width = LEFT + layers.length * (COL + GAP) - GAP + 8;
  const sqTop = TOP;
  const barTop = sqTop + 2 * (SQ + 1.5) + 8;
  const height = barTop + BAR_H + 22;
  return (
    <figure className="map">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={caption ?? "The model's layers"}>
        <text x={LEFT - 10} y={sqTop + SQ + 3} className="row-label" textAnchor="end">Attention units</text>
        <text x={LEFT - 10} y={barTop + BAR_H / 2 - 4} className="row-label" textAnchor="end">Knowledge</text>
        <text x={LEFT - 10} y={barTop + BAR_H / 2 + 9} className="row-label" textAnchor="end">part</text>
        <text x={LEFT - 10} y={height - 6} className="row-label" textAnchor="end">Layer →</text>
        {layers.map((l, i) => {
          const x = LEFT + 4 + i * (COL + GAP);
          const hot = changed.has(i);
          const share = l.dropped ? 0 : l.mlp / full.mlp;
          const fillH = BAR_H * share;
          return (
            <g key={i} className={l.dropped ? "layer dropped" : "layer"}>
              <title>
                {`Layer ${i}: ` +
                  (l.dropped
                    ? "removed entirely"
                    : `${l.groups} of ${full.groups} attention units, ${Math.round(share * 100)}% of its knowledge part left`)}
              </title>
              {hot && (
                <rect x={x - 3} y={sqTop - 5} width={COL + 6} height={height - sqTop - 14} rx={5}
                  fill="none" stroke={highlight} strokeWidth={2} className="pulse" />
              )}
              {Array.from({ length: full.groups }, (_, g) => {
                const kept = !l.dropped && g < l.groups;
                const cx = x + 1 + (g % 4) * (SQ + 1.5);
                const cy = sqTop + Math.floor(g / 4) * (SQ + 1.5);
                return (
                  <rect key={g} x={cx} y={cy} width={SQ} height={SQ} rx={1}
                    className={kept ? "part" : "gone"} />
                );
              })}
              <rect x={x} y={barTop} width={COL} height={BAR_H} rx={4} className="slot" />
              <rect x={x} y={barTop + BAR_H - fillH} width={COL} height={fillH} rx={4}
                className={hot ? "part hot" : "part"} style={hot ? { fill: highlight } : undefined} />
              {l.dropped && (
                <text x={x + COL / 2} y={barTop + BAR_H / 2} className="cross" textAnchor="middle">
                  ✕
                </text>
              )}
              <text x={x + COL / 2} y={height - 6} className="num" textAnchor="middle">
                {i}
              </text>
            </g>
          );
        })}
      </svg>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

export function MapLegend() {
  return (
    <div className="legend">
      <span><i className="sw part" /> Remaining parts</span>
      <span><i className="sw gone" /> Removed parts</span>
      <span><i className="sw hot" /> Latest change</span>
    </div>
  );
}
