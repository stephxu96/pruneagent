// Types for the API (src/pruneagent/web/views.py) and the plain-language wording used on every screen.

export type Layer = { mlp: number; groups: number; dropped: boolean };
export type Change = { layer: number; strategy: "narrow_mlp" | "remove_attention_group" | "drop_layer" | string };

export type RunSummary = {
  id: string;
  model: string;
  state: string;
  reason: string | null;
  round: number;
  kept_steps: number;
  created: string;
  updated: string;
  limit_mb: number;
  original_mb: number | null;
  cut: number | null;
  now_mb: number | null;
  original_error: number | null;
  now_error: number | null;
  has_heldout: boolean;
  has_accuracy: boolean;
};

export type Candidate = {
  id: number;
  changes: Change[];
  why: string;
  error: number | null;
  rise: number | null;
  freed_mb: number;
  size_mb: number | null;
  refused: string[] | null;
};

export type Round = {
  round: number;
  outcome: "ACCEPT" | "ROLLBACK" | "INVALID" | null;
  why: string;
  chosen: number | null;
  candidates: Candidate[];
  refused_reasons: string[];
  agent_seconds: number;
  check_seconds: number;
};

export type Step = { round: number; error: number; size_mb: number; layers: Layer[] };

export type RunDetail = RunSummary & {
  shape: { layers: number; mlp_full: number; groups_full: number };
  flap: { size_mb: number; error: number | null; layers: Layer[] } | null;
  steps: Step[];
  rounds: Round[];
};

export type Score = { size_mb: number; test_error: number; unseen_error: number };
export type Accuracy = { all_questions: { acc: number; stderr: number; questions: number }; sets: Record<string, { acc: number; acc_stderr: number | null }> };
export type Results = { heldout: Record<string, Score> | null; accuracy: Record<string, Accuracy> | null };
export type Compare = {
  runs: { id: string; model: string; cut: number; agent: Score; flap: Score; original: Score }[];
  flap_curve: { cut: number; test_error: number }[];
};

// ---------------------------------------------------------------- wording

export const modelName = (id: string) => id.replace(/^Qwen\//, "").replace(/-Base$/, "");
export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
export const mb = (x: number) => `${Math.round(x).toLocaleString()} MB`;

/** How much worse than the original model, in plain words: "+22% worse". */
export const qualityLoss = (error: number, original: number) => error / original - 1;
export const lossText = (loss: number) => (loss < 0.005 ? "no loss" : loss >= 2 ? `${(loss + 1).toFixed(0)}× worse` : `${pct(loss)} worse`);

export function changeText(c: Change): string {
  const n = c.layer + 1; // people count from 1
  switch (c.strategy) {
    case "narrow_mlp":
      return `slim layer ${n}`;
    case "remove_attention_group":
      return `remove an attention unit from layer ${n}`;
    case "drop_layer":
      return `remove layer ${n} entirely`;
    default:
      return `${c.strategy} on layer ${n}`;
  }
}

/** Group "slim layer 3, slim layer 4, slim layer 5" into "slim layers 3, 4, 5". */
export function changesText(changes: Change[]): string {
  const byKind = new Map<string, number[]>();
  for (const c of changes) byKind.set(c.strategy, [...(byKind.get(c.strategy) ?? []), c.layer + 1]);
  const parts: string[] = [];
  for (const [kind, layers] of byKind) {
    const ls = [...new Set(layers)].sort((a, b) => a - b);
    const times = layers.length > ls.length ? " (twice)" : "";
    if (ls.length === 1) parts.push(changeText({ layer: ls[0] - 1, strategy: kind }) + times);
    else if (kind === "narrow_mlp") parts.push(`slim layers ${ls.join(", ")}${times}`);
    else if (kind === "remove_attention_group") parts.push(`remove attention units from layers ${ls.join(", ")}`);
    else if (kind === "drop_layer") parts.push(`remove layers ${ls.join(", ")} entirely`);
    else parts.push(`${kind} on layers ${ls.join(", ")}`);
  }
  return parts.join("; ");
}

export const outcomeText: Record<string, string> = {
  ACCEPT: "Kept",
  ROLLBACK: "Nothing kept: every idea hurt quality too much",
  INVALID: "Ideas refused: repeated something already tried",
};

/** Which layers changed between two kept steps (for highlighting). */
export function changedLayers(before: Layer[] | undefined, after: Layer[]): Set<number> {
  const out = new Set<number>();
  if (!before) return out;
  after.forEach((l, i) => {
    const b = before[i];
    if (b && (b.mlp !== l.mlp || b.groups !== l.groups || b.dropped !== l.dropped)) out.add(i);
  });
  return out;
}

export const runLabel = (r: { model: string; cut: number | null }) =>
  `${modelName(r.model)}, ${r.cut == null ? "?" : pct(r.cut)} smaller`;

export const stateText: Record<string, string> = {
  running: "Running now",
  paused: "Paused",
  interrupted: "Stopped unexpectedly (can continue)",
  finished: "Finished",
};

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      msg = (await r.json()).detail ?? msg;
    } catch {
      /* keep the status */
    }
    throw new Error(msg);
  }
  return r.json();
}
