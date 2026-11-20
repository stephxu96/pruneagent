// Types for the API (src/pruneagent/web/views.py) and the plain-language wording used on every screen.

export type Layer = { mlp: number; groups: number; dropped: boolean };
export type Change = { layer: number; strategy: "narrow_mlp" | "remove_attention_group" | "drop_layer" | string };

export type RunSummary = {
  id: string;
  model: string;
  search: string; // search strategy: multi_layer | layer_by_layer
  search_label: string;
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
  has_model?: boolean;
  model_mb?: number | null;
  settings: {
    device: string;
    agent: string;
    ideas_per_round: number;
    max_drop_per_step: number;
    max_rounds: number;
    max_ai_calls: number;
    max_layers_removed: number;
    min_layer_width: number;
    sample_text: string;
    test_passages: number;
  };
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
  outcome: "ACCEPT" | "ROLLBACK" | "INVALID" | "LEFT" | null;
  why: string;
  chosen: number | null;
  candidates: Candidate[];
  refused_reasons: string[];
  agent_seconds: number;
  check_seconds: number;
};

export type Step = { round: number; error: number; size_mb: number; mac_mb: number | null; layers: Layer[] };

export type RunDetail = RunSummary & {
  shape: { layers: number; mlp_full: number; groups_full: number };
  flap: { size_mb: number; error: number | null; mac_mb: number | null; layers: Layer[] } | null;
  mac_context: number;
  bandwidth_mb_s: number | null;
  steps: Step[];
  rounds: Round[];
  time_left: { minutes: [number, number] | null; worst_minutes: number; rounds_left: number; mb_left: number } | null;
};

export type Score = { size_mb: number; test_error: number; unseen_error: number };
export type Accuracy = { all_questions: { acc: number; stderr: number; questions: number }; sets: Record<string, { acc: number; acc_stderr: number | null }> };
export type Results = { heldout: Record<string, Score> | null; accuracy: Record<string, Accuracy> | null };
export type Compare = {
  runs: {
    id: string; model: string; cut: number; agent: Score; flap: Score; original: Score; search: string; search_label: string;
    quiz: { agent?: number; flap?: number; original?: number } | null;
  }[];
  flap_curve: { cut: number; test_error: number }[];
  curve_model: string | null;
};

// ---------------------------------------------------------------- wording

export const modelName = (id: string) => (id.split("/").pop() ?? id).replace(/-Base$/, "");
export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
export const mb = (x: number) => `${Math.round(x).toLocaleString()} MB`;

/** How much worse than the original model, in plain words: "+22% worse". */
export const qualityLoss = (error: number, original: number) => error / original - 1;
export const lossText = (loss: number) => (loss < 0.005 ? "no loss" : loss >= 2 ? `${(loss + 1).toFixed(0)}× worse` : `${pct(loss)} worse`);

export function changeText(c: Change): string {
  const n = c.layer; // layers are numbered from 0, as in the code and the midpoint report
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
  for (const c of changes) byKind.set(c.strategy, [...(byKind.get(c.strategy) ?? []), c.layer]);
  const parts: string[] = [];
  for (const [kind, layers] of byKind) {
    const ls = [...new Set(layers)].sort((a, b) => a - b);
    const times = layers.length > ls.length ? " (twice)" : "";
    if (ls.length === 1) parts.push(changeText({ layer: ls[0], strategy: kind }) + times);
    else if (kind === "narrow_mlp") parts.push(`slim layers ${ls.join(", ")}${times}`);
    else if (kind === "remove_attention_group") parts.push(`remove attention units from layers ${ls.join(", ")}`);
    else if (kind === "drop_layer") parts.push(`remove layers ${ls.join(", ")} entirely`);
    else parts.push(`${kind} on layers ${ls.join(", ")}`);
  }
  return parts.join("; ");
}

/** Minutes in plain words, rounded to 5 minutes: "40 minutes", "1 h 20 min". */
export function duration(min: number): string {
  const r = Math.max(5, Math.round(min / 5) * 5);
  if (r < 60) return `${r} minutes`;
  const h = Math.floor(r / 60), m = r % 60;
  return m ? `${h} h ${m} min` : `${h} h`;
}

/** "40 minutes–1 h 10 min" or "20–30 minutes", or one value when both ends round the same. */
export function durationRange(lo: number, hi: number): string {
  const [a, b] = [duration(lo), duration(hi)];
  if (a === b) return a;
  if (a.endsWith(" minutes") && b.endsWith(" minutes")) return `${a.replace(" minutes", "")}–${b}`;
  return `${a}–${b}`;
}

/** The checks after a run (unseen text and 18,904 quiz questions, for the smaller model and FLAP's) grow with the
 * model's size. Measured on the M4: 0.6B (1.2 GB) about 1.5 h, 1.7B (3.4 GB) about 3.3 h, so 55-75 min per GB. */
export const checksMinutes = (originalMb?: number | null): [number, number] => {
  const gb = (originalMb ?? 1200) / 1000;
  return [gb * 55, gb * 75];
};

/** Time left for a running run, in words; says so when the round limit comes before the goal. */
export function timeLeftText(t: { minutes: [number, number] | null; worst_minutes: number; rounds_left: number; mb_left: number }): string {
  if (t.mb_left <= 0) return "Goal met; final check next";
  if (!t.minutes || t.minutes[0] >= t.worst_minutes)
    return `Likely to stop at the round limit before the goal (${t.rounds_left} rounds, about ${duration(t.worst_minutes)} left)`;
  return `About ${durationRange(t.minutes[0], t.minutes[1])} left, at most ${duration(t.worst_minutes)} (${t.rounds_left} rounds left)`;
}

export const outcomeText: Record<string, string> = {
  ACCEPT: "Kept",
  ROLLBACK: "Nothing kept (too much quality loss)",
  INVALID: "Refused (broke a rule; the AI was asked again)",
  LEFT: "Left as it is",
};

/** One-layer-per-step runs go through the layers in order: the layer a round decided (rounds the AI had to redo
 * don't count), or null for other runs. */
export function sweepLayer(run: { search: string; rounds: { round: number; outcome: string | null }[] }, round: number) {
  if (run.search !== "layer_by_layer") return null;
  const r = run.rounds.find((x) => x.round === round);
  if (!r || r.outcome === "INVALID") return null;
  return run.rounds.filter((x) => x.round < round && ["ACCEPT", "ROLLBACK", "LEFT"].includes(x.outcome ?? "")).length;
}

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
  running: "Running",
  paused: "Paused",
  interrupted: "Interrupted (resumable)",
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

/** How the AI cuts (search strategies, src/pruneagent/search.py). The checks after a run are the same for both. */
export const SEARCHES = [
  { id: "multi_layer", label: "Several layers per step", text: "Each step may cut any number of layers at once." },
  { id: "layer_by_layer", label: "One layer per step", text: "One pass, first layer to last: for each layer the AI proposes up to 3 cut plans from a fixed bank, and the best is kept." },
] as const;
