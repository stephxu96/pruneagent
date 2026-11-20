import { useEffect, useState } from "react";
import { api, duration, durationRange, checksMinutes, mb, modelName, pct, type Compare } from "./lib";
import { DEVICES, GB, NumberField, StepBar, type Feasibility, type Job, type Live, type Machine } from "./parts";
import { StartDialog } from "./StartDialog";

// Model pruning (docs/APP_DESIGN.md §2.2): select your model → where it runs → the answer → start. The size follows
// from the device and the feasibility check runs on its own; the user never chooses a percentage.

const STEPS = ["Select your model", "Device", "Estimate", "Start"];
const NEAR = 0.05; // earlier runs within 5 points of this size count as "at this size"

type Device = { id: string; memGb: number; free: number };

export function PrunePage() {
  const q = new URLSearchParams(window.location.hash.split("?")[1] ?? "");
  const [m, setM] = useState<Machine | null>(null);
  const [model, setModel] = useState<string | null>(q.get("model"));
  const [device, setDevice] = useState<Device | null>(null);
  const [at, setAt] = useState(q.get("model") ? 2 : 1);
  const [asking, setAsking] = useState(false);
  const [checked, setChecked] = useState<Feasibility | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    api<Machine>("/api/machine").then((x) => {
      setM(x);
      setDevice({ id: "this", memGb: Math.round(x.machine.memory_gb ?? 8), free: Math.round(x.free_share * 100) });
    }).catch((e) => setMsg(String(e)));
  }, []);

  const [search, setSearch] = useState<string>("multi_layer");
  if (!m || !device) return <p className="muted">Loading…</p>;
  const ready = m.models.filter((x) => x.ready);
  const sel = m.models.find((x) => x.id === model && x.ready);
  const step = !sel ? 1 : at;
  const fit = sel ? fitFor(sel, device, m.max_cut) : null;

  const start = () =>
    api<Job>("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: sel!.id, cut: fit?.kind === "cut" ? fit.cut : 0, backend: "codex", search }) })
      .then((j) => { window.location.hash = j.run_id ? `#/run/${j.run_id}` : "#/"; })
      .catch((e) => setMsg(String(e)));

  return (
    <div className="prune-page">
      <a href="#/" className="back">← Past runs</a>
      <h1>Prune a model</h1>
      <StepBar steps={STEPS} at={asking ? 4 : step} />

      <section className="wizard">
        {step === 1 && (
          <div className="choose-step">
            <span className="m-label">Models ready for pruning</span>
            {ready.length === 0 && <p className="muted">No models set up yet.</p>}
            <div className="model-picks">
              {ready.map((x) => (
                <button key={x.id} className={`pick ${x.id === model ? "on" : ""}`} onClick={() => { setModel(x.id); setAt(2); }}>
                  <b>{modelName(x.id)}</b>
                  <span className="muted">{mb(x.size_mb)}</span>
                </button>
              ))}
            </div>
            <a href="#/models">Model not listed? Set up a model →</a>
          </div>
        )}

        {step === 2 && sel && (
          <>
            <WhereItRuns m={m} model={sel} device={device} setDevice={setDevice} />
            <div className="actions">
              <button className="secondary" onClick={() => setAt(1)}>← Model</button>
              <button className="primary" onClick={() => setAt(3)} disabled={!fit || fit.kind === "too-small"}>Next →</button>
            </div>
          </>
        )}

        {step >= 3 && sel && fit && (
          <Answer model={sel} fit={fit} device={device} onBack={() => setAt(2)}
            onStart={(r) => { setChecked(r); setAsking(true); }} />
        )}
        {msg && <p className="warn">{msg}</p>}
      </section>

      {sel && fit?.kind === "cut" && (
        <StartDialog open={asking} model={sel.id} cut={fit.cut} originalMb={sel.size_mb} check={checked}
          search={search} onSearch={setSearch}
          onStart={start} onCheck={() => undefined} onClose={() => setAsking(false)} />
      )}
    </div>
  );
}

type Fit = { kind: "fits" } | { kind: "cut"; cut: number; room: number } | { kind: "too-small"; cut: number };

/** How much smaller the model must be to fit the device: model space = memory − other apps − working memory. */
function fitFor(model: Machine["models"][number], d: Device, maxCut: number): Fit {
  const total = d.memGb * GB;
  const room = total - total * (d.free / 100) - model.working_mb;
  if (model.size_mb <= room) return { kind: "fits" };
  const cut = Math.ceil((1 - room / model.size_mb) * 100) / 100; // whole percent, rounded up: a small buffer
  if (room <= 0 || cut > maxCut) return { kind: "too-small", cut: room > 0 ? cut : 1 };
  return { kind: "cut", cut, room };
}

function WhereItRuns({ m, model, device, setDevice }: {
  m: Machine;
  model: Machine["models"][number];
  device: Device;
  setDevice: (d: Device) => void;
}) {
  const [now, setNow] = useState<Live | null>(null);
  useEffect(() => {
    if (device.id !== "this") return;
    const read = () => api<Live>("/api/machine/live").then(setNow).catch(() => undefined);
    read();
    const t = setInterval(read, 3000);
    return () => clearInterval(t);
  }, [device.id]);
  const pick = (id: string) => {
    const memGb = id === "this" ? Math.round(m.machine.memory_gb ?? 8) : DEVICES.find((x) => x.id === id)?.memory_gb ?? device.memGb;
    setDevice({ ...device, id, memGb });
  };
  const total = device.memGb * GB;
  const others = total * (device.free / 100);
  const room = Math.max(0, total - others - model.working_mb);
  const parts = [
    { cls: "others", label: "Other apps", mb: others },
    { cls: "working", label: "Working memory", mb: model.working_mb },
    { cls: "room", label: "Space for the model", mb: room },
  ];
  const fit = fitFor(model, device, m.max_cut);
  return (
    <div className="where-step">
      <div className="m-grid">
        <label>
          <span className="m-label">Device</span>
          <select value={device.id} onChange={(e) => pick(e.target.value)}>
            <option value="this">This computer: {m.machine.name} · {m.machine.chip.replace("Apple ", "")}</option>
            {DEVICES.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
            <option value="custom">Custom</option>
          </select>
        </label>
        <label>
          <span className="m-label">Memory (GB)</span>
          <NumberField min={1} max={512} step={1} value={device.memGb} disabled={device.id !== "custom"}
            onCommit={(v) => setDevice({ ...device, memGb: v })} />
        </label>
        <label>
          <span className="m-label">Memory for other apps (%)</span>
          <NumberField min={0} max={95} step={1} value={device.free} onCommit={(v) => setDevice({ ...device, free: v })} />
        </label>
      </div>
      {device.id === "this" && now && (
        <div className="live">
          <span className="dot-live" /> Current state:
          {now.memory_used_pct != null && <span>memory <b>{now.memory_used_pct}%</b> in use</span>}
          {now.gpu_busy_pct != null && <span>GPU <b>{now.gpu_busy_pct}%</b> busy</span>}
        </div>
      )}
      <div>
        <div className="m-label">Memory ({device.memGb} GB)</div>
        <div className="stack" role="img" aria-label="How the memory is shared">
          {parts.map((p) => <span key={p.cls} className={`seg ${p.cls}`} style={{ width: `${(p.mb / total) * 100}%` }} title={`${p.label}: ${mb(p.mb)}`} />)}
        </div>
        <div className="legend">
          {parts.map((p) => <span key={p.cls}><i className={`sw ${p.cls}`} /> {p.label}: {mb(p.mb)}</span>)}
        </div>
      </div>
      {fit.kind === "too-small" && (
        <p className="warn">{modelName(model.id)} would have to be {pct(fit.cut)} smaller to fit. That's too much to cut;
          a smaller model or a device with more memory would work.</p>
      )}
    </div>
  );
}

function Answer({ model, fit, device, onBack, onStart }: {
  model: Machine["models"][number];
  fit: Fit;
  device: Device;
  onBack: () => void;
  onStart: (r: Feasibility) => void;
}) {
  const [check, setCheck] = useState<Job | null>(null);
  const [past, setPast] = useState<Compare | null>(null);
  const cut = fit.kind === "cut" ? fit.cut : null;
  const where = device.id === "this" ? "this computer" : DEVICES.find((x) => x.id === device.id)?.label.split(" · ")[0] ?? "this device";

  // the feasibility check runs on its own for this model and size
  // (a check from the last hour for the same model and size is reused)
  useEffect(() => {
    if (cut == null) return;
    setCheck(null);
    const recent = (j: Job) => j.kind === "precheck" && j.params.model === model.id && Math.abs((j.params.cut ?? -1) - cut) < 1e-6
      && (j.status === "done" || ["queued", "running"].includes(j.status)) && Date.now() - new Date(j.created).getTime() < 3600_000;
    api<Job[]>("/api/jobs").then((js) => {
      const found = js.find(recent);
      if (found) setCheck(found);
      else api<Job>("/api/precheck", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: model.id, cut, backend: "codex" }) }).then(setCheck).catch(() => undefined);
    }).catch(() => undefined);
    api<Compare>("/api/compare").then(setPast).catch(() => undefined);
  }, [model.id, cut]);
  useEffect(() => {
    if (!check || !["queued", "running"].includes(check.status)) return;
    const t = setTimeout(() => api<Job>(`/api/jobs/${check.id}`).then(setCheck).catch(() => undefined), 1500);
    return () => clearTimeout(t);
  }, [check]);

  if (fit.kind === "fits") {
    return (
      <div className="answer">
        <p className="answer-big ok-text">{modelName(model.id)} already fits {where} ({mb(model.size_mb)}). No pruning needed.</p>
        <div className="actions"><button className="secondary" onClick={onBack}>← Device</button></div>
      </div>
    );
  }
  if (fit.kind !== "cut" || cut == null) return null;
  const r: Feasibility | null = check?.result ?? null;
  const checking = !check || ["queued", "running"].includes(check.status);
  const near = (past?.runs ?? []).filter((x) => x.model === model.id && Math.abs(x.cut - cut) <= NEAR);
  const loss = (e: number, base: number) => e / base;
  const agent = near.map((x) => loss(x.agent.test_error, x.original.test_error));
  const flap = near.map((x) => loss(x.flap.test_error, x.original.test_error));
  const times = (v: number[]) => (v.length ? `${Math.min(...v).toFixed(1)}–${Math.max(...v).toFixed(1)}×` : null);
  const calls = r?.time ? Math.ceil(r.time.rounds[1]) : r?.usage?.calls;
  return (
    <div className="answer">
      <p className="answer-big">
        To fit {where}, {modelName(model.id)} must be <b>{pct(cut)} smaller</b>: {mb(model.size_mb)} → under {mb(model.size_mb * (1 - cut))}.
      </p>
      {checking && <p className="muted">Checking it can be done… about half a minute</p>}
      {!checking && r && !r.feasible && <div className="sd-note bad"><b>Can't be done at this size.</b> {r.stop_reason}</div>}
      {!checking && !r && <div className="sd-note bad"><b>The check didn't finish.</b> See Activity for the logs.</div>}
      {!checking && r?.feasible && (
        <div className="f-tiles">
          <div className="f-tile">
            <div className="f-label">Expected quality</div>
            <div className="f-big">{times(agent) ? `${times(agent)} worse` : "Not known yet"}</div>
            <div className="f-sub">
              {times(agent) ? `than the original, from ${near.length} earlier run${near.length > 1 ? "s" : ""} at this size. The standard method: ${times(flap)} worse`
                : "No earlier runs of this model near this size"}
            </div>
          </div>
          <div className="f-tile">
            <div className="f-label">Time</div>
            <div className="f-big">{r.time ? durationRange(r.time.minutes[0] + checksMinutes(model.size_mb)[0], r.time.minutes[1] + checksMinutes(model.size_mb)[1]) : "Not known yet"}</div>
            <div className="f-sub">
              {r.time ? <>Pruning and evaluation{r.time.worst_minutes != null && <>; up to {duration(r.time.worst_minutes + checksMinutes(model.size_mb)[1])}</>}</> : "No earlier runs of this model"}
            </div>
          </div>
          <div className="f-tile">
            <div className="f-label">Usage limits</div>
            {(r.usage?.windows ?? []).map((w) => {
              const cost = w.pct_per_call * (calls ?? r.usage!.calls);
              return <div key={w.name} className="f-limit"><span>{w.name === "weekly" ? "Weekly" : w.name}</span><b>~{cost < 1 ? cost.toFixed(1) : Math.ceil(cost)}%</b></div>;
            })}
            <div className="f-sub">Share of each limit this run uses</div>
          </div>
        </div>
      )}
      <div className="actions">
        <button className="secondary" onClick={onBack}>← Device</button>
        <button className="primary" disabled={checking || !r?.feasible} onClick={() => r && onStart(r)}>Start →</button>
      </div>
    </div>
  );
}
