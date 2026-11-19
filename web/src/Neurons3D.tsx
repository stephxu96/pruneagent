import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { Layer } from "./lib";

// The model as a 3D cloud of neurons. Each slab is one layer, left to right. In a slab, every dot stands for
// 64 neurons of the layer's knowledge part; the row of cubes on top are its attention units. Cut parts shrink
// and fade; parts cut in the current step flash orange first. Which dots fade is illustrative (the real choice of
// neurons is made by their scores); how many fade is exact.

export const NEURONS_PER_DOT = 64;
const GAP_X = 1.0;
const STEP_Y = 0.5;
const STEP = 0.32;
const ROWS = 8; // neurons stack up in a column per layer, like the bars of the flat view

type Status = 0 | 1 | 2; // 0 kept, 1 cut earlier, 2 cut in this step

function colors() {
  const css = getComputedStyle(document.documentElement);
  const get = (name: string, fallback: string) => new THREE.Color(css.getPropertyValue(name).trim() || fallback);
  const bg = get("--panel", "#1f1f1e");
  return {
    kept: get("--part", "#a9b6c2"),
    attn: get("--ink-2", "#c3c2b7"),
    hot: get("--agent", "#d95926"),
    ghost: get("--part", "#a9b6c2").lerp(bg, 0.7),
    wire: get("--part", "#a9b6c2").lerp(bg, 0.86),
  };
}

function layout(n: number, dots: number, groups: number) {
  const cols = Math.ceil(dots / ROWS);
  const neurons: THREE.Vector3[] = [];
  const attn: THREE.Vector3[] = [];
  for (let i = 0; i < n; i++) {
    const x = (i - (n - 1) / 2) * GAP_X;
    for (let d = 0; d < dots; d++) {
      // fill from the bottom row up, so a slab loses its top rows first (like the flat view's bars)
      const r = Math.floor(d / cols);
      const c = d % cols;
      neurons.push(new THREE.Vector3(x, (r - (ROWS - 1) / 2) * STEP_Y, (c - (cols - 1) / 2) * STEP));
    }
    for (let g = 0; g < groups; g++) attn.push(new THREE.Vector3(x, ((ROWS + 1) / 2) * STEP_Y + 0.25, (g - (groups - 1) / 2) * STEP));
  }
  return { neurons, attn };
}

function status(layers: Layer[], prev: Layer[] | undefined, perLayer: number, kind: "mlp" | "groups", full: number): Status[] {
  const out: Status[] = [];
  const keep = (l: Layer | undefined) => (l === undefined ? perLayer : l.dropped ? 0 : Math.round((l[kind] / full) * perLayer));
  layers.forEach((l, i) => {
    const now = keep(l);
    const before = prev ? keep(prev[i]) : now;
    for (let d = 0; d < perLayer; d++) out.push(d < now ? 0 : d < before ? 2 : 1);
  });
  return out;
}

function Dots({ positions, states, size, cube, palette, keptColor }: {
  positions: THREE.Vector3[];
  states: Status[];
  size: number;
  cube?: boolean;
  palette: ReturnType<typeof colors>;
  keptColor: THREE.Color;
}) {
  const ref = useRef<THREE.InstancedMesh>(null!);
  const n = positions.length;
  const cur = useMemo(() => ({ scale: new Float32Array(n).fill(1), color: Array.from({ length: n }, () => keptColor.clone()), age: new Float32Array(n), first: true }), [n, keptColor]);
  const m = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    cur.age.fill(0); // restart the orange flash when the step changes
  }, [states, cur]);

  useFrame((_, dt) => {
    const mesh = ref.current;
    if (!mesh) return;
    // first frame: jump straight to the current state (no fade from a full model); later frames: animate
    const k = cur.first ? 1 : 1 - Math.exp(-dt * 4);
    if (cur.first) cur.age.fill(10);
    cur.first = false;
    for (let i = 0; i < n; i++) {
      const s = states[i];
      cur.age[i] += dt;
      const flashing = s === 2 && cur.age[i] < 1.4;
      const targetScale = s === 0 ? 1 : flashing ? 1.3 : 0.55;
      const target = s === 0 ? keptColor : flashing ? palette.hot : palette.ghost;
      cur.scale[i] += (targetScale - cur.scale[i]) * k;
      cur.color[i].lerp(target, k);
      m.position.copy(positions[i]);
      m.scale.setScalar(cur.scale[i] * size);
      m.updateMatrix();
      mesh.setMatrixAt(i, m.matrix);
      mesh.setColorAt(i, cur.color[i]);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, n]}>
      {cube ? <boxGeometry args={[1, 1, 1]} /> : <sphereGeometry args={[0.5, 12, 10]} />}
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

function Wires({ positions, states, perLayer, layers, color }: {
  positions: THREE.Vector3[];
  states: Status[];
  perLayer: number;
  layers: number;
  color: THREE.Color;
}) {
  const geom = useMemo(() => {
    const pts: number[] = [];
    for (let i = 0; i + 1 < layers; i++) {
      for (let d = 0; d < perLayer; d += 2) {
        const a = i * perLayer + d;
        const b = (i + 1) * perLayer + ((d * 7 + 3) % perLayer);
        if (states[a] !== 0 || states[b] !== 0) continue;
        pts.push(positions[a].x, positions[a].y, positions[a].z, positions[b].x, positions[b].y, positions[b].z);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, [positions, states, perLayer, layers]);
  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial color={color} transparent opacity={0.3} />
    </lineSegments>
  );
}

export function Neurons3D({ layers, prev, full, caption }: {
  layers: Layer[];
  prev?: Layer[];
  full: { mlp: number; groups: number };
  caption?: string;
}) {
  const palette = useMemo(colors, []);
  const dots = Math.round(full.mlp / NEURONS_PER_DOT);
  const { neurons, attn } = useMemo(() => layout(layers.length, dots, full.groups), [layers.length, dots, full.groups]);
  const nStates = useMemo(() => status(layers, prev, dots, "mlp", full.mlp), [layers, prev, dots, full.mlp]);
  const aStates = useMemo(() => status(layers, prev, full.groups, "groups", full.groups), [layers, prev, full.groups]);
  const width = layers.length * GAP_X;
  return (
    <figure className="neurons3d">
      <div className="canvas-wrap">
        <Canvas camera={{ position: [width * 0.04, width * 0.3, width * 0.72], fov: 36 }} dpr={[1, 2]}>
          <Dots positions={neurons} states={nStates} size={0.2} palette={palette} keptColor={palette.kept} />
          <Dots positions={attn} states={aStates} size={0.2} cube palette={palette} keptColor={palette.attn} />
          <Wires positions={neurons} states={nStates} perLayer={dots} layers={layers.length} color={palette.wire} />
          <OrbitControls enablePan={false} enableDamping minDistance={6} maxDistance={width * 1.2} />
        </Canvas>
      </div>
      <div className="legend">
        <span><i className="sw part dot" /> {NEURONS_PER_DOT} neurons</span>
        <span><i className="sw attn" /> attention unit</span>
        <span><i className="sw hot dot" /> cut in this step</span>
        <span><i className="sw gone dot" /> cut earlier</span>
        <span className="muted">Drag to turn. Layers 0 to {layers.length - 1}, left to right.</span>
      </div>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}
