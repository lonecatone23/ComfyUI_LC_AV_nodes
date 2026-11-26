/**
 * LC Voice Shape: one wide graph instead of nine sliders, drawn like the LC LoRA editors. A centerline is "as spoken";
 * drag a column up or down to move that part of the voice (shift-click or double-click puts it back). The sliders stay
 * as the saved values (hidden), so workflows, presets and the API work as before.
 * Presets: picking one moves every column to its values; moving a column switches the preset to Custom. On load a
 * named preset re-applies its current values; Custom keeps the saved ones.
 * One DOM canvas, so it looks the same in Nodes classic and Nodes 2.0.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "LCVoiceShape";
// name, label under the column, neutral ("as spoken"), how the value reads
const COLS = [
  ["pitch", "pitch", 0, (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} st`],
  ["voice_size", "voice size", 1, (v) => `${v.toFixed(2)} ${v < 1 ? "bigger" : v > 1 ? "smaller" : ""}`.trim()],
  ["liveliness", "liveliness", 1, (v) => v.toFixed(2)],
  ["speed", "speed", 1, (v) => `${v.toFixed(2)}x`],
  ["warmth_db", "warmth", 0, (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`],
  ["softness_db", "softness", 0, (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`],
  ["air_db", "air", 0, (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`],
  ["tremble", "tremble", 0, (v) => v.toFixed(2)],
  ["breathiness", "breath", 0, (v) => v.toFixed(2)],
];
const SLIDERS = COLS.map((c) => c[0]);
const GRAPH_H = 210; // the bars
const LABEL_H = 58; // sideways labels under them
const HINT_H = 18; // the line that explains the column under the pointer
const PAD_H = GRAPH_H + LABEL_H + HINT_H;
const MIN_W = 380; // the floor: half the old 760 default
const DEFAULT_W = 380;
const UP = "rgba(255,150,80,0.8)";
const DOWN = "rgba(120,170,255,0.75)";

let PRESETS = null;
const loading = api
  .fetchApi("/lc_av/voice_shape_presets")
  .then((r) => r.json())
  .then((j) => (PRESETS = j.presets || {}))
  .catch(() => (PRESETS = {}));

const w = (node, name) => (node.widgets || []).find((x) => x.name === name);

// ---------------------------------------------------------------- values <-> height
function spec(node, name) {
  const s = node._lcVS?.specs?.[name] || {};
  const col = COLS.find((c) => c[0] === name);
  return { min: s.min ?? -1, max: s.max ?? 1, step: s.step ?? 0.01, rest: col ? col[2] : 0, tip: s.tooltip || "" };
}

/** value -> -1..1 around the centerline (up to max, down to min; a column whose minimum is "as spoken" only goes up) */
function toUnit(node, name, v) {
  const s = spec(node, name);
  if (v >= s.rest) return s.max > s.rest ? (v - s.rest) / (s.max - s.rest) : 0;
  return s.min < s.rest ? -(s.rest - v) / (s.rest - s.min) : 0;
}

function fromUnit(node, name, u) {
  const s = spec(node, name);
  let v = u >= 0 ? s.rest + u * (s.max - s.rest) : s.rest + u * (s.rest - s.min);
  v = Math.round(v / s.step) * s.step;
  return Math.max(s.min, Math.min(s.max, +v.toFixed(4)));
}

function value(node, name) {
  const v = Number(w(node, name)?.value);
  return Number.isFinite(v) ? v : spec(node, name).rest;
}

// ---------------------------------------------------------------- presets
function apply(node, name) {
  const p = PRESETS?.[name];
  if (!p) return;
  node._lcShapeSetting = true;
  try {
    for (const s of SLIDERS) {
      const wd = w(node, s);
      if (wd && p[s] !== undefined) {
        wd.value = p[s];
        try {
          wd.callback?.(wd.value);
        } catch (_e) {}
      }
    }
  } finally {
    node._lcShapeSetting = false;
  }
  node.setDirtyCanvas?.(true, true);
  draw(node);
}

function setValue(node, name, v) {
  const wd = w(node, name);
  if (!wd || Number(wd.value) === v) return;
  wd.value = v;
  try {
    wd.callback?.(v); // goes Custom when it really changed (see hook)
  } catch (_e) {}
  node.setDirtyCanvas?.(true, true);
}

function hook(node) {
  const preset = w(node, "preset");
  if (!preset || node._lcShapeHooked) return;
  node._lcShapeHooked = true;
  const pcb = preset.callback;
  preset.callback = function (v) {
    const r = pcb?.apply(this, arguments);
    if (v !== "Custom") loading.then(() => apply(node, v));
    draw(node);
    return r;
  };
  for (const s of SLIDERS) {
    const wd = w(node, s);
    if (!wd) continue;
    const cb = wd.callback;
    wd.callback = function () {
      const r = cb?.apply(this, arguments);
      if (!node._lcShapeSetting && preset.value !== "Custom") {
        const p = PRESETS?.[preset.value];
        // only a real change from the preset's own value counts as going custom
        if (!p || Math.abs(Number(wd.value) - Number(p[s])) > 1e-6) {
          preset.value = "Custom";
          node.setDirtyCanvas?.(true, true);
        }
      }
      draw(node);
      return r;
    };
  }
}

// ---------------------------------------------------------------- drawing
function layout(cv) {
  const cw = Math.max(200, cv.clientWidth || DEFAULT_W);
  const L = 8, R = 8;
  const gw = cw - L - R;
  const n = COLS.length;
  const top = 18, h = GRAPH_H - top - 6; // value text row above the bars
  return { cw, L, gw, n, cwid: gw / n, top, h, mid: top + h / 2 };
}

function draw(node) {
  const ui = node._lcVS;
  if (!ui?.canvas?.isConnected) return;
  const cv = ui.canvas;
  const dpr = window.devicePixelRatio || 1;
  const lay = layout(cv);
  if (cv.width !== Math.round(lay.cw * dpr) || cv.height !== Math.round(PAD_H * dpr)) {
    cv.width = Math.round(lay.cw * dpr);
    cv.height = Math.round(PAD_H * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, lay.cw, PAD_H);
  const { L, gw, cwid, top, h, mid } = lay;

  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.fillRect(L, top, gw, h);
  // column guides
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  for (let i = 1; i < COLS.length; i++) {
    ctx.beginPath();
    ctx.moveTo(L + i * cwid + 0.5, top);
    ctx.lineTo(L + i * cwid + 0.5, top + h);
    ctx.stroke();
  }
  // the centerline: the voice as spoken
  ctx.strokeStyle = "rgba(255,255,255,0.3)";
  ctx.beginPath();
  ctx.moveTo(L, mid + 0.5);
  ctx.lineTo(L + gw, mid + 0.5);
  ctx.stroke();

  const bw = Math.min(46, cwid * 0.55);
  const narrow = cwid < 64; // a narrow node: smaller value text, no "bigger / smaller" word
  ctx.font = narrow ? "9px sans-serif" : "10px sans-serif";
  ctx.textAlign = "center";
  COLS.forEach(([name, label, , fmt], i) => {
    const cx = L + (i + 0.5) * cwid;
    const s = spec(node, name);
    const v = value(node, name);
    const u = toUnit(node, name, v);
    const y = mid - u * (h / 2);
    const hot = i === ui.hover || i === ui.drag;
    if (s.min >= s.rest) {  // only goes up: grey the lower half
      ctx.fillStyle = "rgba(255,255,255,0.03)";
      ctx.fillRect(L + i * cwid + 1, mid, cwid - 2, h / 2);
    }
    ctx.fillStyle = hot ? "#f0d9a8" : u > 0 ? UP : u < 0 ? DOWN : "rgba(212,164,79,0.7)";
    ctx.fillRect(cx - bw / 2, Math.min(y, mid), bw, Math.max(1.5, Math.abs(y - mid)));
    ctx.fillRect(cx - bw / 2 - 2, y - 1.5, bw + 4, 3); // the handle
    ctx.fillStyle = hot ? "#f0d9a8" : "rgba(255,255,255,0.7)";
    ctx.fillText(fmt(v), cx, top - 5);
    // the label, sideways under the column
    ctx.save();
    ctx.translate(cx + 4, top + h + 8);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = "right";
    ctx.font = hot ? "bold 11px sans-serif" : "11px sans-serif";
    ctx.fillStyle = hot ? "#f0d9a8" : "rgba(255,255,255,0.75)";
    ctx.fillText(label, 0, 0);
    ctx.restore();
  });

  // what the column under the pointer does (its tooltip)
  const i = ui.drag >= 0 ? ui.drag : ui.hover;
  ctx.textAlign = "left";
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  const tip = i >= 0
    ? `${COLS[i][1]}: ${spec(node, COLS[i][0]).tip}`
    : "Drag a column up or down. Shift-click or double-click puts it back on the line (as spoken).";
  ctx.fillText(tip, 4, PAD_H - 5, lay.cw - 8);
}

function hitAt(node, ev) {
  const cv = node._lcVS.canvas;
  const r = cv.getBoundingClientRect();
  const sx = (cv.clientWidth || r.width) / r.width; // works at any canvas zoom
  const x = (ev.clientX - r.left) * sx;
  const y = (ev.clientY - r.top) * sx;
  const lay = layout(cv);
  if (x < lay.L || x > lay.L + lay.gw || y > lay.top + lay.h + LABEL_H) return null;
  const i = Math.max(0, Math.min(lay.n - 1, Math.floor((x - lay.L) / lay.cwid)));
  const u = Math.max(-1, Math.min(1, (lay.mid - y) / (lay.h / 2)));
  return { i, u, inBars: y <= lay.top + lay.h };
}

// ---------------------------------------------------------------- the widget
function setup(node, specs) {
  if (node._lcVS) return;
  for (const s of SLIDERS) {  // the graph replaces the sliders; they keep the values
    const wd = w(node, s);
    if (!wd) continue;
    wd.hidden = true;
    wd.type = "hidden";
    wd.computeSize = () => [0, -4];
  }
  const canvas = document.createElement("canvas");
  canvas.style.cssText = `width:100%;height:${PAD_H}px;display:block;cursor:ns-resize;touch-action:none`;
  const wrap = document.createElement("div");
  wrap.style.cssText = `width:100%;height:${PAD_H}px;`;
  wrap.appendChild(canvas);
  node._lcVS = { canvas, specs, hover: -1, drag: -1 };
  const gw = node.addDOMWidget("lc_voice_graph", "lc_voice_graph", wrap, {
    serialize: false,
    getMinHeight: () => PAD_H,
    getMaxHeight: () => PAD_H,
  });
  if (gw) gw.serialize = false;

  const put = (i, u, ev) => {
    const name = COLS[i][0];
    setValue(node, name, ev.shiftKey ? spec(node, name).rest : fromUnit(node, name, u));
    draw(node);
  };
  canvas.addEventListener("pointerdown", (ev) => {
    const hit = hitAt(node, ev);
    if (!hit || !hit.inBars) return;
    ev.stopPropagation();
    ev.preventDefault();
    try {
      canvas.setPointerCapture(ev.pointerId);
    } catch {
      // no capture: the drag still works while the pointer stays on the graph
    }
    node._lcVS.drag = hit.i; // the column you grabbed, even if the pointer drifts sideways
    put(hit.i, hit.u, ev);
  });
  canvas.addEventListener("pointermove", (ev) => {
    const hit = hitAt(node, ev);
    if (node._lcVS.drag >= 0) {
      ev.stopPropagation();
      if (hit) put(node._lcVS.drag, hit.u, ev);
      return;
    }
    const hv = hit ? hit.i : -1;
    if (hv !== node._lcVS.hover) {
      node._lcVS.hover = hv;
      draw(node);
    }
  });
  const end = () => {
    node._lcVS.drag = -1;
    draw(node);
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("pointerleave", () => {
    node._lcVS.hover = -1;
    if (node._lcVS.drag < 0) draw(node);
  });
  canvas.addEventListener("dblclick", (ev) => {
    const hit = hitAt(node, ev);
    if (!hit) return;
    ev.stopPropagation();
    setValue(node, COLS[hit.i][0], spec(node, COLS[hit.i][0]).rest);
    draw(node);
  });
  new ResizeObserver(() => draw(node)).observe(canvas);
  setTimeout(() => draw(node), 50);
}

app.registerExtension({
  name: "LC.AV.VoiceShape",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;
    const req = nodeData.input?.required || {};
    const specs = {};
    for (const s of SLIDERS) specs[s] = req[s]?.[1] || {};
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      setup(this, specs);
      hook(this);
      // landscape: wide for the graph, only as tall as it needs
      requestAnimationFrame(() => {
        const need = this.computeSize?.()?.[1];
        this.setSize?.([Math.max(DEFAULT_W, this.size[0]), need || this.size[1]]);
      });
      return r;
    };
    // workflows saved with the graph's empty value: drop it before the frontend's cleanup miscounts and shifts values
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const vals = info?.widgets_values;
      const gi = (this.widgets || []).findIndex((x) => x.name === "lc_voice_graph");
      if (Array.isArray(vals) && gi >= 0 && vals.length === this.widgets.length && vals[gi] === "") {
        info.widgets_values = vals.filter((_, i) => i !== gi);
      }
      return configure.apply(this, arguments);
    };
    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      if (size && size[0] < MIN_W) size[0] = MIN_W; // narrower and the labels run into each other
      return onResize?.apply(this, arguments);
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      hook(this);
      if (this.size[0] < MIN_W) this.size[0] = MIN_W;
      requestAnimationFrame(() => {  // saved before the graph: shrink the old slider column to the graph's height
        const need = this.computeSize?.()?.[1];
        if (need) this.setSize([this.size[0], need]);
      });
      const name = w(this, "preset")?.value;
      if (name && name !== "Custom") loading.then(() => apply(this, name)); // named presets follow their current values
      setTimeout(() => draw(this), 50);
      return r;
    };
  },
});
