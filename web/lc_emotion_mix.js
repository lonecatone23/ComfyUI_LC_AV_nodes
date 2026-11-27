/**
 * LC Emotion Editor: a valence / arousal pad (drag the dot) and emotion buttons. A button adds "name:0.5" to the mix and
 * moves the dot to where that feeling sits on the pad; clicking it again takes it back out. Each emotion's spot is
 * drawn faintly on the pad. Purple buttons are saved / premade emotions (click to load one); save emotion keeps the
 * current feeling by voice_name.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "LCEmotionMix";
// where each feeling sits: [valence (unpleasant -1 .. pleasant +1), arousal (calm -1 .. energetic +1)]
const SPOTS = {
  // same spots as SPOTS in lc_tts_engine.py (the pad turns into these IndexTTS-2 emotions)
  happy: [0.65, 0.35], surprised: [0.3, 0.75], angry: [-0.45, 0.65], afraid: [-0.75, 0.3],
  disgusted: [-0.6, 0.05], sad: [-0.6, -0.5], melancholic: [-0.25, -0.8], calm: [0.45, -0.6],
};
// the pad is drawn like LC Lighting Control V2's stage: a fixed square at 2x resolution (sharp at any zoom), the axis
// names in the margin around the playing area so they never sit on an emotion's spot
const PAD_H = 268; // square pad, CSS px (the same size as Lighting V2's stage)
const MARGIN = 18;
const SCALE = 2; // canvas pixels per CSS pixel
const CHIP_ROW = 26; // a button row: 22 px button + 4 px gap
const usable = () => PAD_H - 2 * MARGIN;
const toPx = (v, a) => ({ x: MARGIN + ((v + 1) / 2) * usable(), y: MARGIN + ((1 - a) / 2) * usable() });

function w(node, name) {
  return (node.widgets || []).find((x) => x.name === name);
}

function setVal(node, name, v) {
  const wd = w(node, name);
  if (!wd) return;
  wd.value = Math.round(Math.max(-1, Math.min(1, v)) * 20) / 20;
  try {
    wd.callback?.(wd.value);
  } catch (_e) {}
  node.setDirtyCanvas?.(true, true);
}

function mixParts(node) {
  return String(w(node, "mix")?.value || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function inMix(node, name) {
  return mixParts(node).some((p) => p.split(":")[0].trim().toLowerCase() === name);
}

function draw(node, canvas) {
  const ctx = canvas.getContext("2d");
  const S = PAD_H;
  ctx.setTransform(canvas.width / S, 0, 0, canvas.width / S, 0, 0);
  ctx.clearRect(0, 0, S, S);
  ctx.fillStyle = "#1a1f1f";
  ctx.fillRect(0, 0, S, S);
  ctx.strokeStyle = "#3a4545";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, S - 1, S - 1);
  // the playing area and its crosshair (neutral in the middle)
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.strokeRect(MARGIN + 0.5, MARGIN + 0.5, usable() - 1, usable() - 1);
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath();
  ctx.moveTo(MARGIN, S / 2 + 0.5);
  ctx.lineTo(S - MARGIN, S / 2 + 0.5);
  ctx.moveTo(S / 2 + 0.5, MARGIN);
  ctx.lineTo(S / 2 + 0.5, S - MARGIN);
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.beginPath();
  ctx.arc(S / 2, S / 2, usable() * 0.06, 0, Math.PI * 2);
  ctx.stroke();
  // the axis names, in the margin
  ctx.fillStyle = "rgba(255,255,255,0.4)";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("energetic", S / 2, MARGIN / 2);
  ctx.fillText("calm", S / 2, S - MARGIN / 2);
  ctx.save();
  ctx.translate(S - MARGIN / 2, S / 2);
  ctx.rotate(Math.PI / 2);
  ctx.fillText("pleasant", 0, 0);
  ctx.restore();
  ctx.save();
  ctx.translate(MARGIN / 2, S / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("unpleasant", 0, 0);
  ctx.restore();
  // the dot: where the mix sits (under the names, so a name it sits on stays readable)
  const v = Number(w(node, "valence")?.value) || 0;
  const a = Number(w(node, "arousal")?.value) || 0;
  const d = toPx(v, a);
  ctx.fillStyle = "#e0a040";
  ctx.beginPath();
  ctx.arc(d.x, d.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.45)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  // each emotion's spot, brighter when it is in the mix
  ctx.font = "10px sans-serif";
  ctx.textBaseline = "alphabetic";
  for (const name of node._lcChipNames || []) {
    const s = SPOTS[name];
    if (!s) continue;
    const p = toPx(s[0], s[1]);
    const on = inMix(node, name);
    ctx.fillStyle = on ? "rgba(224,160,64,0.95)" : "rgba(170,180,195,0.55)";
    ctx.beginPath();
    ctx.arc(p.x, p.y, on ? 3 : 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = on ? "bold 10px sans-serif" : "10px sans-serif";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(26,31,31,0.85)";
    ctx.strokeText(name, p.x, p.y - 6);
    ctx.fillText(name, p.x, p.y - 6);
  }
}

function toggleChip(node, name) {
  const mw = w(node, "mix");
  if (!mw) return;
  let parts = mixParts(node);
  const had = inMix(node, name);
  parts = had ? parts.filter((p) => p.split(":")[0].trim().toLowerCase() !== name) : [...parts, `${name}:0.5`];
  mw.value = parts.join(", ");
  try {
    mw.callback?.(mw.value);
  } catch (_e) {}
  // the dot sits on the average spot of the emotions now in the mix (center when none has a spot)
  const spots = parts.map((p) => SPOTS[p.split(":")[0].trim().toLowerCase()]).filter(Boolean);
  const avg = (i) => (spots.length ? spots.reduce((s, p) => s + p[i], 0) / spots.length : 0);
  setVal(node, "valence", avg(0));
  setVal(node, "arousal", avg(1));
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "LC.AV.EmotionMix",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const node = this;
      node._lcChipNames = [];
      const box = document.createElement("div");
      box.style.cssText = "display:flex;flex-direction:column;gap:8px;width:100%;";
      const canvas = document.createElement("canvas");
      canvas.width = PAD_H * SCALE;
      canvas.height = PAD_H * SCALE;
      canvas.style.cssText = `width:${PAD_H}px;height:${PAD_H}px;flex:0 0 ${PAD_H}px;border-radius:6px;cursor:crosshair;align-self:center;touch-action:none;`;
      const chips = document.createElement("div");
      chips.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;justify-content:center;align-content:flex-start;";
      box.append(canvas, chips);
      const paint = () => {
        draw(node, canvas);
        for (const b of chips.children) b.style.outline = inMix(node, b.dataset.name) ? "2px solid #e0a040" : "none";
      };
      const pick = (ev) => {
        const rect = canvas.getBoundingClientRect();
        // pad pixels at any canvas zoom, then the playing area inside the margin
        const px = ((ev.clientX - rect.left) * PAD_H) / Math.max(1, rect.width);
        const py = ((ev.clientY - rect.top) * PAD_H) / Math.max(1, rect.height);
        const x = Math.max(0, Math.min(1, (px - MARGIN) / usable()));
        const y = Math.max(0, Math.min(1, (py - MARGIN) / usable()));
        setVal(node, "valence", x * 2 - 1);
        setVal(node, "arousal", 1 - y * 2);
        paint();
      };
      let down = false;
      canvas.addEventListener("pointerdown", (e) => {
        down = true;
        canvas.setPointerCapture(e.pointerId);
        pick(e);
        e.stopPropagation();
      });
      canvas.addEventListener("pointermove", (e) => down && pick(e));
      canvas.addEventListener("pointerup", () => (down = false));
      canvas.addEventListener("dblclick", () => {
        setVal(node, "valence", 0);
        setVal(node, "arousal", 0);
        paint();
      });
      // pad + every row of buttons: rows are counted from the buttons' own positions, so the node always fits them
      const rows = () => {
        const tops = new Set([...chips.children].map((b) => b.offsetTop));
        return Math.max(1, tops.size || Math.ceil(chips.children.length / 4));
      };
      const padHeight = () => PAD_H + 8 + rows() * CHIP_ROW + 20;
      this.addDOMWidget("emotion_pad", "lc_emotion_pad", box, { serialize: false, getMinHeight: padHeight, getMaxHeight: padHeight });
      const fit = () => {
        const need = node.computeSize();
        node.setSize([Math.max(node.size[0], need[0]), Math.max(node.size[1], need[1])]);
        node.setDirtyCanvas?.(true, true);
      };
      const chipCss = (color) =>
        `background:${color};color:#eee;border:0;border-radius:11px;height:22px;line-height:22px;padding:0 9px;font-size:11px;cursor:pointer;`;
      // blue = the engine's emotions (toggle into the mix), purple = saved and premade emotions (click to load one)
      const loadChips = () =>
        api
          .fetchApi("/lc_av/emotions")
          .then((res) => res.json())
          .then((j) => {
            chips.replaceChildren();
            node._lcChipNames = [];
            for (const name of j.built_in || []) {
              node._lcChipNames.push(name);
              const b = document.createElement("button");
              b.textContent = name;
              b.dataset.name = name;
              b.title = "Add to the mix and move the dot to it (click again to take it out)";
              b.style.cssText = chipCss("#3a5a7a");
              b.onclick = () => {
                toggleChip(node, name);
                paint();
              };
              chips.append(b);
            }
            for (const name of j.designed || []) {
              const e = (j.saved || {})[name] || {};
              const b = document.createElement("button");
              b.textContent = name;
              b.title = `Load ${name}: ${e.mix || "pad only"}${e.premade ? " (premade)" : ""}`;
              b.style.cssText = chipCss("#6a4a7a");
              b.onclick = () => {
                for (const [k, v] of [["voice_name", name], ["mix", e.mix || ""]]) {
                  const wd = w(node, k);
                  if (wd) {
                    wd.value = v;
                    try {
                      wd.callback?.(v);
                    } catch (_e) {}
                  }
                }
                setVal(node, "valence", e.valence || 0);
                setVal(node, "arousal", e.arousal || 0);
                paint();
              };
              chips.append(b);
            }
            requestAnimationFrame(() => {
              paint();
              fit();
            });
          })
          .catch(() => {});
      loadChips();
      // save emotion: keeps the feeling (mix + pad) by voice_name as one small file; no audio, no voice
      const saveBtn = this.addWidget("button", "💾 save emotion", null, async () => {
        const name = String(w(node, "voice_name")?.value || "").trim();
        const body = { name, mix: w(node, "mix")?.value || "", valence: w(node, "valence")?.value || 0, arousal: w(node, "arousal")?.value || 0 };
        let res;
        try {
          res = await (await api.fetchApi("/lc_av/emotions/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
        } catch (e) {
          res = { error: String(e) };
        }
        const toast = app.extensionManager?.toast;
        if (res.error) toast?.add({ severity: "warn", summary: "LC Emotion Editor", detail: res.error, life: 5000 });
        else {
          toast?.add({ severity: "success", summary: "LC Emotion Editor", detail: `Saved "${res.name}". Use it as [Name|${res.name}].`, life: 4000 });
          loadChips();
        }
      });
      saveBtn.serialize = false;
      for (const n of ["valence", "arousal", "mix"]) {
        const wd = w(node, n);
        const cb = wd?.callback;
        if (wd)
          wd.callback = function () {
            const out = cb?.apply(this, arguments);
            paint();
            return out;
          };
      }
      requestAnimationFrame(() => {
        paint();
        fit();
      });
      return r;
    };
  },
});
