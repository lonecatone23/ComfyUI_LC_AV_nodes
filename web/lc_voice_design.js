/**
 * LC Voice Design: picking a preset fills the eight dropdowns, picking a script fills the sample_text box, so you see
 * what will be spoken. Changing a dropdown by hand switches the preset to Custom; typing in the box switches the
 * script to Custom (the box is then what Maya reads). On load a named preset / script re-applies its current values;
 * Custom keeps the saved ones.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "LCVoiceDesign";
const PARTS = ["sex", "age", "accent", "pitch", "timbre", "tone", "intensity", "pacing"];

let DATA = null;
const loading = api
  .fetchApi("/lc_av/voice_design_presets")
  .then((r) => r.json())
  .then((j) => (DATA = { presets: j.presets || {}, scripts: j.scripts || {} }))
  .catch(() => (DATA = { presets: {}, scripts: {} }));

const w = (node, name) => (node.widgets || []).find((x) => x.name === name);

function setValue(node, name, value) {
  const wd = w(node, name);
  if (!wd || wd.value === value) return;
  wd.value = value;
  if (wd.inputEl && wd.inputEl.value !== value) wd.inputEl.value = value; // the multiline box shows it at once
}

function applyPreset(node) {
  const p = DATA?.presets?.[w(node, "preset")?.value];
  if (!p) return;
  node._lcVDSetting = true;
  try {
    for (const k of PARTS) if (p[k] !== undefined) setValue(node, k, p[k]);
  } finally {
    node._lcVDSetting = false;
  }
  node.setDirtyCanvas?.(true, true);
}

function applyScript(node) {
  const t = DATA?.scripts?.[w(node, "script")?.value];
  if (!t) return;
  node._lcVDSetting = true;
  try {
    setValue(node, "sample_text", t);
  } finally {
    node._lcVDSetting = false;
  }
  node.setDirtyCanvas?.(true, true);
}

function toCustom(node, name) {
  const wd = w(node, name);
  if (wd && wd.value !== "Custom") {
    wd.value = "Custom";
    node.setDirtyCanvas?.(true, true);
  }
}

function hook(node) {
  if (node._lcVDHooked) return;
  node._lcVDHooked = true;
  const wrap = (name, fn) => {
    const wd = w(node, name);
    if (!wd) return;
    const old = wd.callback;
    wd.callback = function (...a) {
      const r = old?.apply(this, a);
      if (!node._lcVDSetting) fn();
      return r;
    };
  };
  wrap("preset", () => loading.then(() => applyPreset(node)));
  wrap("script", () => loading.then(() => applyScript(node)));
  for (const k of PARTS) wrap(k, () => toCustom(node, "preset"));
  const box = w(node, "sample_text");
  const edited = () => {
    if (node._lcVDSetting) return;
    const named = DATA?.scripts?.[w(node, "script")?.value];
    if (named && box.value !== named) toCustom(node, "script");
  };
  wrap("sample_text", edited);
  box?.inputEl?.addEventListener("input", () => setTimeout(edited, 0)); // classic textarea: typing, not just blur
}

app.registerExtension({
  name: "LC.AV.VoiceDesign",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      hook(this);
      loading.then(() => {
        applyPreset(this);
        applyScript(this);
      });
      return r;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configure?.apply(this, arguments);
      hook(this);
      loading.then(() => {
        applyPreset(this); // a named preset / script follows its current values; Custom keeps what was saved
        applyScript(this);
      });
      return r;
    };
  },
});
