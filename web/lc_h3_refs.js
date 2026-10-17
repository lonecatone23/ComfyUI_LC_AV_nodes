/**
 * LC H3 Reference Assembler (BETA): a form with one row per wired picture: its <Picture n> number, role, name and a note. The rows live in the
 * hidden `references` STRING widget as JSON (one item per socket number, so unplugging a picture keeps its row).
 */
import { app } from "../../scripts/app.js";

const NODE = "LCH3ReferenceAssembler";
const SLOTS = 9;
const PREFIX = "image_";
const MIN_W = 460;
const ROLES = ["cast face", "cast full body", "setting", "prop", "style", "off"];
const ROLE_TIP = {
  "cast face": "A close-up of the person's face: the strongest way to keep a face",
  "cast full body": "The person head to toe: clothes and build (add a face close-up too)",
  setting: "A place: H3 takes the place and its light, not the people in it",
  prop: "An object that must look exactly like this (logo, colors, shape)",
  style: "Only the look: colors, light, grain",
  off: "Not sent to H3",
};
const HOLDER = { "cast face": "Name, e.g. Lonecat", "cast full body": "Same name as the face", setting: "e.g. the fishing pier", prop: "e.g. red cup", style: "", off: "" };

const CSS = `
.lcrf{font:12px Arial,sans-serif;color:#ddd;box-sizing:border-box;padding:6px 8px;display:flex;flex-direction:column;gap:5px}
.lcrf *{box-sizing:border-box}
.lcrf input,.lcrf select{background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:4px;padding:3px 5px;font:12px Arial,sans-serif;min-width:0}
.lcrf-row{display:flex;gap:4px;align-items:center}
.lcrf-n{font-weight:700;color:#eb4;white-space:nowrap;min-width:78px}
.lcrf-n.off{color:#666}
.lcrf-tip{font-size:11px;color:#999}
`;

function ensureStyle() {
  if (document.getElementById("lcrf-style")) return;
  const s = document.createElement("style");
  s.id = "lcrf-style";
  s.textContent = CSS;
  document.head.appendChild(s);
}

const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function getData(node) {
  try {
    const d = JSON.parse(widget(node, "references")?.value || "{}");
    return { items: Array.isArray(d.items) ? d.items : [] };
  } catch (_) {
    return { items: [] };
  }
}

function setData(node, data, rerender) {
  const w = widget(node, "references");
  if (w) w.value = JSON.stringify(data);
  if (rerender) render(node);
  fit(node);
  try {
    app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.();
  } catch (_) {}
}

function hideJson(node) {
  const w = widget(node, "references");
  if (!w || w._lcHidden) return;
  w._lcHidden = true;
  w.type = "hidden";
  w.computeSize = () => [0, -4];
  try {
    w.hidden = true;
  } catch (_) {}
  if (w.options) w.options.hidden = true;
}

// ------------------------------------------------------------------ which picture sockets are wired
// the sockets themselves grow natively (Autogrow in Python): never add, remove or reorder node.inputs here
const isLinked = (node, idx) => (typeof node.isInputConnected === "function" ? node.isInputConnected(idx) : node.inputs?.[idx]?.link != null);
const SLOT_RE = /(?:^|\.)image_(\d+)$/;

function wiredSlots(node) {
  const out = [];
  (node.inputs || []).forEach((inp, idx) => {
    const m = String(inp?.name || "").match(SLOT_RE);
    if (m && isLinked(node, idx)) out.push(parseInt(m[1], 10));
  });
  return out.sort((a, b) => a - b);
}

// ------------------------------------------------------------------ the form
function height(node) {
  const h = node.__lcrf?.inner.offsetHeight || 0;
  if (h > 0) node.__lcrfH = Math.ceil(h) + 8;
  return node.__lcrfH || 80;
}

function fit(node) {
  if (!node.__lcrf) return;
  requestAnimationFrame(() => {
    if (!node.__lcrf.inner.offsetHeight) return;
    const min = node.computeSize();
    if (Math.abs(node.size[1] - min[1]) > 0.5) node.setSize([Math.max(node.size[0], MIN_W), min[1]]);
    node.setDirtyCanvas?.(true, true);
  });
}

function ensureFace(node) {
  if (node.__lcrf) return node.__lcrf;
  ensureStyle();
  const wrap = document.createElement("div");
  const inner = document.createElement("div");
  inner.className = "lcrf";
  wrap.appendChild(inner);
  wrap.addEventListener("keydown", (e) => e.stopPropagation());
  const dom = node.addDOMWidget("lc_refs_face", "LC_REFS_FACE", wrap, { getMinHeight: () => height(node), serialize: false });
  dom.serializeValue = () => undefined;
  node.__lcrf = { wrap, inner };
  try {
    new ResizeObserver(() => fit(node)).observe(inner);
  } catch (_) {}
  return node.__lcrf;
}

function render(node) {
  const face = ensureFace(node);
  const d = getData(node);
  const slots = wiredSlots(node);
  let added = false;
  const item = (s) => {
    let it = d.items.find((x) => Number(x.slot) === s);
    if (!it) {
      added = true;
      it = { slot: s, role: d.items.some((x) => x.role === "setting") || s === 1 ? "cast face" : "setting", name: "", note: "" };
      d.items.push(it);
    }
    return it;
  };
  const names = [...new Set(d.items.map((x) => String(x.name || "").trim()).filter(Boolean))];
  let n = 0;
  const rows = slots.map((s) => {
    const it = item(s);
    const on = it.role !== "off";
    if (on) n += 1;
    return `<div class="lcrf-row">
      <span class="lcrf-n${on ? "" : " off"}" title="How H3 and the prompt refer to it (image_${s})">${on ? `&lt;Picture ${n}&gt;` : "off"}</span>
      <select data-s="${s}" data-f="role" title="${esc(ROLE_TIP[it.role] || "")}">${ROLES.map((r) => `<option value="${r}"${r === it.role ? " selected" : ""} title="${esc(ROLE_TIP[r])}">${r}</option>`).join("")}</select>
      <input data-s="${s}" data-f="name" list="lcrf-names-${node.id}" value="${esc(it.name)}" placeholder="${esc(HOLDER[it.role] || "")}" style="width:130px" ${it.role === "style" || !on ? "disabled" : ""}>
      <input data-s="${s}" data-f="note" value="${esc(it.note)}" placeholder="Note (optional), e.g. keep the logo readable" style="flex:1" ${on ? "" : "disabled"}>
    </div>`;
  });
  face.inner.innerHTML = (rows.length ? rows.join("") : `<div class="lcrf-tip">Connect reference pictures to image_1, image_2 ... (up to 9).</div>`)
    + `<datalist id="lcrf-names-${node.id}">${names.map((x) => `<option value="${esc(x)}">`).join("")}</datalist>`
    + (n > 9 ? `<div class="lcrf-tip" style="color:#e77">H3 reads 9 pictures at most.</div>` : "")
    + `<div class="lcrf-tip">Use the same name for a person's face and full body. Shots ask for references by these names.</div>`;
  const save = (re) => setData(node, d, re);
  if (added) save(false); // a newly wired picture's default role is what Python reads too
  face.wrap.querySelectorAll("[data-s]").forEach((el) => {
    const it = item(Number(el.dataset.s));
    const f = el.dataset.f;
    if (el.tagName === "SELECT") {
      el.onchange = () => {
        it[f] = el.value;
        save(true);
      };
    } else {
      el.oninput = () => {
        it[f] = el.value;
        save(false);
      };
      if (f === "name") el.onchange = () => save(true);
    }
  });
  fit(node);
}

app.registerExtension({
  name: "LC.AV.H3ReferenceAssembler",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      hideJson(this);
      // a fresh node starts with no example rows: they fill in as pictures are connected
      const w = widget(this, "references");
      if (w) w.value = JSON.stringify({ items: [] }); // a loaded workflow's saved rows replace this right after
      setTimeout(() => render(this), 0);
      if (!this.size || this.size[0] < MIN_W) this.setSize([MIN_W, this.computeSize()[1]]);
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      hideJson(this);
      // wires are restored after configure: draw the rows again once they are in
      setTimeout(() => render(this), 20);
      setTimeout(() => render(this), 400);
      return r;
    };
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange?.apply(this, arguments);
      setTimeout(() => render(this), 0);
      return r;
    };
  },
});
