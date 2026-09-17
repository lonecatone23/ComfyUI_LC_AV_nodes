/**
 * LC Load Audio — start_time / end_time cut + draggable timeline handles.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { lcApplyLaunchColor } from "./lc_color.js";

const NODE_CLASS = "LCLoadAudio";
const PREVIEW_COLOR = "#2a6a6a";
const ACCEPT = "audio/*,video/mp4,.mp3,.wav,.flac,.ogg,.m4a,.aac,.wma,.mp4,.webm,.mkv";

function viewUrl(filename, type, subfolder) {
  const q = new URLSearchParams({
    filename: filename || "",
    type: type || "input",
    subfolder: subfolder || "",
  });
  return api.apiURL(`/view?${q.toString()}`);
}

function widgetNamed(node, name) {
  return (node.widgets || []).find((x) => x.name === name);
}

function audioCombo(node) {
  return widgetNamed(node, "audio");
}

function numWidget(node, name, fallback) {
  const v = widgetNamed(node, name)?.value;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function setNum(node, name, value) {
  const w = widgetNamed(node, name);
  if (!w) return;
  const v = Math.max(0, Number(value) || 0);
  w.value = Math.round(v * 100) / 100;
  if (typeof w.callback === "function") {
    try {
      w.callback(w.value);
    } catch (_e) {}
  }
}

function fileDur(node) {
  return node._lcFileDur || 0;
}

function clipRange(node) {
  const total = fileDur(node);
  let start = Math.max(0, numWidget(node, "start_time", 0));
  let end = numWidget(node, "end_time", 0);
  const legacy = numWidget(node, "duration", 0);
  if (end <= 0 && legacy > 0) end = start + legacy;
  if (end <= 0) end = total > 0 ? total : start;
  if (total > 0) {
    start = Math.min(start, total);
    end = Math.min(end, total);
  }
  if (end < start) {
    const t = start;
    start = end;
    end = t;
  }
  return { start, end, total };
}

function fmt(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const s = Math.floor(sec % 60);
  const m = Math.floor(sec / 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function xToTime(track, ev, total) {
  const rect = track.getBoundingClientRect();
  const t = Math.min(1, Math.max(0, (ev.clientX - rect.left) / Math.max(1, rect.width)));
  return t * (total || 0);
}

function ensureBar(node) {
  if (node._lcBar) return node._lcBar;

  const wrap = document.createElement("div");
  wrap.style.cssText =
    "width:100%;max-width:100%;box-sizing:border-box;padding:4px 0 0 0;user-select:none;";

  const track = document.createElement("div");
  track.style.cssText =
    "position:relative;height:16px;border-radius:6px;background:#163333;overflow:visible;cursor:pointer;";

  const sel = document.createElement("div");
  sel.style.cssText =
    "position:absolute;top:0;bottom:0;left:0;width:0;background:#2a8a8a;opacity:0.55;border-radius:6px;pointer-events:none;";

  const fill = document.createElement("div");
  fill.style.cssText =
    "position:absolute;top:0;bottom:0;left:0;width:0;background:#6ec8c8;opacity:0.9;border-radius:6px;pointer-events:none;";

  function handle(color) {
    const h = document.createElement("div");
    h.style.cssText =
      `position:absolute;top:-3px;width:8px;height:22px;margin-left:-4px;background:${color};border-radius:2px;cursor:ew-resize;z-index:2;box-shadow:0 0 0 1px #042;`;
    return h;
  }
  const hStart = handle("#cfe");
  const hEnd = handle("#cfe");
  const playhead = document.createElement("div");
  playhead.style.cssText =
    "position:absolute;top:-2px;width:2px;height:20px;margin-left:-1px;background:#fff;z-index:3;pointer-events:none;";

  track.appendChild(sel);
  track.appendChild(fill);
  track.appendChild(hStart);
  track.appendChild(hEnd);
  track.appendChild(playhead);

  const label = document.createElement("div");
  label.style.cssText =
    "font-size:10px;line-height:14px;color:#cfe;opacity:0.9;margin-top:3px;font-family:sans-serif;";
  label.textContent = "0:00 – 0:00";

  wrap.appendChild(track);
  wrap.appendChild(label);

  const widget = node.addDOMWidget("lc_audio_bar", "bar", wrap, {
    serialize: false,
    hideOnZoom: false,
    getValue() {
      return "";
    },
    setValue() {},
  });
  widget.computeSize = function (width) {
    const w = Math.max(180, width || node.size?.[0] || 270);
    return [w, 36];
  };

  function bindDrag(el, which) {
    el.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      el.setPointerCapture(ev.pointerId);
      const move = (e) => {
        const { total } = clipRange(node);
        if (!(total > 0)) return;
        const t = xToTime(track, e, total);
        if (which === "start") {
          const end = numWidget(node, "end_time", total);
          setNum(node, "start_time", Math.min(t, end > 0 ? end : total));
        } else {
          const start = numWidget(node, "start_time", 0);
          setNum(node, "end_time", Math.max(t, start));
        }
        paintBar(node);
      };
      const up = () => {
        el.releasePointerCapture(ev.pointerId);
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
    });
  }
  bindDrag(hStart, "start");
  bindDrag(hEnd, "end");

  track.addEventListener("pointerdown", (ev) => {
    if (ev.target === hStart || ev.target === hEnd) return;
    const { start, end, total } = clipRange(node);
    if (!(total > 0)) return;
    const t = xToTime(track, ev, total);
    if (node._lcPlayer) {
      node._lcPlayer.currentTime = Math.min(end, Math.max(start, t));
      paintBar(node);
    }
  });

  node._lcBar = { wrap, track, sel, fill, hStart, hEnd, playhead, label, widget };
  return node._lcBar;
}

function paintBar(node) {
  const bar = node._lcBar;
  if (!bar) return;
  const { start, end, total } = clipRange(node);
  const span = total > 0 ? total : 1;
  const left = (start / span) * 100;
  const right = (end / span) * 100;
  bar.sel.style.left = `${left}%`;
  bar.sel.style.width = `${Math.max(0, right - left)}%`;
  bar.hStart.style.left = `${left}%`;
  bar.hEnd.style.left = `${right}%`;
  const player = node._lcPlayer;
  const now = player ? player.currentTime : start;
  const t = Math.min(1, Math.max(0, now / span));
  bar.playhead.style.left = `${t * 100}%`;
  const shown = Math.max(0, now - start);
  bar.fill.style.left = `${left}%`;
  bar.fill.style.width = `${Math.max(0, Math.min(right, t * 100) - left)}%`;
  bar.label.textContent = `${fmt(start)} – ${fmt(end)}${total ? `  (${fmt(total)})` : ""}  ${fmt(shown)}`;
}

function setPlaying(node, playing) {
  node._lcPlaying = !!playing;
  paintBar(node);
  node.setDirtyCanvas?.(true, true);
}

function stopPlayer(node) {
  if (node._lcRaf) {
    cancelAnimationFrame(node._lcRaf);
    node._lcRaf = 0;
  }
  if (node._lcPlayer) {
    try {
      node._lcPlayer.pause();
      node._lcPlayer.src = "";
    } catch (_e) {}
    node._lcPlayer = null;
  }
  setPlaying(node, false);
}

function tick(node) {
  const player = node._lcPlayer;
  if (!player) return;
  const { start, end } = clipRange(node);
  if (player.currentTime >= end - 0.02) {
    player.pause();
    player.currentTime = start;
    setPlaying(node, false);
    paintBar(node);
    return;
  }
  paintBar(node);
  if (!player.paused && !player.ended) {
    node._lcRaf = requestAnimationFrame(() => tick(node));
  }
}

function playSelected(node) {
  const file = audioCombo(node)?.value;
  if (!file) {
    console.warn("[LC AV] pick an audio file first.");
    return;
  }
  ensureBar(node);
  if (node._lcPlayer) {
    try {
      node._lcPlayer.pause();
    } catch (_e) {}
    node._lcPlayer = null;
  }
  const url = viewUrl(String(file).replace(/\\/g, "/"), "input", "");
  const audio = new Audio(url);
  audio.preload = "auto";
  audio.volume = 1;
  node._lcPlayer = audio;

  const applyCut = () => {
    if (Number.isFinite(audio.duration) && audio.duration > 0) {
      node._lcFileDur = audio.duration;
    }
    const { start } = clipRange(node);
    try {
      audio.currentTime = start;
    } catch (_e) {}
    paintBar(node);
  };

  audio.addEventListener("loadedmetadata", applyCut);
  audio.addEventListener("ended", () => {
    const { start } = clipRange(node);
    try {
      audio.currentTime = start;
    } catch (_e) {}
    setPlaying(node, false);
    paintBar(node);
  });

  applyCut();
  audio.play()?.catch((e) => {
    console.warn("[LC AV] preview play failed", e, url);
    setPlaying(node, false);
  });
  setPlaying(node, true);
  node._lcRaf = requestAnimationFrame(() => tick(node));
}

function togglePreview(node) {
  ensureBar(node);
  const player = node._lcPlayer;
  if (player && !player.paused && !player.ended) {
    player.pause();
    setPlaying(node, false);
    return;
  }
  if (player && player.paused && player.src) {
    const { start, end } = clipRange(node);
    if (player.currentTime < start || player.currentTime >= end) {
      player.currentTime = start;
    }
    player.play()?.catch(() => {});
    setPlaying(node, true);
    node._lcRaf = requestAnimationFrame(() => tick(node));
    return;
  }
  playSelected(node);
}

function hookCutWidgets(node) {
  for (const name of ["start_time", "end_time", "duration", "audio"]) {
    const w = widgetNamed(node, name);
    if (!w || w._lcCutHook) continue;
    w._lcCutHook = true;
    const prev = w.callback;
    w.callback = (...args) => {
      paintBar(node);
      return prev?.apply(w, args);
    };
  }
}

function addToCombo(node, filename) {
  const w = audioCombo(node);
  if (!w) return;
  w.options = w.options || {};
  const values = Array.isArray(w.options.values) ? w.options.values.slice() : [];
  if (filename && !values.includes(filename)) values.push(filename);
  w.options.values = values;
  w.value = filename;
  node.setDirtyCanvas?.(true, true);
}

async function uploadAndSelect(node) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ACCEPT;
  input.style.display = "none";
  document.body.appendChild(input);
  input.addEventListener("change", async () => {
    const file = input.files && input.files[0];
    input.remove();
    if (!file) return;
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("overwrite", "true");
      form.append("type", "input");
      const resp = await api.fetchApi("/upload/image", { method: "POST", body: form });
      if (!resp.ok) throw new Error(`upload ${resp.status}`);
      const data = await resp.json();
      addToCombo(node, data.name || data.filename || file.name);
      stopPlayer(node);
    } catch (e) {
      console.warn("[LC AV] Load audio upload failed", e);
    }
  });
  input.click();
}

app.registerExtension({
  name: "LC.AV.AudioPreview",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if ((nodeData?.name || "") !== NODE_CLASS) return;

    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origCreated?.apply(this, arguments);
      lcApplyLaunchColor(this, PREVIEW_COLOR);
      hookCutWidgets(this);
      ensureBar(this);

      const names = (this.widgets || []).map((w) => w.name);
      if (!names.includes("Load audio") && !names.includes("refresh files")) {
        this.addWidget("button", "Load audio", null, () => uploadAndSelect(this));
      } else {
        const old = widgetNamed(this, "refresh files") || widgetNamed(this, "Load audio");
        if (old) {
          old.name = "Load audio";
          old.label = "Load audio";
          old.callback = () => uploadAndSelect(this);
        }
      }
      if (!names.includes("preview / pause")) {
        this.addWidget("button", "preview / pause", null, () => togglePreview(this));
      } else {
        const old = widgetNamed(this, "preview / pause");
        if (old) old.callback = () => togglePreview(this);
      }
      return r;
    };
  },
});

console.log("[LC.AV.AudioPreview] loaded");
