/**
 * LC Save Video preview
 * Width: never touched (full node width).
 * Unpinned height: hug the frame after each encode.
 * Pinned height: floor is the node height at the moment you pin.
 * May grow past the floor for a taller frame; shrinks only down to
 * that floor. To redefine the floor, unpin, resize, and re-pin —
 * onResize is NOT used to update the floor (see note below).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// video sound: off until you turn it on, then remembered (in this browser) and shared by every LC video player.
// Muting one mutes them all, and a player that loads a new video (the next segment) picks up the current setting.
function lcSound(v) {
  const KEY = "lc.video.sound", VOL = "lc.video.volume";
  const players = (window.__lcVideoPlayers ||= new Set());
  let on = false, vol = 1;
  try {
    on = localStorage.getItem(KEY) === "on";
    vol = Math.min(1, Math.max(0, parseFloat(localStorage.getItem(VOL) ?? "1")));
  } catch (_e) {}
  v.__lcSoundSync = true; // a change made here, not by the viewer: don't spread it
  v.muted = !on;
  if (Number.isFinite(vol)) v.volume = vol;
  setTimeout(() => (v.__lcSoundSync = false), 0); // volumechange arrives after this returns
  players.add(v);
  if (!v.__lcSound) {
    v.__lcSound = true;
    v.addEventListener("volumechange", () => {
      if (v.__lcSoundSync) return;
      try {
        localStorage.setItem(KEY, v.muted ? "off" : "on");
        localStorage.setItem(VOL, String(v.volume));
      } catch (_e) {}
      for (const o of players) {
        if (!o.isConnected) {
          players.delete(o);
          continue;
        }
        if (o !== v && (o.muted !== v.muted || o.volume !== v.volume)) {
          o.__lcSoundSync = true;
          o.muted = v.muted;
          o.volume = v.volume;
          setTimeout(() => (o.__lcSoundSync = false), 0);
        }
      }
    });
  }
  return v;
}

const NODE_CLASSES = new Set(["LCSaveVideo", "LCPreviewVideo"]);
const PAD = 10;

function viewUrl(info) {
  const q = new URLSearchParams({
    filename: info.filename || "",
    type: info.type || "output",
    subfolder: info.subfolder || "",
    t: String(Date.now()), // the joined movie keeps its name and grows: without this the browser plays its old copy
  });
  return api.apiURL(`/view?${q.toString()}`);
}

function nodeWidth(node) {
  return Math.max(180, (node.size && node.size[0]) || 270);
}

function frameH(node, vid) {
  const w = nodeWidth(node);
  if (vid.videoWidth > 0 && vid.videoHeight > 0) {
    return Math.round((w * vid.videoHeight) / vid.videoWidth);
  }
  return 120;
}

function isPinned(node) {
  return !!(node.flags && node.flags.pinned);
}

function setFloor(node, h) {
  node.properties = node.properties || {};
  const v = Math.round(Number(h) || 0);
  if (v > 0) node.properties.lc_pin_h = v;
}

function clearFloor(node) {
  if (node.properties && node.properties.lc_pin_h != null) {
    delete node.properties.lc_pin_h;
  }
}

function readFloor(node) {
  const v = Number(node.properties?.lc_pin_h);
  return Number.isFinite(v) && v > 0 ? v : null;
}

/** Pin edge: floor becomes the height at the instant of pin. */
function syncPinState(node) {
  const pinned = isPinned(node);
  if (pinned && !node._lcWasPinned) {
    setFloor(node, node.size?.[1]);
  } else if (!pinned && node._lcWasPinned) {
    clearFloor(node);
  } else if (!pinned) {
    clearFloor(node);
  }
  node._lcWasPinned = pinned;
}

function contentHeight(node) {
  try {
    const sz = node.computeSize(node.size?.[0] || nodeWidth(node));
    if (Array.isArray(sz) && sz[1] > 0) return sz[1];
  } catch (_e) {}
  return node.size?.[1] || 80;
}

function applyHeight(node) {
  if (!node.size) return;
  syncPinState(node);
  const floor = isPinned(node) ? readFloor(node) : null;
  const content = contentHeight(node);
  node._lcApplyingH = true;
  if (floor != null) {
    node.size[1] = Math.max(floor, content);
  } else {
    node.size[1] = content;
  }
  node._lcAppliedH = node.size[1];
  node._lcApplyingH = false;
}

function layout(node) {
  const pack = node._lcVidPreview;
  if (!pack) return;
  const { vid, wrap } = pack;
  wrap.style.width = "100%";
  wrap.style.maxWidth = "100%";
  wrap.style.overflow = "hidden";
  wrap.style.boxSizing = "border-box";
  wrap.style.margin = "0";
  wrap.style.padding = `0 0 ${PAD}px 0`;
  if (vid.src && wrap.style.display !== "none") {
    const h = frameH(node, vid);
    vid.style.width = "100%";
    vid.style.maxWidth = "100%";
    vid.style.height = `${h}px`;
    vid.style.objectFit = "contain";
    vid.style.display = "block";
    vid.style.margin = "0";
  }
  applyHeight(node);
  node.setDirtyCanvas?.(true, true);
}

function ensurePreview(node) {
  if (node._lcVidPreview) return node._lcVidPreview;

  const wrap = document.createElement("div");
  wrap.style.cssText =
    `width:100%;max-width:100%;overflow:hidden;display:none;box-sizing:border-box;line-height:0;margin:0;padding:0 0 ${PAD}px 0;`;

  const vid = document.createElement("video");
  vid.controls = true;
  vid.controlsList = "nodownload";
  vid.loop = true;
  lcSound(vid);
  vid.autoplay = false; // a new video (the next segment) loads paused: press play when you want to watch
  vid.playsInline = true;
  vid.preload = "metadata";
  vid.style.cssText =
    "width:100%;max-width:100%;display:block;object-fit:contain;border-radius:4px;background:#111;box-sizing:border-box;margin:0;";
  wrap.appendChild(vid);

  const widget = node.addDOMWidget("lc_video_preview", "preview", wrap, {
    serialize: false,
    hideOnZoom: false,
    getValue() {
      return "";
    },
    setValue() {},
  });

  // The right side Parameters panel writes its own column width onto widgets it lists; on this one that left the
  // video wider or narrower than the node after selecting it. Always follow the node's width instead.
  Object.defineProperty(widget, "width", { get: () => undefined, set: () => {}, configurable: true });

  widget.computeSize = function () {
    const w = nodeWidth(node);
    if (!vid.src || wrap.style.display === "none") return [w, 0];
    return [w, frameH(node, vid) + PAD];
  };

  vid.addEventListener("loadedmetadata", () => {
    wrap.style.display = "block";
    layout(node);
  });

  node._lcVidPreview = { wrap, vid, widget };
  return node._lcVidPreview;
}

function playMessage(node, message) {
  const info = message?.video?.[0] || message?.gifs?.[0];
  if (!info || !info.filename) return;
  const { vid, wrap } = ensurePreview(node);
  lcSound(vid); // the current sound setting, not the one this player started with
  vid.src = viewUrl(info);
  wrap.style.display = "block";
  layout(node);
}

app.registerExtension({
  name: "LC.AV.SaveVideoPreview",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_CLASSES.has(nodeData?.name || "")) return;

    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExecuted?.apply(this, arguments);
      playMessage(this, message);
    };

    // NOTE: onResize does NOT update the pin floor. onResize fires for
    // reasons that have nothing to do with the user dragging the corner —
    // most notably, the preview widget's own computeSize legitimately
    // changes (e.g. empty [w,0] -> [w, frameH+PAD] once a new video's
    // metadata loads), and LiteGraph relayouts + calls onResize with that
    // size. Trusting onResize here means that ordinary event gets treated
    // as "the user just redefined the floor," silently overwriting it.
    // Confirmed by direct reproduction: pin at a given height, then feed
    // onResize an unrelated size — the floor immediately corrupted to
    // that value under both Comfy.VueNodes.Enabled true and false. The
    // floor is only ever set at the pin()/onConfigure/onDrawForeground
    // pin-transition edges below. Consequence: dragging the node while
    // it's already pinned no longer redefines the floor — unpin, resize,
    // and re-pin to do that.
    const origResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      origResize?.apply(this, arguments);
      if (!this._lcApplyingH) layout(this);
    };

    const origPin = nodeType.prototype.pin;
    nodeType.prototype.pin = function (value) {
      origPin?.apply(this, arguments);
      const pinned = isPinned(this);
      if (pinned) {
        setFloor(this, this.size?.[1]);
        this._lcWasPinned = true;
      } else {
        clearFloor(this);
        this._lcWasPinned = false;
      }
    };

    const origCfg = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      origCfg?.apply(this, arguments);
      this._lcWasPinned = isPinned(this);
      if (!isPinned(this)) {
        clearFloor(this);
      } else if (readFloor(this) == null) {
        setFloor(this, this.size?.[1]);
      }
    };

    const origDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
      origDraw?.apply(this, arguments);
      const pinned = isPinned(this);
      if (pinned !== this._lcWasPinned) {
        if (pinned) setFloor(this, this.size?.[1]);
        else clearFloor(this);
        this._lcWasPinned = pinned;
      }
    };
  },
});

console.log("[LC.AV.SaveVideoPreview] loaded");
