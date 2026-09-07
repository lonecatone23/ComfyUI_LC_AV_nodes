/**
 * LC Save Video preview — native controls (volume + fullscreen),
 * frame size follows node width × video aspect.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "LCSaveVideo";
const CHROME = 40;

function viewUrl(info) {
  const q = new URLSearchParams({
    filename: info.filename || "",
    type: info.type || "output",
    subfolder: info.subfolder || "",
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

function layout(node) {
  const pack = node._lcVidPreview;
  if (!pack) return;
  const { vid, wrap } = pack;
  wrap.style.width = "100%";
  wrap.style.maxWidth = "100%";
  wrap.style.overflow = "hidden";
  wrap.style.boxSizing = "border-box";
  if (vid.src && wrap.style.display !== "none") {
    const h = frameH(node, vid);
    vid.style.width = "100%";
    vid.style.maxWidth = "100%";
    vid.style.height = `${h}px`;
    vid.style.objectFit = "contain";
    vid.style.display = "block";
  }
  node.setDirtyCanvas?.(true, true);
}

function ensurePreview(node) {
  if (node._lcVidPreview) return node._lcVidPreview;

  const wrap = document.createElement("div");
  wrap.style.cssText =
    "width:100%;max-width:100%;overflow:hidden;display:none;box-sizing:border-box;line-height:0;";

  const vid = document.createElement("video");
  vid.controls = true;
  vid.controlsList = "nodownload";
  vid.loop = true;
  vid.muted = false;
  vid.autoplay = true;
  vid.playsInline = true;
  vid.preload = "metadata";
  vid.style.cssText =
    "width:100%;max-width:100%;display:block;object-fit:contain;border-radius:4px;background:#111;box-sizing:border-box;";
  wrap.appendChild(vid);

  const widget = node.addDOMWidget("lc_video_preview", "preview", wrap, {
    serialize: false,
    hideOnZoom: false,
    getValue() {
      return "";
    },
    setValue() {},
  });

  widget.computeSize = function () {
    const w = nodeWidth(node);
    if (!vid.src || wrap.style.display === "none") return [w, 0];
    return [w, frameH(node, vid) + CHROME];
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
  vid.src = viewUrl(info);
  wrap.style.display = "block";
  vid.play()?.catch(() => {});
  layout(node);
}

app.registerExtension({
  name: "LC.AV.SaveVideoPreview",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if ((nodeData?.name || "") !== NODE_CLASS) return;

    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExecuted?.apply(this, arguments);
      playMessage(this, message);
    };

    const origResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      origResize?.apply(this, arguments);
      layout(this);
    };
  },
});

console.log("[LC.AV.SaveVideoPreview] loaded");
