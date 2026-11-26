/**
 * LC Load Voice:
 *  - ComfyUI's upload button and audio player for the clip picker (the frontend adds the upload button for an "audio"
 *    picker marked audio_upload, and it needs the AUDIO_UI player next to it). Both live in the browser only, so the
 *    saved prompt never needs a value for them. The player sits before the upload button: the button looks it up.
 *  - Picking (or uploading) a file fills voice_name with the file's name: Haley.wav -> Haley. A name you typed stays
 *    until you pick a different file. Loading a saved workflow never renames anything.
 */
import { app } from "../../scripts/app.js";

const stem = (v) => String(v || "").replace(/\\/g, "/").split("/").pop().replace(/\.[^.]+$/, "").trim();
const W = (node, name) => (node.widgets || []).find((w) => w.name === name);

function hook(node) {
  const pick = W(node, "audio");
  const nameW = W(node, "voice_name");
  if (!pick || !nameW || node._lcVoiceHooked) return;
  node._lcVoiceHooked = true;
  let last = pick.value;
  const prev = pick.callback;
  pick.callback = function (v, ...rest) {
    const r = prev?.apply(this, [v, ...rest]);
    if (pick.value !== last) {
      last = pick.value;
      const s = stem(pick.value);
      if (s && !s.startsWith("(")) nameW.value = s;
      node.setDirtyCanvas?.(true, true);
    }
    return r;
  };
  // an upload sets the picker's value without its callback: watch for it
  node._lcVoicePoll = setInterval(() => {
    if (!node.graph) return clearInterval(node._lcVoicePoll);
    if (pick.value !== last) pick.callback?.(pick.value);
  }, 500);
}

app.registerExtension({
  name: "LC.AV.LoadVoice",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LCLoadVoice") return;
    const input = (nodeData.input ??= {});
    const req = (input.required ??= {});
    const upload = req.upload; // added by Comfy.UploadAudio when it ran first
    delete req.upload;
    req.audioUI = ["AUDIO_UI", {}];
    if (upload) req.upload = upload;

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      // a fresh node: the name follows the first file in the list
      const pick = W(this, "audio"), nameW = W(this, "voice_name");
      if (pick && nameW && !this._lcConfigured) {
        const s = stem(pick.value);
        if (s && !s.startsWith("(")) nameW.value = s;
      }
      hook(this);
      return r;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      this._lcConfigured = true; // a saved node keeps its saved name
      const r = configure?.apply(this, arguments);
      if (this._lcVoiceHooked) {
        const pick = W(this, "audio");
        this._lcVoiceHooked = false;
        clearInterval(this._lcVoicePoll);
        if (pick) hook(this);
      }
      return r;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      clearInterval(this._lcVoicePoll);
      return removed?.apply(this, arguments);
    };
  },
});
