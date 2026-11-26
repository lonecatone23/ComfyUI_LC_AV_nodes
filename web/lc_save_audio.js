/**
 * LC Save Audio: the same audio player ComfyUI puts on its own Save Audio nodes (the frontend adds it for an
 * AUDIO_UI input; it is added here in the browser only, so the saved prompt never needs a value for it).
 */
import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "LC.AV.SaveAudio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LCSaveAudio") return;
    const input = (nodeData.input ??= {});
    const req = (input.required ??= {});
    req.audioUI = ["AUDIO_UI", {}];
  },
});
