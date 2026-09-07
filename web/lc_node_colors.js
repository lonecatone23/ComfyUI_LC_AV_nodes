/**
 * Default node colors for LC Audio_Video Nodes.
 *
 * LC123 image pack:
 *   utility / sampling  #324b4b
 *   sigma / schedule    #1c6d6d
 *   io / logic          #28281E
 *
 * AV family (related teal, distinct from sigma #1c6d6d):
 *   video / pipe        #1a5a7a
 *   audio               #2a6a6a
 *   save video          #1a5a7a  (same as Create Video)
 *   save metadata       #28281E  (same io chrome as LC123 save nodes)
 *
 * Do not copy every LC123 color. computeSize is not set here — size
 * lives on each node’s own JS if added later (MINIMUM only).
 */
import { app } from "../../scripts/app.js";

const COLORS = {
    LCAvPipeOut: { color: "#1a5a7a", bgcolor: "#1a5a7a" },
    LCAvPipeEdit: { color: "#1a5a7a", bgcolor: "#1a5a7a" },
    LCCreateVideo: { color: "#1a5a7a", bgcolor: "#1a5a7a" },
    LCVaeDecode: { color: "#322", bgcolor: "#533" },
    LCGetLatentSize: { color: "#324b4b", bgcolor: "#324b4b" },
    LCFinalFrame: { color: "#1a5a7a", bgcolor: "#1a5a7a" },
    LCSaveVideo: { color: "#1a5a7a", bgcolor: "#1a5a7a" },
    LCPipeToAvPipe: { color: "#1a5a7a", bgcolor: "#1a5a7a" },
    LCSaveVideoMetadata: { color: "#28281E", bgcolor: "#28281E" },
    LCLoadAudio: { color: "#2a6a6a", bgcolor: "#2a6a6a" },
    LCAudioDuration: { color: "#2a6a6a", bgcolor: "#2a6a6a" },
    LCAudioCrop: { color: "#2a6a6a", bgcolor: "#2a6a6a" },
    LCAudioChannel: { color: "#2a6a6a", bgcolor: "#2a6a6a" },
    LCAudioSeparate: { color: "#2a6a6a", bgcolor: "#2a6a6a" },
    LCAudioEqualizer3Band: { color: "#2a6a6a", bgcolor: "#2a6a6a" },
    LCAudioVolume: { color: "#2a6a6a", bgcolor: "#2a6a6a" },
};

app.registerExtension({
    name: "LC.AV.NodeColors",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cfg = COLORS[nodeData.name];
        if (!cfg) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            this.color = cfg.color;
            this.bgcolor = cfg.bgcolor;
            if (cfg.size) this.size = cfg.size.slice();
        };
    },
});
