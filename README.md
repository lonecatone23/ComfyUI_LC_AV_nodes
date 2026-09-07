# ComfyUI LC Audio_Video Nodes

Custom nodes for ComfyUI by lonecatone23.

* **Repo:** https://github.com/lonecatone23/ComfyUI_LC_AV_nodes
* **Image companion:** https://github.com/lonecatone23/ComfyUI_LC123_nodes
* **Civitai:** lonecatone23
* **Support:** https://ko-fi.com/lonecatone
* **Version:** 0.3.1

> Small tools that remove friction — frames, soundtrack, metadata, fewer extra packs.

This pack is **not** a fork of LC123. Do not merge the two folders.
`LC_PIPE` stays on the image pack. This pack uses **`LC_AV_PIPE`**.

Release history lives in **git tags**. This page describes the pack as it is now.

---

## Install

```
ComfyUI/custom_nodes/ComfyUI_LC_AV_nodes/__init__.py
```

`__init__.py` must sit **directly** in that folder — not in `ComfyUI_LC_AV_nodes/ComfyUI_LC_AV_nodes/`.
If you unzip a pack zip *inside* an existing clone, move the inner files up one level.

Restart ComfyUI. Console should print `[LC AV] total … nodes`.
Hard-refresh the browser after a `web/` JS update.

**Requirements:** ComfyUI’s Python (`torch`, `numpy`). Save Video needs **ffmpeg** on PATH.
Audio Separate needs **torchaudio** (no Numba / SoX).

Optional example graph: `workflows/LC_AV Node Examples.json`  
(BETA — examples only. Add nodes from the node menu; do not treat the graph as a production template.)

---

## What this pack is

| Title | Class | Role |
| --- | --- | --- |
| LC AV Pipe (in/edit) 🎬 | `LCAvPipeEdit` | Pack or edit `LC_AV_PIPE`. In: `pipe`, `av_pipe`, then video / audio / fps / … Out: `av_pipe`. |
| LC AV Pipe Out 🎬 | `LCAvPipeOut` | Unpack. In: `pipe` + `av_pipe`. No `LC_PIPE` on the output side. |
| LC Pipe → AV Pipe 🔌 | `LCPipeToAvPipe` | One `LC_PIPE` in, one `LC_AV_PIPE` out. Shared keys only. |
| LC Create Video 🎬 | `LCCreateVideo` | IMAGE batch + fps (+ optional AUDIO) → `LC_AV_PIPE`. Does not encode a file. |
| LC Save Video 💾 | `LCSaveVideo` | ffmpeg encode. Inputs: images, audio, av_pipe, metadata. No `LC_PIPE` socket. |
| LC Save Video Metadata 🏷️ | `LCSaveVideoMetadata` | `LC_SAVE_META` + A1111 `parameters`. Works with Save Video or LC123 Save Image. |
| LC Final Frame 🖼️ | `LCFinalFrame` | Last frame of an IMAGE batch. |
| LC VAE Decode 🧩 | `LCVaeDecode` | Boolean wrapper: core `VAEDecode` or core `VAEDecodeTiled`. |
| LC Get Latent Size 📐 | `LCGetLatentSize` | Pixel width / height, batch, frames. Optional VAE for compression. |
| LC Load Audio 📂 | `LCLoadAudio` | Load from input. `start_time` / `end_time` cut, preview / pause, timeline handles. |
| LC Audio Duration ⏱️ | `LCAudioDuration` | Seconds int + float (and minutes). |
| LC Audio Crop ✂️ | `LCAudioCrop` | Crop AUDIO with `MM:SS` start / end. |
| LC Audio Channel 🎚️ | `LCAudioChannel` | Stereo → L / R. |
| LC Audio Separate 🎼 | `LCAudioSeparate` | Demucs stems: Bass, Drums, Other, Vocals. |
| LC Audio EQ 3-Band 🎛️ | `LCAudioEqualizer3Band` | Low shelf, mid peak, high shelf. |
| LC Audio Volume 🔊 | `LCAudioVolume` | Gain 0–4, soft clip. |

---

## LC_AV_PIPE 🎬

Type string: **`LC_AV_PIPE`**.

Edit node sockets, top → bottom:

* `pipe` (`LC_PIPE`)
* `av_pipe` (`LC_AV_PIPE`)
* video, audio, frame_rate, duration, sample_rate, width, height, frame_count
* positive / negative, seed, total_steps, sampler_name, models

Wired slots override; empty slots pass through.

Do **not** put video into `LC_PIPE`. Convert shared fields with **LC Pipe → AV Pipe**.

No **civitai_air** on the pipe — paste AIR on **Save Video Metadata**.

---

## Save Video 💾

VHS-style options: frame_rate, loop_count, filename_prefix, format, pix_fmt, crf, pingpong, save_output, prune_outputs.

`prune_outputs` keeps intermediate frame PNGs off disk.

`save_metadata` writes sidecar JSON + container comment. **Civitai** reads PNG chunks, not mp4 comment — keep `civitai_still` on and upload the first-frame PNG (fill `models` + `civitai_air` on the metadata node).

On-node preview after encode. Resize the node; the player follows aspect.

Wire **IMAGE** from VAE Decode (or Create Video `images`) into `images`. Metadata is optional. `av_pipe` can carry frames if they were packed there.

---

## LC VAE Decode 🧩

`tiled` off = `nodes.VAEDecode`.  
`tiled` on = `nodes.VAEDecodeTiled` (tile_size, overlap, temporal_size, temporal_overlap).

Same latent + same VAE as the two core nodes. Use the MiniMax **Video** VAE for video latents.

---

## Load Audio 📂

**Load audio** uploads into Comfy `input/` like stock Load Audio.

**start_time** / **end_time** (seconds). `end_time` 0 = end of file. Output `duration` is the cut length.

**preview / pause** plays only that cut. Drag the two handles on the timeline (after one preview so file length is known).

---

## Colors

`web/lc_node_colors.js` — AV family teal (`#2a6a6a` audio, `#1a5a7a` video). Distinct from LC123 sigma `#1c6d6d`. VAE Decode uses stock Comfy red.

`web/lc_color.js` is the launch-color helper (does not overwrite a saved color).

---

## License

MIT. See `LICENSE`.
