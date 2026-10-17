# ComfyUI LC Audio_Video Nodes

Custom nodes for ComfyUI by lonecatone23.

* **Repo:** https://github.com/lonecatone23/ComfyUI_LC_AV_nodes
* **Image companion:** https://github.com/lonecatone23/ComfyUI_LC123_nodes
* **Civitai:** lonecatone23
* **Support:** https://ko-fi.com/lonecatone
* **Version:** 0.3.13

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

Shows a progress bar per tile (also on the MiniMax Video VAE, which tiles itself), and Cancel stops it between tiles.

---

## Load Audio 📂

**Load audio** uploads into Comfy `input/` like stock Load Audio.

**start_time** / **end_time** (seconds). `end_time` 0 = end of file. Output `duration` is the cut length.

**preview / pause** plays only that cut. Drag the two handles on the timeline (after one preview so file length is known).

---

## H3 Long Video 🎬

Minute-long MiniMax H3 videos made in short pieces, one segment per Run, on the same card that makes 5 s clips. The plan comes from **LC Vision Director 🎬** (LC Vision). Example workflow: `LC H3 Long Video Example.json`.

**LC H3 Segment 🎞️**
- Builds one segment for your sampler: positive + latent out, seed passed through.
- **index** 0 = the next segment not saved yet. Set a number to redo that one.
- A CONTINUE segment starts from the last 1.6 s (picture and sound) of the one before, so the motion carries straight on.
- **memory** `clip` (default): about 1 s of segment 1 (her head only) goes in as a reference on a CUT your story writes, and as a gentle nudge on any CONTINUE that starts with no clear face (she walked away or turned round). **memory_gap** 5 s keeps the cut from opening on that clip's pose.
- **color_match** 0 = off. Turn it on (0.2 to 0.4) when Assemble flags a scene drifting warmer or cooler.
- 💡 960 x 544 is the low VRAM size. Keep one size for the whole project.
- **prompt** output: the segment's prompt, quotes and all. Wire it into LC H3 Dialogue Check to check each segment's speech.

**LC H3 Segment Save 💾**
- Decodes and saves the segment to `output/LC_H3/<project>/`: a preview mp4, the part for the final video and the ending the next segment starts from.
- Shows a **face match** to segment 1 (0 to 1): 0.55 and up = same person, under 0.40 = someone else, redo it with another seed. Other people in the shot and faces turned away are left out.
- Face check models: `face_detection_yunet_2023mar.onnx` + `face_recognition_sface_2021dec.onnx` in `models/facerestore` (OpenCV zoo). Without them it just skips the check.

**LC H3 Assemble 🧩**
- Joins every saved segment into `<project>_full.mp4`, straight from disk (never the whole video in RAM).
- **blend_frames** crossfades each CONTINUE join over the shared frames so it doesn't pop.
- Reports how smooth every join is, how far the color drifts inside each scene, and every face match. Wire Segment Save's **info** into **after** so it runs after the new segment is saved.

**LC H3 Audio Sync (BETA) 🎤**
- Makes an H3 video perform YOUR track (a song, a line of dialogue) instead of sound the model invents. Put it between the latent and the sampler.
- Your recording is locked into the sound half, so the model only decides the picture that fits it, mouth included.
- 💡 Wire the H3 **positive** conditioning through it too: the track is also handed to H3 as a clean guide.
- ✋ **Put each line in the prompt with its time:** `At 0 seconds she says, "Hello there."` H3 moves the mouth when the prompt says the words are spoken, not when it hears them. Without the time she talks over the silence instead. LC H3 Dialogue Check's **prompt_line** writes these lines for you from your recording.
- **start_seconds** picks where in the track the clip starts. A track shorter than the clip is filled with **silence** or **loop**ed.
- The **audio** output is the track cut to the clip's length: wire it to the save node so picture and sound end together.
- ⚠️ H3 was trained to about 15 s; longer clips get a warning in **info**.

**LC Frame Interpolate 🎞️**
- Smooth motion with RIFE 4.26: makes the in-between frames so a 24 fps clip plays at 48, 60, or any rate (not just 2x). Each new frame lands at its real moment in time.
- **scene_cut** holds hard cuts instead of morphing across them (a morph across a cut is a smeared ghost frame). 0 = off.
- Wire the clip's **audio** and the output is exactly as long as the sound, so picture and audio end together.
- **model**: rife426.pth (best on AI video in our tests, downloads once, 21 MB) or **crossfade** (no model, fast, soft on motion). Uses `models/rife/` or ComfyUI-Frame-Interpolation's checkpoints if you have them.
- RIFE code from ComfyUI-Frame-Interpolation (Fannovel16) and Practical-RIFE (hzwer), both MIT.

**LC H3 Dialogue Check (BETA) 🗣️**
- Listens to a finished clip with Whisper and compares what was said with the dialogue in quotes in your prompt. A low **match** means garbled speech: redo the shot.
- **cut_points** lists the pauses, the clean places to cut or start the next segment.
- **prompt_line** gives the transcript back as timed lines: `At 0 seconds they say, "..."`. Feed your recording through it before LC H3 Audio Sync, paste the lines into the prompt and change "they" to the character.
- Whisper downloads once to `models/stt/` (small = about 460 MB) and runs on the CPU: a few seconds a clip, and the GPU stays free for H3.

💡 **Vision Assistant → Movie** (LC Vision) shows all of it in plain words, with one-click redo and Make all remaining.

---

## Colors

`web/lc_node_colors.js` — AV family teal (`#2a6a6a` audio, `#1a5a7a` video). Distinct from LC123 sigma `#1c6d6d`. VAE Decode uses stock Comfy red.

`web/lc_color.js` is the launch-color helper (does not overwrite a saved color).

---

## License

MIT. See `LICENSE`.
