# ComfyUI LC Audio_Video Nodes

Custom nodes for ComfyUI by lonecatone23.

* **Repo:** https://github.com/lonecatone23/ComfyUI_LC_AV_nodes
* **Image companion:** https://github.com/lonecatone23/ComfyUI_LC123_nodes
* **CivitAI:** lonecatone23
* **Support:** https://ko-fi.com/lonecatone
* **Version:** 0.5.25 · **27 nodes**

> Small tools that remove friction: frames, soundtrack, metadata, fewer extra packs.

This pack is **not** a fork of LC123. Do not merge the two folders.
`LC_PIPE` stays on the image pack. This pack uses **`LC_AV_PIPE`**.

Release history lives in **git tags**. This page describes the pack as it is now.

---

## Install

```
ComfyUI/custom_nodes/ComfyUI_LC_AV_nodes/__init__.py
```

`__init__.py` must sit **directly** in that folder, not in `ComfyUI_LC_AV_nodes/ComfyUI_LC_AV_nodes/`.
If you unzip a pack zip *inside* an existing clone, move the inner files up one level.

Restart ComfyUI. Console should print `[LC AV] total … nodes`.
Hard-refresh the browser after a `web/` JS update.

**Requirements:** ComfyUI’s Python (`torch`, `numpy`). Save Video needs **ffmpeg** on PATH.
Audio Separate needs **torchaudio** (no Numba / SoX).

Optional example graph: `workflows/LC_AV Node Examples.json`  
(BETA: examples only. Add nodes from the node menu; do not treat the graph as a production template.)

---

## What this pack is

| Title | Class | Role |
| --- | --- | --- |
| LC AV Pipe (in/edit) 🎬 | `LCAvPipeEdit` | Pack or edit `LC_AV_PIPE`. In: `pipe`, `av_pipe`, then video / audio / fps / … Out: `av_pipe`. |
| LC AV Pipe Out 🎬 | `LCAvPipeOut` | Unpack. In: `pipe` + `av_pipe`. No `LC_PIPE` on the output side. |
| LC Pipe → AV Pipe 🔌 | `LCPipeToAvPipe` | One `LC_PIPE` in, one `LC_AV_PIPE` out. Shared keys only. |
| LC Create Video 🎬 | `LCCreateVideo` | IMAGE batch + fps (+ optional AUDIO) → `LC_AV_PIPE`. Does not encode a file. |
| LC Save Video 💾 | `LCSaveVideo` | ffmpeg encode. Inputs: images, audio, av_pipe, metadata, relay. No `LC_PIPE` socket. **relay** (from LC Relay VSR Upscale, MaskMaker) saves the upscaled video a few frames at a time so it never fills your RAM (5 s at 4x: 0.3 GB instead of about 12 GB); with Relay bypassed, images is saved instead. Writes the ComfyUI prompt + workflow into the mp4 for CivitAI. |
| LC Load Video 🎞️ | `LCLoadVideo` | A video file in, its frames, sound and frame rate out, exactly as the file has them. Streams with a progress bar; Cancel works. |
| LC Preview Video 📺 | `LCPreviewVideo` | Plays a video file, or frames with their sound, on the node. Nothing saved to output. |
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

No **civitai_air** on the pipe. Paste AIR on **Save Video Metadata**.

---

## Load Video 🎞️

Path in (a full path, or one inside ComfyUI's output or input folder): every frame at the file's own size, its sound and its frame rate out. Nothing to set: no resizing, frame skipping or trimming. Frames stream in through ffmpeg with a progress bar, and Cancel stops it. Wire LC H3 Make's video straight in.

## Preview Video 📺

Plays a video on the node: a file (path), or a batch of frames with its sound (fps sets the rate). Frames are encoded to a temp file, so nothing lands in your output folder.

---

## Save Video 💾

VHS-style options: frame_rate, loop_count, filename_prefix, format, pix_fmt, crf, pingpong, save_output, prune_outputs.

`prune_outputs` keeps intermediate frame PNGs off disk.

`save_metadata` writes sidecar JSON + container comment. **CivitAI** reads PNG chunks, not the mp4 comment. Keep `civitai_still` on and upload the first-frame PNG (fill `models` + `civitai_air` on the metadata node).

Frames stream into ffmpeg one at a time with a progress bar, so long clips don't fill your RAM, and Cancel stops it (no half-written file is left).

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

Minute-long MiniMax H3 videos made in short pieces, one shot at a time, on the same card that makes 5 s clips. **LC H3 Story 🎬** holds the cast, places and shots. **LC H3 Make 🎥** renders them and joins the movie. Example workflow: `MMH3 Showrunner V2.5.2` (with a guided tour: **Vision Assistant** → **Tour**).

📘 **[User Guide](docs/showrunner/USER_GUIDE.md)**: every part of the workflow, step by step, with screenshots. ✍️ **[Prompting Guide](docs/showrunner/PROMPTING_GUIDE.md)**: how to word looks, outfits, actions and lines.

**LC H3 Story (BETA) 🎬**
- The whole movie on one node: **cast**, **outfits**, **places**, **props** and **shots**, each on its own card.
- ✋ **The H3 prompt shows the shot's lines.** Make wraps the people, pictures and sound around them when the shot is made (the full prompt is saved in `takes/seg_NNN.json`). Open the shot's **H3 prompt** to read it. One thing is left out quietly: a place's sound line loses any clause about speech ("nobody speaks"), because naming speech makes H3 speak. You see the result in the H3 prompt.
- **Cast card:** a name, who they are (woman, man, cat, etc.), their look, a headshot (head and hair only), a 👗 outfit picture and a voice.
    - 💡 Use a neutral face on a plain grey background. A card's background, props and expression leak into every shot it's in.
- **Outfit card:** a change of clothes, with a simple name ("red cocktail dress"), who wears it, a head-to-toe description and a picture. A state of dress works too ("naked", "topless, black jeans", "wrapped in a white towel"): H3 gets it as written.
- **Prop card:** an object that matters (a cup, a sign), with its own picture.
- 📚 Every cast, outfit, prop and 🌐 360 slot has a **Library** tab: 102 ready-made pictures, free (CC BY 4.0).
- **Place card:** a description, the sound of the room ("rain on the window, cups clinking, footsteps": sounds only, never who speaks) and a picture of it empty. A **360 panorama** lets close-ups face any side of the room: aim them with the 🌐 viewer on the shot.
- **Shot card:** framing, who it's on (everyone, one person, or **nobody** for an empty shot), camera move, cut or keep rolling, the action, who speaks and the line.
    - 🎞 **keep rolling** carries straight on from the shot before and **locks** its framing, place, 🌐 angle, outfits and start frame (🔒). People can still walk in, and they get their face pictures.
    - 🎛 **LoRA for this shot**: any LoRA for that shot only, on top of the pipe's LoRAs, picked like the LC LoRA Loader (folder picker, strength, ℹ trigger words). Keep rolling uses its cut's LoRAs.
    - **seconds**: **auto** fits the beats and the line, or type your own. The number you type wins.
    - **Name (V.O.)** in the speaker list is a voiceover: heard off screen, lips stay closed.
    - 💡 One line over a cut: end it with `...` on this shot and start the next shot's line with `...`. The next shot picks the line up straight away.
    - 🖼 **start** / 🏁 **end** pin the first or last frame. H3 copies a pinned frame exactly, so make it sharp.
    - 📐 draws boxes: who stands where. **✨** on the 🖼 start slot turns them into a start frame.
    - Count props once ("each holds one cup") or extras show up.
- **✨ Write the story** turns a short pitch into cards with a local LLM, once. After that it's plain text you edit.
    - **🔞 NSFW** (off by default): lets the writer write adult scenes, the same rule as LC Vision's NSFW style tag.
- **The board** under the shots: a tile per shot. ✅ made, ✏️ changed since it was made, ⚠ join broken (a keep-rolling shot whose shot before was remade), 🔒 locked. ▶ plays it, 🩹 repairs it (a new take that keeps its last moment, so the shot after it still joins), ↻ makes that one shot again and stops, 📌 on a paused frame sets that place's room anchor, 🖼 on a paused frame starts the next shot from it.
- **▾** folds a shot to one line. The project bar stays pinned at the top.
- **max_shot_seconds**: the longest shot this machine renders in one go (20 default). A longer shot is made in parts, each carrying on from the one before.
- **H3 prompt** (under each shot): the shot's lines (Place, Camera, the timed action). The people, picture and sound sections are added around them when the shot is made. **↻** builds it from the cards again right now. **✎ Edit** lets you write the lines by hand: they go in as written, and changes to the cards stop changing it until you press **↶ Back to automatic**.
- **✨ on a picture slot** (cast, outfit, place, 🌐 360, prop, 🖼 start, 🏁 end) makes that picture. The Story switches on the part of the workflow that makes it, keeps Make off, runs once and puts everything back. A blank card name never writes over a picture, and **↶** brings back the picture before (the last 8 are kept). Press it while the movie renders and the picture is queued after the shot rendering now; the movie carries on after it.
- **📐 size** (under the project name) shows the size the project is made at. Change it and the next Run makes every shot made at the old size again. **💾 Save as** keeps a variant under a new name. **🆕 New** clears the board (it warns first when there is unsaved progress).
- The 🌐 shot angle is also the background of the 📐 boxes, so you place people on the view the shot will use.
- **🌐 360 from card** (under a place's room picture): LC Vision reads the room picture and writes the whole place out, then ✨ draws the 360 from that. Needs LC Vision.
- Everything saves to `output/LC_H3/<project>/` (story.json, plan.json, the takes and the movie). Pictures go in `input/lc_h3_refs/<project>/`.

**The picture makers** (inside the **Cards & Storyboard 🃏** subgraph of the example workflow; you run them with ✨ on the Story)
- **LC H3 Story Card Prompt**: a card's description as an image prompt and an empty latent at the size that picture needs (headshot, outfit, room, 360 panorama, prop). **kind** and **name** are set by ✨.
- **LC H3 Shot Card**: one shot's start or end frame as a storyboard job for any image model: the prompt, the 📐 boxes, the background (the shot's 360 view or the room) and the cast pictures.
- **LC H3 Save to Story**: puts the finished picture on its card (**from card**), or on a shot's start or end frame. A new take gets a number, so older pictures are kept (↶).
- Any image model works: the model sits in the workflow (the **LC Model Optimizer** and the LoRA loader in front of the subgraph), never in these nodes.

**LC H3 Make (BETA) 🎥**
- Renders the Story's shots one after another and joins them into the movie. Wire in the Story's **plan** and **refs**, and an **LC MiniMax H3 Pipe V2** (model, text encoder, both VAEs, size, steps, sampler, seed).
- **make**: **all remaining** (default) makes the whole movie in one Run. **next segment** makes one. **redo** makes segment number **index** again, with a new seed.
    - 💡 One Run does it all. **Stop** ends it with nothing left in the queue. Run again to carry on.
    - Edited a shot that's already made? Run remakes only the ones marked ✏️, the rest stay.
- **redo_after**: redoing a shot also sets aside the keep-rolling shots after it, since they no longer join. The old take is always kept in `takes/`.
- **refs_on**: **cuts only** (default) gives the smoothest joins. A keep-rolling shot carries on from the frames before it, and a fresh reference pulls it away.
- **hold_scene** (on): the first wide shot of a place becomes its layout, so later shots there keep who sits where.
- **memory**: **clip** (default) feeds about 1 s of the first shot back on every cut so faces, clothes and voices stay put. **memory_gap** 5 s stops the cut from opening on that clip's pose.
- **color_match**: 0 = off. 0.2 to 0.4 when **info** flags a scene drifting warmer or cooler.
- **crf** 16 is close to lossless.
- **info** reports the face match per shot (0.55 and up = same person, under 0.40 = someone else, redo it) and how smooth each join is.
- A shot's 🎛 LoRAs go on a copy of the model for that shot only; the next cut starts from the pipe's model again.
- 💡 960 x 544 is the low VRAM size. Frame Interpolate to 48 fps is the post step that clearly helps; upscalers were barely worth it in my tests.

**LC H3 Audio Sync (BETA) 🎤**
- Makes an H3 video perform YOUR track (a song, a line of dialogue) instead of sound the model invents. Put it between the latent and the sampler.
- Your recording is locked into the sound half, so the model only decides the picture that fits it, mouth included.
- 💡 Wire the H3 **positive** conditioning through it too: the track is also handed to H3 as a clean guide.
- ✋ **Put each line in the prompt with its time:** `At 0 seconds she says, "Hello there."` H3 moves the mouth when the prompt says the words are spoken, not when it hears them. Without the time she talks over the silence instead.
- **start_seconds** picks where in the track the clip starts. A track shorter than the clip is filled with **silence** or **loop**ed.
- The **audio** output is the track cut to the clip's length: wire it to the save node so picture and sound end together.
- ⚠️ H3 was trained to about 15 s; longer clips get a warning in **info**.

**LC Frame Interpolate 🎞️**
- Smooth motion with RIFE 4.26: makes the in-between frames so a 24 fps clip plays at 48, 60, or any rate (not just 2x). Each new frame lands at its real moment in time.
- **scene_cut** holds hard cuts instead of morphing across them (a morph across a cut is a smeared ghost frame). 0 = off.
- Wire the clip's **audio** and the output is exactly as long as the sound, so picture and audio end together.
- **model**: rife426.pth (best on AI video in our tests, downloads once, 21 MB) or **crossfade** (no model, fast, soft on motion). Uses `models/rife/` or ComfyUI-Frame-Interpolation's checkpoints if you have them.
- RIFE code from ComfyUI-Frame-Interpolation (Fannovel16) and Practical-RIFE (hzwer), both MIT.

💡 **Vision Assistant → Movie** (LC Vision) shows all of it in plain words, with one-click redo and Make all remaining.

---

## Colors

`web/lc_node_colors.js`: AV family teal (`#2a6a6a` audio, `#1a5a7a` video). Distinct from LC123 sigma `#1c6d6d`. VAE Decode uses stock Comfy red.

`web/lc_color.js` is the launch-color helper (does not overwrite a saved color).

---

## License

MIT. See `LICENSE`.
