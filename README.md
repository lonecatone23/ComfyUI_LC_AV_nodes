# ComfyUI LC Audio_Video Nodes

Custom nodes for ComfyUI by lonecatone23.

* **Repo:** https://github.com/lonecatone23/ComfyUI_LC_AV_nodes
* **Image companion:** https://github.com/lonecatone23/ComfyUI_LC123_nodes
* **CivitAI:** lonecatone23
* **Support:** https://ko-fi.com/lonecatone
* **Version:** 0.5.30 · **27 nodes**

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
| LC Save Video 💾 | `LCSaveVideo` | ffmpeg encode. Inputs: images, audio, av_pipe, metadata, relay. No `LC_PIPE` socket. **relay** (from LC Relay VSR Upscale, MaskMaker) saves the upscaled video a few frames at a time so it never fills your RAM (5 s at 4x: 0.3 GB instead of about 12 GB); with Relay bypassed, images is saved instead. Writes the ComfyUI prompt + workflow into the mp4 for CivitAI; **civitai_lookup** links your models and LoRAs too. |
| LC Load Video 🎞️ | `LCLoadVideo` | A video file in, its frames, sound and frame rate out, exactly as the file has them. Streams with a progress bar; Cancel works. |
| LC Preview Video 📺 | `LCPreviewVideo` | Plays a video file, or frames with their sound, on the node. Nothing saved to output. |
| LC Save Video Metadata 🏷️ | `LCSaveVideoMetadata` | `LC_SAVE_META` + A1111 `parameters`. Works with Save Video or LC123 Save Image. |
| LC Final Frame 🖼️ | `LCFinalFrame` | Last frame of an IMAGE batch. |
| LC VAE Decode 🧩 | `LCVaeDecode` | Boolean wrapper: core `VAEDecode` or core `VAEDecodeTiled`. |
| LC Get Latent Size 📐 | `LCGetLatentSize` | Pixel width / height, batch, frames. Optional VAE for compression. |
| LC Load Audio 📂 | `LCLoadAudio` | Load from input. `start_time` / `end_time` cut, preview / pause, timeline handles. **transcript** out: the .txt / .srt LC Save Audio saved next to the clip, cut to match. |
| LC Audio Duration ⏱️ | `LCAudioDuration` | Seconds int + float (and minutes). |
| LC Audio Crop ✂️ | `LCAudioCrop` | Crop AUDIO with `MM:SS` start / end. |
| LC Audio Channel 🎚️ | `LCAudioChannel` | Stereo → L / R. |
| LC Audio Separate 🎼 | `LCAudioSeparate` | Demucs stems: Bass, Drums, Other, Vocals. |
| LC Audio EQ 3-Band 🎛️ | `LCAudioEqualizer3Band` | Low shelf, mid peak, high shelf. |
| LC Audio Volume 🔊 | `LCAudioVolume` | Gain 0–4, soft clip. |
| LC Foley 🔊 | `LCFoley` | Sound effects in sync with a video (frames + fps), or from a prompt alone. MMAudio, 44.1 kHz. Model files (about 5.6 GB) download to `models/mmaudio` on first use (fp16 safetensors from the Kijai/MMAudio_safetensors Hugging Face repo, so files already there are reused). Nothing to pip install. About 6 GB VRAM; under about 7.5 GB free it loads one model at a time (about 3 GB, same result). With **free_after_pass** on in LC Dual Model Optimizer (LC123), it lets go of MMAudio after the sound to save RAM. |

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

VHS-style options: frame_rate, loop_count, filename_prefix, format, pix_fmt, crf, civitai_lookup, pingpong, save_output, prune_outputs.

⚠️ **0.5.30:** `civitai_lookup` takes the slot of the old `civitai_still` (the first-frame PNG, gone along with the sidecar `.json`). Workflows saved with the PNG on open with the lookup **on**: switch it off if you don't want hashes sent to CivitAI.

`prune_outputs` keeps intermediate frame PNGs off disk.

`save_metadata` writes the prompt + workflow tags and a settings comment into the video. **CivitAI** reads the tags: fill `models` + `civitai_air` on the metadata node and the upload links your workflow page and reads as an external generator. CivitAI reads no file hashes from a video, so models and LoRAs only link when `civitai_lookup` is on: it finds each file's CivitAI page by its hash (one call to civitai.com per file, cached in `user/LC/civitai_air_cache.json`). Needs LC123.

**0.6.1:** samplers that get their seed, steps, sampler or scheduler through wires (pipes, switches) made CivitAI drop the whole upload's metadata. The copy of the prompt written for CivitAI now leaves those wired values out and adds a plain summary of the run (prompt, negative, seed, steps, cfg, sampler, size) from Save Video Metadata, so the upload shows the generation data and the workflow link. The workflow tag and the run itself are unchanged.

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

What LC Save Audio made shows in the list too, marked `[output]`. So do the pack's test voices (`voices/...`). **transcript** brings back the .txt / .srt saved next to the clip, only the lines inside the cut. Empty when there is none (run LC Transcribe instead).

---

## Voice and dialogue 🗣️ (BETA)

Make a voice, give it feelings, and have a cast read a script. Everything is built in: no other packs.

### 🛑🚫✋ Read first

- The models download on first use: IndexTTS-2 for LC Dialogue (about 8 GB), Maya1 for LC Voice Design, Whisper for **check** and LC Transcribe.
- Voices live in `models/voices`. **LC Test Voice** comes with the pack. Saved emotions are small files of their own, no audio:

```
ComfyUI/
├── models/
│   └── voices/
│       └── Alita.wav           ← a speaker
└── user/
    └── LC_AV/
        └── emotions/
            └── bitter.json     ← a saved emotion
```

**LC Voice Design 🎙️ (BETA)**
- Maya1 makes a voice from the words it was trained on: **sex, age, accent, pitch, timbre, tone, intensity, pacing**. Anything else makes it babble, so those are dropdowns.
- **preset**: 8 tested voices (F_American_20yo, M_British_30yo, etc.). Hover it for what each one is. Custom uses the dropdowns.
- **script**: what the voice reads (Range, Storyteller, Argument, Good news). Custom reads the box.
- Each seed is a different person for the same picks.
- **check** listens to each take (Whisper) and tries the next seed when Maya babbles or reads the description out loud (up to 4 takes).

**LC Voice Shaper 🦜 (BETA)**
- Pitch and voice size move on their own, so deeper sounds like a bigger person, not a slowed tape.
- Presets for baritone, bass, tenor, young girl, young boy, elderly woman and man, etc. Drag a column on the graph to nudge it.
- **tremble** and **breathiness** are what make a voice old. 0.3 to 0.5 each. Breathiness gets hissy past about 0.6.

**LC Save Audio 💽**
- Wire **voice_name** in and it also saves the voice as `models/voices/<voice_name>.wav`, a speaker for LC Dialogue.

**LC Emotion Editor 🎨**
- Build a feeling (valence / arousal pad plus named emotions) and save it by name. The test voice is only there so you can hear it.
- `bitter` works for every speaker. `Haley bitter` is Haley's own and wins over `bitter` for her.
- ⚠️ Feed it about 30 seconds of a lively voice. A short flat clip gives very little difference.

**LC Load Voice 🎙️**
- One speaker per node; **voice_name** is the name the script uses. Chain several for the whole cast on one wire.

**LC Dialogue 🗣️**
- IndexTTS-2 speaks the script: a voice and an emotion mix per line, and the feeling can change mid-line.
- Edit one line and only that line renders again. 💡 Keep the seed on **fixed** or every line renders again.

**LC Voice Changer 🔁 (BETA)**
- Keeps the words, timing and feeling of a recording and swaps the voice (ChatterBox VC). Act a line yourself, then give it a designed voice.
- **voice**: only its first 10 s are used, so start the clip on clear speech.
- **passes**: each one runs the result through again, closer to the target. Past 3 it starts to smear.
- The model (about 1 GB) downloads to `models/TTS/chatterbox/English` on first use; a copy from TTS-Audio-Suite is reused.

**LC Voice Match 🔍** / **LC Voice Clean 🧽**
- Match: how alike two voices are (1.0 = the same voice). Clean: noise reduction, de-esser and gate.

### 💡 Writing a Voice Design script

- Pick a **script**, or write your own on **Custom**. About 30 seconds.
- A feeling tag every sentence or two: `<laugh>` `<sigh>` `<angry>` `<whisper>` `<gasp>` `<excited>` `<cry>` `<giggle>` `<chuckle>`.
- Every clone of the voice inherits the range of this one take. No tags = a flat voice, and everything made from it stays flat (the info line warns you).
- Leave `<scream>` out. It cracks.
- Start from a **preset**, then change one dropdown at a time. **tone** energetic gives the most range.

### 💡 Writing a Dialogue script

```
[Alice|angry:0.6,sad:0.3] You're late again. I made dinner hours ago.
[Bob|sad:0.4,arousal:-0.5] I know. [pause:400ms] I know. The meeting ran over.
[Alice|sad] I'm fine. [angry:0.9] I said I'm fine!
```

- One line per turn: `[Speaker|emotion:weight] text`. Emotions: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm.
    - The weights are shares of the mix: IndexTTS-2 holds the whole mix at 0.8 or less, so `sad:0.5,angry:0.3` and `sad:1,angry:0.6` sound the same.
- `valence` (unpleasant to pleasant) and `arousal` (calm to energetic) go from -1 to 1.
- `[angry:0.9]` inside a line switches the feeling from that word on. `[neutral]` clears it.
- `[pause:600ms]` for a beat, `rate:slow` / `rate:fast`. Commas, periods and question marks do the timing too.
- `[Haley|bitter]` uses a saved emotion from LC Emotion Editor.
- Lines starting with `#` are skipped (handy for notes to yourself).

**Emotions 🎭**

| Tag | Sounds like | Also accepts |
|---|---|---|
| `happy` | bright, smiling | joy |
| `angry` | sharp, pushed | anger, mad |
| `sad` | heavy, low | sadness |
| `afraid` | tight, shaky | fear, scared, fearful |
| `disgusted` | sneering, contempt | disgust |
| `melancholic` | tired, wistful | tired |
| `surprised` | lifted, sudden | surprise |
| `calm` | relaxed, even | relaxed |

Saved feelings that come with the pack (LC Emotion Editor): bitter, bored, excited, furious, heartbroken, nervous, sarcastic, smug, tender, terrified. A saved name wins over a built-in one.

**Strength limits ⚠️**
- Each emotion: 0 to 1.2. Anything higher is cut to 1.2.
- `strength:0.5` scales the whole line's mix (same 1.2 ceiling per emotion).
- `valence` / `arousal`: -1 to 1.
- `rate`: 0 to 4 (0.8x to 1.2x speed), or slow / normal / fast.

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
- **📦 Export / 📥 Import:** take a project to another computer. Export packs the story, plan and every picture it uses (they live in the input folder) into one zip; Import puts them back there and opens the project.
    - Renders stay out unless you tick them in (every take plus the movie gets big).
    - A name that's taken: import it under a new name, or replace it (the old folder moves to `output/LC_H3/_replaced`, nothing is deleted). A different picture with the same name is kept, the imported one gets `_imported` added.
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
- RIFE by hzwer (Practical-RIFE, MIT); the model code is HolyWu's vs-rife (MIT). See `vendor/rife/UPSTREAM.txt`.

💡 **Vision Assistant → Movie** (LC Vision) shows all of it in plain words, with one-click redo and Make all remaining.

---

## Colors

`web/lc_node_colors.js`: AV family teal (`#2a6a6a` audio, `#1a5a7a` video). Distinct from LC123 sigma `#1c6d6d`. VAE Decode uses stock Comfy red.

`web/lc_color.js` is the launch-color helper (does not overwrite a saved color).

---

## License

MIT. See `LICENSE`.

Third-party code inside this pack keeps its own license:

- **`vendor/dbs_split`:** Dynamic Band Splitting (Apache-2.0). See `vendor/dbs_split/UPSTREAM.md`.
- **`vendor/snac`:** SNAC audio decoder (MIT). See `vendor/snac/UPSTREAM.md`.
- **`vendor/indextts`:** IndexTTS-2 inference code (bilibili Model Use License Agreement, bilibili), taken from TTS-Audio-Suite by Shmuel Ronen (MIT). See `vendor/indextts/UPSTREAM.md`.
- **`vendor/chatterbox_vc`:** ChatterBox VC (MIT, Resemble AI) with the S3 tokenizer (Apache-2.0, Xingchen Song), taken from TTS-Audio-Suite by Shmuel Ronen (MIT). See `vendor/chatterbox_vc/UPSTREAM.md`.
- **`vendor/mmaudio`:** MMAudio inference code (MIT, Sony Research), with BigVGAN v2 (MIT, NVIDIA), Synchformer (MIT, Vladimir Iashin) and the CLIP parts of OpenCLIP (MIT), taken from the original repositories. See `vendor/mmaudio/UPSTREAM.md`.
- **`vendor/rife`:** RIFE model code (IFNet 4.7, 4.17, 4.26) from vs-rife (MIT, HolyWu), for hzwer's Practical-RIFE models (MIT). See `vendor/rife/UPSTREAM.txt`.
- **`lc_dsp.py`:** beat, onset, pitch and time-stretch code ported from librosa 0.11 (ISC). Credits are in the file header.

Models download on first use and keep their own licenses: Maya1 (Apache-2.0), SNAC 24 kHz (MIT), Whisper (MIT), ChatterBox (MIT), IndexTTS-2 (bilibili Model Use License Agreement). LC Foley's MMAudio weights are **CC BY-NC 4.0 (non-commercial)**; its CLIP (Apple DFN5B) has Apple's own license and the BigVGAN v2 vocoder is MIT. They are never bundled with this pack.
