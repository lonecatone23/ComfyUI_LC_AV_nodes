"""
LC Save Video + LC Save Video Metadata
--------------------------------------
Video does not get PNG text chunks. Write container tags when the encoder
allows it, plus a sidecar .json with the same field names as LC Save Image
where possible (positive, negative, steps, sampler, seed, models,
civitai_air, hashes, parameters).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

import folder_paths

from .lc_av_media import audio_duration_sec, audio_parts, tensor_to_uint8_frames
from .lc_av_meta import (
    META_TYPE,
    as_meta,
    build_parameters,
    civitai_resources_payload,
    pipe_get,
    sidecar_fields,
    txt,
)
from .lc_av_pipe import PIPE_TYPE
from .lc_av_types import LC_PIPE, copy_shared_from_lc_pipe, meta_in


def _join_path(*parts: str) -> str:
    chunks = []
    for p in parts:
        s = txt(p).replace("\\", "/").strip("/")
        if s:
            chunks.append(s)
    return "/".join(chunks)


def _write_civitai_png(path, frame_u8, meta, params):
    """First-frame PNG with the same text keys LC Save Image / Civitai read."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    img = Image.fromarray(frame_u8[..., :3], mode="RGB")
    info = PngInfo()
    if params:
        info.add_text("parameters", params)
    air = txt(meta.get("civitai_air"))
    resources = civitai_resources_payload(air)
    if resources:
        info.add_text("civitaiResources", json.dumps(resources, separators=(",", ":")))
        info.add_text("civitai_air", air)
    hashes = meta.get("hashes")
    if hashes:
        info.add_text("hashes", hashes if isinstance(hashes, str) else json.dumps(hashes))
    for key in ("positive", "negative", "models"):
        val = txt(meta.get(key))
        if val:
            info.add_text(key, val)
    img.save(path, pnginfo=info)


class LCSaveVideoMetadata:
    """Collect save metadata. Optional LC_AV_PIPE in; widgets override. No pipe out."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pipe": (
                    LC_PIPE,
                    {
                        "tooltip": "LC123 LC_PIPE. Prompts/seed/steps/size only.",
                    },
                ),
                "av_pipe": (
                    PIPE_TYPE,
                    {
                        "tooltip": "LC_AV_PIPE. Widgets override both pipes.",
                    },
                ),
                "target": (
                    ["auto", "image", "video"],
                    {
                        "default": "auto",
                        "tooltip": "auto = include video extras when fps/duration exist. Same metadata dict works on LC Save Image and LC Save Video.",
                    },
                ),
                "positive": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Override pipe positive prompt."},
                ),
                "negative": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Override pipe negative prompt."},
                ),
                "seed_value": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": False,
                        "tooltip": "Seed. -1 = use pipe seed if present. Plain INT — no control-after-generate.",
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "tooltip": "0 = use pipe steps."},
                ),
                "sampler": (
                    "STRING",
                    {"default": "", "tooltip": "Empty = use pipe sampler_name."},
                ),
                "scheduler": (
                    "STRING",
                    {"default": "", "tooltip": "Empty = use pipe scheduler."},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.05,
                        "tooltip": "0 = use pipe cfg_1.",
                    },
                ),
                "width": (
                    "INT",
                    {"default": 0, "min": 0, "max": 65536, "tooltip": "0 = use pipe width."},
                ),
                "height": (
                    "INT",
                    {"default": 0, "min": 0, "max": 65536, "tooltip": "0 = use pipe height."},
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0 = use pipe denoise.",
                    },
                ),
                "models": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Model names. One field, comma-separated. Not a second widget.",
                    },
                ),
                "civitai_air": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Exact Civitai AIR URN paste or model URL. Optional. Do not invent version/file ids.",
                    },
                ),
                "extra_params": (
                    "STRING",
                    {"default": "", "tooltip": "Appended to the parameters line."},
                ),
            },
        }

    RETURN_TYPES = (META_TYPE, "STRING")
    RETURN_NAMES = ("metadata", "parameters")
    FUNCTION = "build"
    CATEGORY = "LC AV/io"
    DESCRIPTION = (
        "Same metadata dict as LC123 Save Metadata (type LC_SAVE_META). "
        "Wire to LC Save Image or LC Save Video. Pipe accepts LC_PIPE or LC_AV_PIPE."
    )

    def build(
        self,
        pipe=None,
        av_pipe=None,
        target="auto",
        positive="",
        negative="",
        seed_value=-1,
        steps=0,
        sampler="",
        scheduler="",
        cfg=0.0,
        width=0,
        height=0,
        denoise=0.0,
        models="",
        civitai_air="",
        extra_params="",
    ):
        meta = {"_type": META_TYPE, "target": target or "auto"}

        merged = {}
        if isinstance(pipe, dict):
            merged.update(copy_shared_from_lc_pipe(pipe))
        if isinstance(av_pipe, dict):
            merged.update(av_pipe)
        pipe = merged

        pos = txt(positive) or txt(pipe_get(pipe, "positive_prompt", "positive"))
        neg = txt(negative) or txt(pipe_get(pipe, "negative_prompt", "negative"))
        if pos:
            meta["positive"] = pos
        if neg:
            meta["negative"] = neg

        seed = int(seed_value) if seed_value is not None else -1
        if seed < 0:
            pseed = pipe_get(pipe, "seed")
            try:
                seed = int(pseed) if pseed is not None else -1
            except (TypeError, ValueError):
                seed = -1
        if seed >= 0:
            meta["seed"] = seed

        st = int(steps or 0)
        if st <= 0:
            pst = pipe_get(pipe, "steps", "total_steps")
            try:
                st = int(pst) if pst is not None else 0
            except (TypeError, ValueError):
                st = 0
        if st > 0:
            meta["steps"] = st
            meta["total_steps"] = st

        samp = txt(sampler) or txt(pipe_get(pipe, "sampler_name", "sampler"))
        if samp:
            meta["sampler"] = samp
            meta["sampler_name"] = samp

        sched = txt(scheduler) or txt(pipe_get(pipe, "scheduler", "scheduler_name"))
        if sched:
            meta["scheduler"] = sched

        cf = float(cfg or 0.0)
        if cf == 0.0:
            pcf = pipe_get(pipe, "cfg_1", "cfg")
            try:
                cf = float(pcf) if pcf is not None else 0.0
            except (TypeError, ValueError):
                cf = 0.0
        if cf:
            meta["cfg"] = cf

        w = int(width or 0)
        h = int(height or 0)
        if w <= 0:
            try:
                w = int(pipe_get(pipe, "width") or 0)
            except (TypeError, ValueError):
                w = 0
        if h <= 0:
            try:
                h = int(pipe_get(pipe, "height") or 0)
            except (TypeError, ValueError):
                h = 0
        if w > 0:
            meta["width"] = w
        if h > 0:
            meta["height"] = h

        dn = float(denoise or 0.0)
        if dn == 0.0:
            pdn = pipe_get(pipe, "denoise")
            try:
                dn = float(pdn) if pdn is not None else 0.0
            except (TypeError, ValueError):
                dn = 0.0
        if dn:
            meta["denoise"] = dn

        mods = txt(models) or txt(pipe_get(pipe, "models"))
        if mods:
            parts = [p.strip() for p in mods.split(",") if p.strip()]
            meta["models"] = ", ".join(parts)

        air = txt(civitai_air) or txt(pipe_get(pipe, "civitai_air"))
        if air:
            meta["civitai_air"] = air

        extra = txt(extra_params)
        if extra:
            meta["extra_params"] = extra

        for k in ("width", "height", "frame_rate", "fps", "duration", "frame_count", "sample_rate"):
            v = pipe_get(pipe, k)
            if v is not None:
                meta[k] = v
        if "frame_rate" not in meta and meta.get("fps") not in (None, ""):
            meta["frame_rate"] = meta["fps"]

        params = build_parameters(
            meta,
            meta.get("width") or 0,
            meta.get("height") or 0,
            meta.get("frame_rate") or meta.get("fps"),
            meta.get("duration"),
        )
        meta["parameters"] = params
        return (meta, params)



def _ffmpeg_bin():
    return shutil.which("ffmpeg")


def _write_wav(path, wave, sr):
    import wave as wavmod

    arr = np.asarray(wave)
    if arr.ndim == 1:
        arr = arr[None, :]
    ch, n = arr.shape
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    interleaved = pcm.T.reshape(-1)
    with wavmod.open(path, "wb") as w:
        w.setnchannels(int(ch))
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(interleaved.tobytes())


def _apply_pingpong(frames):
    if frames.shape[0] < 2:
        return frames
    return np.concatenate([frames, frames[-2:0:-1]], axis=0)


def _apply_loops(frames, loop_count):
    extra = int(loop_count or 0)
    if extra <= 0:
        return frames
    return np.concatenate([frames] * (extra + 1), axis=0)


_FORMATS = {
    "video/h264-mp4": {
        "ext": "mp4",
        "vcodec": ["-c:v", "libx264", "-pix_fmt", "PIX", "-movflags", "+faststart"],
        "acodec": ["-c:a", "aac"],
    },
    "video/h265-mp4": {
        "ext": "mp4",
        "vcodec": ["-c:v", "libx265", "-pix_fmt", "PIX", "-tag:v", "hvc1", "-movflags", "+faststart"],
        "acodec": ["-c:a", "aac"],
    },
    "video/hevc-mp4": {
        "ext": "mp4",
        "vcodec": ["-c:v", "libx265", "-pix_fmt", "PIX", "-tag:v", "hvc1", "-movflags", "+faststart"],
        "acodec": ["-c:a", "aac"],
    },
    "video/av1-mp4": {
        "ext": "mp4",
        "vcodec": ["-c:v", "libsvtav1", "-pix_fmt", "PIX", "-movflags", "+faststart"],
        "acodec": ["-c:a", "aac"],
    },
    "video/vp9-webm": {
        "ext": "webm",
        "vcodec": ["-c:v", "libvpx-vp9", "-pix_fmt", "PIX"],
        "acodec": ["-c:a", "libopus"],
    },
    "video/av1-webm": {
        "ext": "webm",
        "vcodec": ["-c:v", "libsvtav1", "-pix_fmt", "PIX"],
        "acodec": ["-c:a", "libopus"],
    },
}


def _encode_ffmpeg(frames_u8, dest, fps, fmt_key, crf, pix_fmt, audio=None, comment="", audio_bitrate="192k"):
    ff = _ffmpeg_bin()
    if not ff:
        raise RuntimeError("ffmpeg not on PATH")
    spec = _FORMATS[fmt_key]
    n, h, w, _c = frames_u8.shape
    tmpdir = tempfile.mkdtemp(prefix="lc_av_")
    try:
        raw = os.path.join(tmpdir, "frames.rgb")
        frames_u8.tofile(raw)
        vcodec = [x if x != "PIX" else pix_fmt for x in spec["vcodec"]]
        cmd = [
            ff, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", raw,
        ]
        wav_path = None
        wave, sr = audio_parts(audio)
        if wave is not None and sr > 0:
            wav_path = os.path.join(tmpdir, "a.wav")
            _write_wav(wav_path, wave, sr)
            cmd += ["-i", wav_path]
        cmd += vcodec + ["-crf", str(int(crf))]
        if comment:
            clip = comment[:8000]
            cmd += [
                "-metadata", f"comment={clip}",
                "-metadata", f"description={clip}",
                "-metadata", "comment_lang=eng",
            ]
        if wav_path:
            cmd += spec["acodec"] + ["-b:a", str(audio_bitrate), "-shortest"]
        cmd.append(dest)
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class LCSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_rate": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.0,
                        "max": 240.0,
                        "step": 0.01,
                        "tooltip": "Playback fps. 0 = use AV pipe fps or 24. Not an AnimateDiff default.",
                    },
                ),
                "loop_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Extra full plays after the first (0 = play once).",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "LC_AV",
                        "tooltip": "Stem under Comfy output. Subfolder/name is allowed.",
                    },
                ),
                "format": (
                    list(_FORMATS.keys()),
                    {
                        "default": "video/h264-mp4",
                        "tooltip": "VHS-style video/* only. No gif/webp, no VAE/latent.",
                    },
                ),
                "pix_fmt": (
                    ["yuv420p", "yuv420p10le", "yuv444p"],
                    {
                        "default": "yuv420p",
                        "tooltip": "VHS uses yuv420p10le with h265. yuv420p is the compatible default.",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 22,
                        "min": 0,
                        "max": 51,
                        "tooltip": "Same meaning as VHS Video Combine crf.",
                    },
                ),
                "save_metadata": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "metadata",
                        "label_off": "no metadata",
                        "tooltip": "Sidecar JSON + mp4 comment/description + Civitai still PNG (parameters + civitaiResources).",
                    },
                ),
                "civitai_still": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "png still",
                        "label_off": "video only",
                        "tooltip": "Write first-frame PNG with A1111 parameters / civitaiResources. Civitai does not read mp4 comment.",
                    },
                ),
                "pingpong": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "pingpong",
                        "label_off": "forward",
                        "tooltip": "Append reversed frames (skip endpoints) for a clean loop.",
                    },
                ),
                "save_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "output",
                        "label_off": "temp",
                        "tooltip": "True = Comfy output folder. False = temp folder.",
                    },
                ),
                "prune_outputs": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "prune",
                        "label_off": "keep previews",
                        "tooltip": "Never write intermediate/utility frame PNGs or preview GIFs. Only the video file (and optional sidecar json) is saved. Raw encode files stay in a temp dir that is deleted after ffmpeg.",
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Frame batch. Used if av_pipe has no video."}),
                "audio": ("AUDIO", {"tooltip": "Optional soundtrack. Overrides av_pipe audio."}),
                "av_pipe": (PIPE_TYPE, {"tooltip": "LC_AV_PIPE with video/audio/fps."}),
                "metadata": (
                    meta_in,
                    {"tooltip": "LC123 Save Metadata or LC Save Video Metadata (LC_SAVE_META)."},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_as",)
    FUNCTION = "save"
    CATEGORY = "LC AV/io"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Save video with Video Combine-style options (frame rate, loop, format, "
        "pingpong, crf, pix_fmt, audio). No AnimateDiff, VAE, latent, gif, or webp."
    )

    def save(
        self,
        frame_rate=24.0,
        loop_count=0,
        filename_prefix="LC_AV",
        format="video/h264-mp4",
        pix_fmt="yuv420p",
        crf=22,
        save_metadata=True,
        civitai_still=True,
        pingpong=False,
        save_output=True,
        prune_outputs=True,
        images=None,
        audio=None,
        av_pipe=None,
        metadata=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        pipe = av_pipe if isinstance(av_pipe, dict) else {}

        from .lc_av_media import frames_from_any

        frames_src = frames_from_any(images)
        if frames_src is None and isinstance(pipe, dict):
            frames_src = frames_from_any(pipe)
        if frames_src is None:
            raise ValueError(
                "LC Save Video: no frames. Wire the IMAGE batch from VAE Decode "
                "(or Create Video images) into images, or an LC_AV_PIPE that holds video. "
                "A Comfy VIDEO object also works. Metadata/pipe-only is not enough."
            )

        frames = tensor_to_uint8_frames(frames_src)
        if pingpong:
            frames = _apply_pingpong(frames)
        frames = _apply_loops(frames, loop_count)
        n, h, w, _c = frames.shape

        fps_f = float(frame_rate or 0.0)
        if fps_f <= 0 and isinstance(pipe, dict):
            try:
                fps_f = float(pipe.get("frame_rate") or pipe.get("fps") or 0.0)
            except (TypeError, ValueError):
                fps_f = 0.0
        if fps_f <= 0:
            fps_f = 24.0

        aud = audio
        if aud is None and isinstance(pipe, dict):
            aud = pipe.get("audio") or pipe.get("waveform")

        duration = float(n) / fps_f
        aud_dur = audio_duration_sec(aud)

        meta = as_meta(metadata)
        meta.setdefault("width", w)
        meta.setdefault("height", h)
        meta.setdefault("frame_rate", fps_f)
        meta.setdefault("duration", duration)
        meta.setdefault("frame_count", n)
        params = build_parameters(meta, w, h, fps_f, duration)
        meta["parameters"] = params

        fmt_key = format if format in _FORMATS else "video/h264-mp4"
        ext = _FORMATS[fmt_key]["ext"]
        combined = txt(filename_prefix) or "LC_AV"

        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        full_dir, file_stem, counter, subfolder, _pfx = folder_paths.get_save_image_path(
            combined, output_dir, w, h
        )
        os.makedirs(full_dir, exist_ok=True)

        fname = f"{file_stem}_{counter:05d}.{ext}"
        dest = os.path.join(full_dir, fname)
        while os.path.exists(dest):
            counter += 1
            fname = f"{file_stem}_{counter:05d}.{ext}"
            dest = os.path.join(full_dir, fname)

        comment = params if save_metadata else ""
        _encode_ffmpeg(
            frames, dest, fps_f, fmt_key, crf, pix_fmt,
            audio=aud, comment=comment, audio_bitrate="192k",
        )

        if save_metadata and civitai_still:
            still = os.path.splitext(dest)[0] + ".png"
            try:
                _write_civitai_png(still, frames[0], meta, params)
            except Exception as e:
                print(f"[LC AV] Civitai still skip: {e}")

        if save_metadata:
            side = dest + ".json"
            payload = sidecar_fields(meta, {
                "hashes": meta.get("hashes"),
                "civitaiResources": civitai_resources_payload(txt(meta.get("civitai_air"))),
                "audio_duration": aud_dur or None,
                "container": ext,
                "format": fmt_key,
                "filename": fname,
                "crf": crf,
                "pix_fmt": pix_fmt,
                "pingpong": bool(pingpong),
                "loop_count": int(loop_count or 0),
            })
            try:
                with open(side, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except OSError as e:
                print(f"[LC AV] sidecar skip: {e}")

        rel = dest
        try:
            rel = os.path.relpath(dest, output_dir)
        except Exception:
            pass
        rel = str(rel).replace("\\", "/")
        preview = {
            "filename": fname,
            "subfolder": subfolder or "",
            "type": "output" if save_output else "temp",
            "format": fmt_key,
        }
        return {
            "ui": {
                "text": [rel],
                "gifs": [preview],
                "video": [preview],
            },
            "result": (rel,),
        }


NODE_CLASS_MAPPINGS = {
    "LCSaveVideo": LCSaveVideo,
    "LCSaveVideoMetadata": LCSaveVideoMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCSaveVideo": "LC Save Video 💾",
    "LCSaveVideoMetadata": "LC Save Video Metadata 🏷️",
}
