"""
LC Save Audio
-------------
Saves AUDIO as FLAC, WAV (24-bit), MP3 or Opus, with the workflow embedded (FLAC / MP3 / Opus), a player on the node,
and an optional transcript saved next to it (.txt, .srt subtitles, or both). save_output off = preview only (temp).
"""

from __future__ import annotations

import json
import os
import re
from io import BytesIO

import numpy as np

from .lc_av_media import audio_parts

_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)
_BITRATE = {"320k": 320000, "256k": 256000, "192k": 192000, "128k": 128000, "96k": 96000, "64k": 64000}
_LINE = re.compile(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*-\s*(\d+):(\d+(?:\.\d+)?)\s+(.*)$")


def _srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(text):
    """LC Dialogue / LC Transcribe 'lines' ('0:01.20 - 0:03.40  Alice: hi') -> SRT. Already-SRT text passes through."""
    if re.search(r"\d\d:\d\d:\d\d,\d{3}\s*-->", text):
        return text
    cues = []
    for ln in text.splitlines():
        m = _LINE.match(ln)
        if m:
            a = int(m.group(1)) * 60 + float(m.group(2))
            b = int(m.group(3)) * 60 + float(m.group(4))
            cues.append(f"{len(cues) + 1}\n{_srt_time(a)} --> {_srt_time(b)}\n{m.group(5).strip()}")
    return "\n\n".join(cues)


def _encode(wave, sr, fmt, quality, metadata):
    import av

    if fmt == "wav":
        from .lc_av_media import write_wav

        buf = BytesIO()
        write_wav(buf, wave, sr, bits=24)
        return buf.getvalue()
    if fmt == "opus" and sr not in _OPUS_RATES:
        from .lc_av_media import resample

        new = 48000 if sr > 48000 else next(r for r in _OPUS_RATES if r > sr)
        wave = resample(wave, sr, new)
        sr = new
    wave = np.ascontiguousarray(wave[:2] if wave.shape[0] > 2 else wave, dtype=np.float32)
    layout = "mono" if wave.shape[0] == 1 else "stereo"
    buf = BytesIO()
    out = av.open(buf, mode="w", format={"opus": "opus", "mp3": "mp3", "flac": "flac"}[fmt])
    for k, v in metadata.items():
        out.metadata[k] = v
    codec = {"opus": "libopus", "mp3": "libmp3lame", "flac": "flac"}[fmt]
    st = out.add_stream(codec, rate=sr, layout=layout)
    if fmt in ("mp3", "opus"):
        st.bit_rate = _BITRATE.get(quality, 192000 if fmt == "mp3" else 128000)
    frame = av.AudioFrame.from_ndarray(wave.T.reshape(1, -1), format="flt", layout=layout)
    frame.sample_rate = sr
    frame.pts = 0
    out.mux(st.encode(frame))
    out.mux(st.encode(None))
    out.close()
    return buf.getvalue()


class LCSaveAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/LC",
                                               "tooltip": "Folder/name inside output. %date:yyyy-MM-dd% works."}),
                "format": (["flac", "wav", "mp3", "opus"], {"default": "flac",
                           "tooltip": "flac: lossless and small. wav: 24-bit for editors. mp3 / opus: for sharing."}),
                "quality": (["320k", "256k", "192k", "128k", "96k", "64k"], {"default": "192k",
                            "tooltip": "mp3 and opus only. 192k is plenty for speech."}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip": "Off: preview only, nothing kept."}),
                "embed_workflow": ("BOOLEAN", {"default": True, "tooltip": "Puts the workflow in the file "
                                               "(flac / mp3 / opus): drag it into ComfyUI to load it."}),
                "transcript_file": (["txt", "srt", "txt + srt", "none"], {"default": "txt + srt",
                                    "tooltip": "Saves the transcript next to the audio when one is plugged in. "
                                               "srt = subtitles for video editors."}),
            },
            "optional": {
                "transcript": ("STRING", {"forceInput": True, "tooltip": "LC Dialogue's or LC Transcribe's transcript."}),
                "voice_name": ("STRING", {"forceInput": True, "tooltip": "From LC Voice Shape: also saves the "
                                          "sound as models/voices/<name>.wav, a speaker for LC Dialogue (replaces "
                                          "one with that name). Empty = not saved as a voice."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "saved_as")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Save audio as FLAC, WAV, MP3 or Opus with the workflow inside, plus the transcript as .txt / .srt."

    def save(self, audio, filename_prefix="audio/LC", format="flac", quality="192k", save_output=True,
             embed_workflow=True, transcript_file="txt + srt", transcript=None, voice_name=None, prompt=None,
             extra_pnginfo=None):
        import folder_paths
        from comfy.cli_args import args

        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0:
            raise ValueError("LC Save Audio: there is no sound to save.")
        wave = np.asarray(wave, dtype=np.float32)
        if save_output:
            base, kind, prefix = folder_paths.get_output_directory(), "output", filename_prefix
        else:
            import random

            base, kind = folder_paths.get_temp_directory(), "temp"
            prefix = "LC_temp_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        folder, name, counter, sub, _ = folder_paths.get_save_image_path(prefix, base)
        meta = {}
        if embed_workflow and not args.disable_metadata and format != "wav":
            if prompt is not None:
                meta["prompt"] = json.dumps(prompt)
            for k, v in (extra_pnginfo or {}).items():
                meta[k] = json.dumps(v)
        file = f"{name}_{counter:05}.{format}"
        path = os.path.join(folder, file)
        with open(path, "wb") as f:
            f.write(_encode(wave, sr, format, quality, meta))
        if save_output and transcript and transcript.strip() and transcript_file != "none":
            stem = os.path.splitext(path)[0]
            if "txt" in transcript_file:
                open(stem + ".txt", "w", encoding="utf-8").write(transcript.strip() + "\n")
            if "srt" in transcript_file:
                srt = to_srt(transcript)
                if srt:
                    open(stem + ".srt", "w", encoding="utf-8").write(srt + "\n")
        saved = path if save_output else ""
        if voice_name and voice_name.strip():
            import re

            from .lc_av_media import write_wav

            vname = re.sub(r"[^\w\- ]+", "", voice_name).strip()
            if not vname:
                raise ValueError("LC Save Audio: voice_name needs letters, numbers, spaces, - or _.")
            vdir = os.path.join(folder_paths.models_dir, "voices")
            os.makedirs(vdir, exist_ok=True)
            vpath = os.path.join(vdir, vname + ".wav")
            write_wav(vpath, wave, sr)  # a plain WAV, the way LC Dialogue and the voice nodes read speakers
            saved = (saved + " | " if saved else "") + vpath
        return {"ui": {"audio": [{"filename": file, "subfolder": sub, "type": kind}]},
                "result": (audio, saved)}


NODE_CLASS_MAPPINGS = {"LCSaveAudio": LCSaveAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"LCSaveAudio": "LC Save Audio 💽"}
