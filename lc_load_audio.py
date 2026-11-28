"""
LC Load Audio
-------------
Load a file from Comfy input/output/temp into AUDIO.
Duration output is seconds as FLOAT.
"""

from __future__ import annotations

import os
import re

import numpy as np

import folder_paths

from .lc_av_media import make_audio, with_preview


_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".mp4", ".webm", ".mkv")


def _scan_dir(root):
    found = []
    if not root or not os.path.isdir(root):
        return found
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            if n.lower().endswith(_EXTS):
                rel = os.path.relpath(os.path.join(dirpath, n), root)
                found.append(rel.replace("\\", "/"))
    return found


VOICES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voices")  # the pack's test voices


def _voice_path(name: str):
    """'voices/LC Test Voice.flac' -> the file in the pack's assets/voices, or None."""
    name = (name or "").replace("\\", "/")
    if not name.startswith("voices/"):
        return None
    path = os.path.join(VOICES, name[len("voices/"):])
    return path if os.path.isfile(path) and _inside(path, [VOICES]) else None


def _list_audio():
    files = []
    try:
        files.extend(_scan_dir(folder_paths.get_input_directory()))
    except Exception:
        pass
    try:
        if "audio" in getattr(folder_paths, "folder_names_and_paths", {}):
            files.extend(folder_paths.get_filename_list("audio") or [])
    except Exception:
        pass
    files.extend("voices/" + f for f in _scan_dir(VOICES))  # test voices that come with the pack
    try:  # what LC Save Audio made (output/audio), with its transcript next to it
        out = folder_paths.get_output_directory()
        files.extend(f"audio/{f} [output]" for f in _scan_dir(os.path.join(out, "audio")))
    except Exception:
        pass
    seen = set()
    out = []
    for f in files:
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return sorted(out) or [""]


def _inside(path: str, bases) -> bool:
    """True when path sits in one of the base folders (no ../ or absolute path out of them).
    abspath, not realpath: a symlinked folder inside input/ still counts."""
    real = os.path.abspath(path)
    for base in bases:
        root = os.path.abspath(base)
        try:
            if os.path.commonpath([real, root]) == root:
                return True
        except ValueError:  # another drive
            continue
    return False


def _resolve(name: str) -> str:
    name = (name or "").strip().replace("\\", "/")
    if not name:
        raise ValueError("LC Load Audio: pick a file.")
    if _voice_path(name):
        return _voice_path(name)
    bases = []
    for fn in (
        folder_paths.get_input_directory,
        folder_paths.get_output_directory,
        folder_paths.get_temp_directory,
    ):
        try:
            bases.append(fn())
        except Exception:
            pass
    if hasattr(folder_paths, "exists_annotated_filepath"):
        try:
            if folder_paths.exists_annotated_filepath(name):
                path = folder_paths.get_annotated_filepath(name)
                if _inside(path, bases):
                    return path
        except Exception:
            pass
    # the combo also lists models/audio files
    try:
        if "audio" in getattr(folder_paths, "folder_names_and_paths", {}):
            path = folder_paths.get_full_path("audio", name)
            if path and os.path.isfile(path):
                return path
    except Exception:
        pass
    if os.path.isabs(name) and os.path.isfile(name):  # a full path typed or linked in: worked before, keep it
        return name
    leaf = name.split("/")[-1]
    for base in bases:
        for cand in (os.path.join(base, name), os.path.join(base, leaf)):
            if os.path.isfile(cand) and _inside(cand, bases):
                return cand
    raise FileNotFoundError(f"LC Load Audio: not found: {name}")


_LINE = re.compile(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*-\s*(\d+):(\d+(?:\.\d+)?)\s+(.*)$")
_SRT = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{3})")


def _clock(t):
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:05.2f}"


def _srt_to_lines(text):
    """SRT cues -> the pack's own lines ('0:01.20 - 0:03.40  Alice: hi'), the shape LC Dialogue and LC Transcribe use."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        rows = [r for r in block.strip().splitlines() if r.strip()]
        for i, r in enumerate(rows):
            m = _SRT.search(r)
            if m:
                g = [int(x) for x in m.groups()]
                a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                out.append(f"{_clock(a)} - {_clock(b)}  " + " ".join(rows[i + 1:]).strip())
                break
    return "\n".join(out)


def _transcript(path, start_s, end_s, total):
    """The transcript LC Save Audio saved next to the clip (.txt first, it is the transcript as made; else the .srt).
    A cut keeps the lines inside it, timed from the cut's start. Empty when there is no transcript file."""
    stem = os.path.splitext(path)[0]
    text = ""
    for ext in (".txt", ".srt"):
        if os.path.isfile(stem + ext):
            with open(stem + ext, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if ext == ".srt" or _SRT.search(text):
                text = _srt_to_lines(text)
            break
    text = text.strip()
    if not text or (start_s <= 0.01 and end_s >= total - 0.01):
        return text
    kept = []
    for ln in text.splitlines():
        m = _LINE.match(ln)
        if not m:
            continue
        a = int(m.group(1)) * 60 + float(m.group(2))
        b = int(m.group(3)) * 60 + float(m.group(4))
        if b <= start_s or a >= end_s:
            continue
        kept.append(f"{_clock(a - start_s)} - {_clock(min(b, end_s) - start_s)}  {m.group(5).strip()}")
    return "\n".join(kept)


def _load_file(path: str):
    from .lc_av_media import read_audio

    try:
        return read_audio(path)
    except RuntimeError as e:
        raise RuntimeError(f"LC Load Audio: {e}") from e


class LCLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    _list_audio(),
                    {
                        "tooltip": "File in Comfy input/, plus what LC Save Audio made (output/audio, marked [output]: its transcript comes along), and the pack's test voices (voices/...). Use the Load audio button to upload. Combo lists audio and common video containers.",
                    },
                ),
            },
            "optional": {
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1e7,
                        "step": 0.01,
                        "tooltip": "Start of the cut in seconds. Preview and the AUDIO output both begin here. Drag the left timeline handle after one preview so file length is known.",
                    },
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1e7,
                        "step": 0.01,
                        "tooltip": "End of the cut in seconds. 0 = play through the end of the file. Preview stops here; output duration is end minus start. Drag the right timeline handle.",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, audio, start_time=0.0, end_time=0.0, duration=0.0, **_k):
        try:
            path = _resolve(audio)
            st = os.stat(path)
            return f"{path}:{st.st_mtime_ns}:{st.st_size}:{start_time}:{end_time}:{duration}"
        except Exception:
            return f"{audio}:{start_time}:{end_time}:{duration}"

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "duration", "transcript")
    OUTPUT_TOOLTIPS = ("The audio (the cut, when start / end are set).", "The cut length in seconds.",
                       "The transcript LC Save Audio saved next to the clip (.txt, else .srt), only the lines inside "
                       "the cut. Empty when there is none: run LC Transcribe instead.")
    FUNCTION = "load"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = (
        "Load audio from Comfy input/. start_time and end_time trim the file "
        "(end_time 0 = EOF). Output duration is the cut length in seconds (FLOAT). "
        "preview / pause plays only that cut; drag the timeline handles after the first preview."
    )

    def load(self, audio, start_time=0.0, end_time=0.0, duration=0.0, **_k):
        path = _resolve(audio)
        wave, sr = _load_file(path)
        if wave.ndim == 1:
            wave = wave[None, :]
        total = float(wave.shape[-1]) / float(sr) if sr else 0.0
        start_s = max(0.0, float(start_time or 0.0))
        end_s = float(end_time or 0.0)
        if end_s <= 0 and duration and float(duration) > 0:
            end_s = start_s + float(duration)
        if end_s <= 0 or end_s > total:
            end_s = total
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        start = int(start_s * sr)
        stop = int(end_s * sr)
        wave = wave[:, start:stop]
        sec = float(wave.shape[-1]) / float(sr) if sr else 0.0
        out = make_audio(wave, sr)
        return with_preview((out, sec, _transcript(path, start_s, end_s, total)), out, prefix="lc_load")


NODE_CLASS_MAPPINGS = {
    "LCLoadAudio": LCLoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCLoadAudio": "LC Load Audio 📂",
}


def _register_list_route():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    inst = getattr(PromptServer, "instance", None)
    if inst is None:
        return

    async def _audio_files(_request):
        return web.json_response({"files": _list_audio()})

    async def _voice_file(request):  # preview of a pack test voice (only files inside assets/voices)
        path = _voice_path("voices/" + request.query.get("name", ""))
        if not path:
            return web.Response(status=404)
        return web.FileResponse(path)

    try:
        inst.routes.get("/lc_av/audio_files")(_audio_files)
        inst.routes.get("/lc_av/voice_file")(_voice_file)
    except Exception as e:
        print(f"[LC AV] audio list route: {e}")


_register_list_route()
