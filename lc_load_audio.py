"""
LC Load Audio
-------------
Load a file from Comfy input/output/temp (or an absolute path) into AUDIO.
Duration output is seconds as FLOAT.
"""

from __future__ import annotations

import os

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
    seen = set()
    out = []
    for f in files:
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return sorted(out) or [""]


def _resolve(name: str) -> str:
    name = (name or "").strip().replace("\\", "/")
    if not name:
        raise ValueError("LC Load Audio: pick a file.")
    if os.path.isfile(name):
        return name
    if hasattr(folder_paths, "exists_annotated_filepath"):
        try:
            if folder_paths.exists_annotated_filepath(name):
                return folder_paths.get_annotated_filepath(name)
        except Exception:
            pass
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
    leaf = name.split("/")[-1]
    for base in bases:
        for cand in (os.path.join(base, name), os.path.join(base, leaf)):
            if os.path.isfile(cand):
                return cand
    raise FileNotFoundError(f"LC Load Audio: not found: {name}")


def _load_file(path: str):
    errors = []
    try:
        import torchaudio

        wave, sr = torchaudio.load(path)
        arr = wave.detach().cpu().numpy().astype(np.float32)
        return arr, int(sr)
    except Exception as e:
        errors.append(f"torchaudio: {e}")
    try:
        import soundfile as sf

        arr, sr = sf.read(path, always_2d=True)
        return arr.T.astype(np.float32), int(sr)
    except Exception as e:
        errors.append(f"soundfile: {e}")
    try:
        import wave as wavmod

        with wavmod.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"wav width {sw}")
        data = data.reshape(-1, ch).T if ch > 1 else data[None, :]
        return data, int(sr)
    except Exception as e:
        errors.append(f"wav: {e}")
    raise RuntimeError(
        f"LC Load Audio: could not decode {path}. ({'; '.join(errors)})"
    )


class LCLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    _list_audio(),
                    {
                        "tooltip": "File in Comfy input/ (also checks output/ and temp/). Use the Load audio button to upload. Combo lists audio and common video containers.",
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

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration")
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
        return with_preview((out, sec), out, prefix="lc_load")


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

    try:
        inst.routes.get("/lc_av/audio_files")(_audio_files)
    except Exception as e:
        print(f"[LC AV] audio list route: {e}")


_register_list_route()
