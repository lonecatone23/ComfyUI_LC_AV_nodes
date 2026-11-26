"""
LC Dialogue
-----------
Turns a script into one spoken track. Every line names its speaker and can carry its own emotion mix; an emotion tag
inside a line switches the feeling mid-sentence. The speech engine plugs in (LC IndexTTS Engine today), so the script
format stays the same when other engines are added.

    [Alice|angry:0.6,sad:0.3] You're late again.
    [Bob|sad:0.4,arousal:-0.5] I know. [pause:400ms] I know.
    [Alice|sad] I'm fine. [angry:0.9] I said I'm fine!

Tags: any emotion the engine knows (weight 0..1.2, default 1), valence / arousal (-1..1), strength (overall), rate (0..4
or slow / normal / fast), pause (600ms or 1.2s). [neutral] clears the emotion. Voices: LC Load Voice wired into voices (a clip named for the script), or models/voices/<speaker>.wav
(also .flac / .mp3 / .ogg), the same folder TTS-Audio-Suite uses.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from .lc_loudness import gain_to_lufs
from .lc_tts_engine import ENGINE_TYPE

_AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
_RATES = {"slow": 0, "normal": None, "fast": 3}

_cache = {}  # rendered pieces: editing one line only re-renders that line


def _voice_dirs():
    import folder_paths

    dirs = []
    try:
        dirs += folder_paths.get_folder_paths("voices")
    except Exception:
        pass
    dirs += [os.path.join(folder_paths.models_dir, "voices"), os.path.join(folder_paths.get_input_directory(), "voices")]
    dirs.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voices"))  # LC Test Voice, last
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


def _voice_files():
    out = []
    for d in _voice_dirs():
        for root, _, files in os.walk(d):
            out += [os.path.relpath(os.path.join(root, f), d).replace("\\", "/") for f in files
                    if f.lower().endswith(_AUDIO_EXT)]
    return sorted(dict.fromkeys(out), key=str.lower)


def _find_voice(name):
    want = name.strip().lower()
    for d in _voice_dirs():
        for root, _, files in os.walk(d):
            for f in files:
                stem, ext = os.path.splitext(f)
                rel = os.path.relpath(os.path.join(root, f), d).replace("\\", "/").lower()
                if ext.lower() in _AUDIO_EXT and (stem.lower() == want or rel == want):
                    return os.path.join(root, f)
    return None


def _load_audio(path):
    import torch

    from .lc_av_media import read_audio

    a, sr = read_audio(path)
    return {"waveform": torch.from_numpy(np.ascontiguousarray(a))[None], "sample_rate": int(sr)}


PREMADE_EMOTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "emotions")


def user_emotions_dir():
    """Where LC Emotion Mix saves: <user dir>/LC_AV/emotions, one small file per emotion."""
    import folder_paths

    return os.path.join(folder_paths.get_user_directory(), "LC_AV", "emotions")


def mix_of(mix, valence=0.0, arousal=0.0):
    """'sad:0.5, angry:0.3' (+ the pad) -> {emotion: weight, valence, arousal}."""
    out = {}
    for item in [x.strip() for x in re.split(r"[,\n]", mix or "") if x.strip()]:
        k, _, v = item.partition(":")
        out[k.strip().lower()] = float(v) if v.strip() else 1.0
    if valence:
        out["valence"] = round(float(valence), 2)
    if arousal:
        out["arousal"] = round(float(arousal), 2)
    return out


def saved_emotions():
    """{lower name: {name, mix, valence, arousal, premade}}: your saved emotions, then the premades that ship with the
    pack (yours win on the same name)."""
    out = {}
    for d, premade in ((user_emotions_dir(), False), (PREMADE_EMOTIONS, True)):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".json"):
                continue
            try:
                e = json.load(open(os.path.join(d, f), encoding="utf-8"))
                name = str(e.get("name") or os.path.splitext(f)[0]).strip()
                if name.lower() not in out:
                    out[name.lower()] = {"name": name, "mix": str(e.get("mix") or ""),
                                         "valence": float(e.get("valence") or 0), "arousal": float(e.get("arousal") or 0),
                                         "premade": premade}
            except Exception as ex:
                print(f"[LC Dialogue] could not read emotion {f}: {ex}")
    return out


def _presets():
    """Saved emotions as script mixes: {lower name: {emotion: weight, valence, arousal}}."""
    return {k: mix_of(e["mix"], e["valence"], e["arousal"]) for k, e in saved_emotions().items()}


def _seconds(v):
    v = v.strip().lower()
    if v.endswith("ms"):
        return float(v[:-2]) / 1000
    return float(v.rstrip("s"))


def _parse(script, known):
    """-> list of lines: {speaker, pieces: [(text, mix, strength, rate)], pause_before}.
    A saved emotion (LC Emotion Mix): [Haley|bitter] uses 'Haley bitter' when there is one, else 'bitter'."""
    known = {k.lower() for k in known}
    presets = _presets()
    tag_re = re.compile(r"\[([^\]]*)\]")
    lines, speaker = [], None

    def saved_for(k):
        own = f"{(speaker or '').strip().lower()} {k}".strip()
        if speaker and own in presets:
            return own
        return k if k in presets else None

    def tags_of(body, mix, strength, rate):
        for item in [x.strip() for x in re.split(r"[,|]", body) if x.strip()]:
            k, _, v = item.partition(":")
            k = k.strip().lower()
            s = saved_for(k) if k not in ("neutral", "none", "strength", "rate", "valence", "arousal") else None
            if k in ("neutral", "none"):
                mix = {}
            elif s:  # a saved emotion ('Haley bitter' for Haley first), scaled by its weight
                w = float(v) if v.strip() else 1.0
                mix = {e: x * w for e, x in presets[s].items()}
            elif k in known or k in ("valence", "arousal"):
                mix = dict(mix)
                mix[k] = float(v) if v.strip() else 1.0
            elif k == "strength":
                strength = float(v)
            elif k == "rate":
                rate = _RATES[v.strip().lower()] if v.strip().lower() in _RATES else int(v)
            else:
                raise ValueError(f"LC Dialogue: unknown tag '{item}'. This engine knows: "
                                 f"{', '.join(sorted(known | set(presets)))}, valence, arousal, strength, rate, pause, neutral.")
        return mix, strength, rate

    def is_tags(body):
        first = re.split(r"[,|:]", body, 1)[0].strip().lower()
        return first in known or first in presets or first in ("valence", "arousal", "strength", "rate", "neutral", "none", "pause")

    for raw in script.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        mix, strength, rate, pause = {}, 1.0, None, 0.0
        m = re.match(r"^\[([^\]]*)\]\s*", raw)
        if m and not is_tags(m.group(1)):
            head = m.group(1).split("|", 1)
            speaker = head[0].strip()
            if len(head) > 1:
                mix, strength, rate = tags_of(head[1], mix, strength, rate)
            raw = raw[m.end():]
        pieces, pos, buf = [], 0, ""
        for t in tag_re.finditer(raw):
            buf += raw[pos:t.start()]
            pos = t.end()
            body = t.group(1)
            if body.strip().lower().startswith("pause"):
                if buf.strip():
                    pieces.append((buf.strip(), mix, strength, rate))
                buf = ""
                pieces.append(("", None, _seconds(body.split(":", 1)[1]), None))  # a pause inside the line
                continue
            if buf.strip():
                pieces.append((buf.strip(), mix, strength, rate))
            buf = ""
            mix, strength, rate = tags_of(body, {}, 1.0, rate)  # a tag inside a line switches the feeling
        buf += raw[pos:]
        if buf.strip():
            pieces.append((buf.strip(), mix, strength, rate))
        if pieces and all(p[1] is None for p in pieces) and lines:  # a line that is only [pause:x]
            lines[-1]["pause_after"] += sum(p[2] for p in pieces)
            continue
        if pieces:
            lines.append({"speaker": speaker, "pieces": pieces, "pause_after": pause})
    return lines


def _clock(sec):
    m, s = divmod(float(sec), 60)
    return f"{int(m)}:{s:05.2f}"


def _trim(w, sr, floor_db=-45.0, pad=0.03):
    """Cut leading / trailing silence, keeping a short pad so words don't clip."""
    level = np.abs(w).max(0)
    on = np.nonzero(level > 10 ** (floor_db / 20) * max(1e-4, float(level.max())))[0]
    if on.size == 0:
        return w
    a, b = max(0, on[0] - int(pad * sr)), min(w.shape[-1], on[-1] + int(pad * sr))
    return w[:, a:b]


def _fit(w, sr, target_sr, chans):
    if sr != target_sr:
        from .lc_av_media import resample

        w = resample(w, sr, target_sr)
    if w.shape[0] < chans:
        w = np.repeat(w[:1], chans, 0)
    return w[:chans]


class LCDialogue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (ENGINE_TYPE, {"tooltip": "A speech engine (LC IndexTTS Engine)."}),
                "script": ("STRING", {"multiline": True, "default":
                    "[Alice|angry:0.6,sad:0.3] You're late again. I made dinner hours ago.\n"
                    "[Bob|sad:0.4,arousal:-0.5] I know. [pause:400ms] I know. The meeting ran over.\n"
                    "[Alice|sad] I'm fine. [angry:0.9] I said I'm fine!",
                    "tooltip": "[Speaker|emotion:weight,...] line. [Haley|bitter] uses the saved emotion 'Haley "
                               "bitter' (LC Emotion Mix) when there is one, then 'bitter'. An emotion tag inside a line switches the feeling from "
                               "there. [pause:600ms] adds a pause. Lines starting with # are skipped."}),
                "default_voice": (["(none)"] + _voice_files(), {"tooltip": "Voice for lines with no [Speaker], and "
                                  "for speakers with no clip in models/voices."}),
                "gap": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0, "step": 0.05,
                                  "tooltip": "Silence between lines (seconds)."}),
                "seed": ("INT", {"default": 8675309, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": "fixed",
                                 "tooltip": "Each piece uses seed + its number, so one line can be redone alone."}),
                "even_lines": ("BOOLEAN", {"default": True,
                                           "tooltip": "Brings every line to the same loudness before joining, so "
                                                      "a shouted line is not twice as loud as a sad one. Finish "
                                                      "with LC Audio Leveler for the final level."}),
            },
            "optional": {"voices": ("LC_VOICES", {"tooltip": "The cast from LC Load Voice: every voice on one wire, "
                                                  "each with its script name. Beats a file with the same name in "
                                                  "models/voices."})},
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "timings", "info", "transcript")
    FUNCTION = "speak"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = ("Speaks a script: a voice per speaker (models/voices/<name>.wav), an emotion mix per line, and "
                   "emotion changes mid-line. transcript = who says what, when; timings = the same as JSON.")

    def speak(self, engine, script, default_voice="(none)", gap=0.35, seed=0, even_lines=True, **wired):
        import hashlib

        import comfy.model_management as mm
        import comfy.utils

        from .lc_voice_load import MAX_SECONDS, cap

        # the LC Load Voice cast: it wins over files with the same name
        given = {}
        for key, v in (wired.get("voices") or {}).items():
            clip = v.get("audio") if isinstance(v, dict) else None
            if clip is None or getattr(clip.get("waveform"), "numel", lambda: 0)() == 0:
                continue
            wv = clip["waveform"]
            tag = hashlib.sha1(wv.detach().float().cpu().numpy().tobytes()[:4_000_000]).hexdigest()[:16]
            given[str(key).lower()] = (clip, f"wired:{tag}:{int(clip['sample_rate'])}:{wv.shape[-1]}")
        lines = _parse(script, engine.emotions())
        if not lines:
            raise ValueError("LC Dialogue: the script has no lines.")
        voices, clips, missing = {}, {}, []
        for ln in lines:
            name = ln["speaker"]
            if name in voices:
                continue
            key = (name or "").strip().lower()
            if key in given:
                clips[name], voices[name] = given[key]
                continue
            path = _find_voice(name) if name else None
            if path is None and default_voice != "(none)":
                path = _find_voice(default_voice)
            if path is None:
                missing.append(name or "(no speaker)")
            voices[name] = path
        if missing:
            raise ValueError(f"LC Dialogue: no voice clip for {', '.join(missing)}. Wire LC Load Voice into voices with that "
                             f"name, put <name>.wav in "
                             f"{(_voice_dirs() or ['models/voices'])[0]}, or pick a default_voice.")
        for n, p in voices.items():
            if n not in clips:
                clips[n] = cap(_load_audio(p), MAX_SECONDS)  # long references trip the engine up
        jobs = sum(1 for ln in lines for p in ln["pieces"] if p[1] is not None)
        bar = comfy.utils.ProgressBar(jobs)
        out, timings, sr, chans, n, idx, fresh = [], [], None, 1, 0, 0, 0
        for li, ln in enumerate(lines):
            start = None
            for text, mix, strength, rate in ln["pieces"]:
                if mix is None:  # a pause inside the line
                    pad = np.zeros((chans, int(round(strength * (sr or 24000)))), np.float32)
                    out.append(pad)
                    n += pad.shape[-1]
                    continue
                mm.throw_exception_if_processing_interrupted()
                key = (id(engine), text, voices[ln["speaker"]], tuple(sorted(mix.items())), strength, rate, seed + idx)
                if key not in _cache:
                    _cache[key] = engine.speak(text, clips[ln["speaker"]], mix, strength, seed + idx, rate)
                    fresh += 1
                w, wsr = _cache[key]
                idx += 1
                bar.update(1)
                if sr is None:
                    sr, chans = wsr, w.shape[0]
                w = _trim(_fit(w, wsr, sr, chans), sr)
                if even_lines:
                    w = w * np.float32(10 ** (gain_to_lufs(w, sr, -20.0, 12.0) / 20))
                if start is None:
                    start = n / sr
                out.append(w)
                n += w.shape[-1]
                tail = np.zeros((chans, int(0.05 * sr)), np.float32)  # breath between pieces of one line
                out.append(tail)
                n += tail.shape[-1]
            timings.append({"line": li + 1, "speaker": ln["speaker"] or "", "start": round(start or 0, 2),
                            "end": round(n / sr, 2),
                            "text": " ".join(p[0] for p in ln["pieces"] if p[1] is not None)})
            if li < len(lines) - 1:
                pad = np.zeros((chans, int(round((gap + ln["pause_after"]) * sr))), np.float32)
                out.append(pad)
                n += pad.shape[-1]
        if len(_cache) > 400:
            for k in list(_cache)[:200]:
                _cache.pop(k, None)
        import torch

        wave = np.concatenate(out, axis=-1)
        transcript = "\n".join(f"{_clock(t['start'])} - {_clock(t['end'])}  {t['speaker'] or '(no speaker)'}: {t['text']}"
                               for t in timings)
        info = (f"{engine.name}: {len(lines)} lines, {idx} pieces ({fresh} rendered, {idx - fresh} reused), "
                f"{wave.shape[-1] / sr:.1f} s, {len({(ln['speaker'] or '').strip().lower() for ln in lines})} speakers.")
        return ({"waveform": torch.from_numpy(wave)[None], "sample_rate": sr}, json.dumps(timings, indent=1), info, transcript)


NODE_CLASS_MAPPINGS = {"LCDialogue": LCDialogue}
NODE_DISPLAY_NAME_MAPPINGS = {"LCDialogue": "LC Dialogue 🗣️"}
