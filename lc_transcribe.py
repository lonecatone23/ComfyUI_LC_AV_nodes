"""
LC Transcribe
-------------
Timed transcript of any audio: what was said, and when. OpenAI Whisper through transformers (part of every ComfyUI
install), with word timings. The model downloads to models/stt/<name> on first use; any other Whisper folder in
models/stt (Hugging Face format, with config.json) shows in the list too. Nothing extra to pip install.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from .lc_av_media import audio_parts, resample

# name -> (Hugging Face repo, size for the first-use toast)
_BUILTIN = {
    "whisper-small": ("openai/whisper-small", "970 MB"),
    "whisper-large-v3-turbo": ("openai/whisper-large-v3-turbo", "1.6 GB"),
}
_FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
          "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json",
          "special_tokens_map.json", "model.safetensors")
_OPTIONAL = {"normalizer.json", "added_tokens.json", "special_tokens_map.json", "tokenizer.json"}
_models = {}


def _stt_dir():
    import folder_paths

    return os.path.join(folder_paths.models_dir, "stt")


def _local_names():
    d = _stt_dir()
    if not os.path.isdir(d):
        return []
    return [n for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n, "config.json"))
            and os.path.isfile(os.path.join(d, n, "model.safetensors"))]


def _model_names():
    return list(_BUILTIN) + [n for n in _local_names() if n not in _BUILTIN]


def _clock(sec, srt=False):
    h, rem = divmod(float(sec), 3600)
    m, s = divmod(rem, 60)
    if srt:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"
    return f"{int(m + 60 * h)}:{s:05.2f}"


def _fetch(name):
    """models/stt/<name>, downloading a built-in model's files on first use."""
    path = os.path.join(_stt_dir(), name)
    if name not in _BUILTIN or os.path.isfile(os.path.join(path, "model.safetensors")):
        return path
    import urllib.error

    from .lc_first_download import download, hf_url, toast

    repo, size = _BUILTIN[name]
    toast(f"Whisper ({name})", size)
    for f in _FILES:
        dest = os.path.join(path, f)
        if os.path.isfile(dest):
            continue
        try:
            download(hf_url(repo, f), dest, f"Whisper {name} {f}", size if f == "model.safetensors" else "",
                     announce=False)
        except urllib.error.HTTPError as e:
            if e.code == 404 and f in _OPTIONAL:
                continue
            raise
    return path


def _device(device):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load(name, device):
    import torch

    device = _device(device)
    key = (name, device)
    if key not in _models:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

        path = _fetch(name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        proc = WhisperProcessor.from_pretrained(path)
        model = WhisperForConditionalGeneration.from_pretrained(path, dtype=dtype).to(device).eval()
        _models.clear()  # one Whisper in memory at a time
        _models[key] = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                                feature_extractor=proc.feature_extractor, device=device, dtype=dtype)
    return _models[key]


def words(x16, name, language="en", device="auto"):
    """16 kHz mono float32 -> [{"word", "start", "end"}] with Whisper's word timings."""
    asr = _load(name, device)
    kw = {"task": "transcribe"}
    if language and language.strip():
        kw["language"] = language.strip()
    out = asr({"raw": np.ascontiguousarray(x16, dtype=np.float32), "sampling_rate": 16000},
              return_timestamps="word", generate_kwargs=kw)  # no chunk_length_s: Whisper's own long-form mode
    dur = len(x16) / 16000.0
    res = []
    for c in out.get("chunks") or []:
        t = c.get("text", "").strip()
        a, b = c.get("timestamp") or (None, None)
        if not t or a is None:
            continue
        b = a if b is None else b
        res.append({"word": t, "start": round(min(float(a), dur), 2), "end": round(min(float(b), dur), 2)})
    return res


def segments(ws, gap=0.8, max_words=24):
    """Words -> sentence-like segments [{"start", "end", "text"}]: split after . ? ! or on a pause."""
    segs, cur = [], []
    for w in ws:
        if cur and (w["start"] - cur[-1]["end"] > gap or len(cur) >= max_words):
            segs.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.?!]['\")\]]*$", w["word"]):
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return [{"start": s[0]["start"], "end": s[-1]["end"], "text": " ".join(w["word"] for w in s)} for s in segs]


class LCTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        names = _model_names()
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (names, {"default": names[0],
                                  "tooltip": "Downloads to models/stt on first use. whisper-small (970 MB) is quick; "
                                             "large-v3-turbo (1.6 GB) is more accurate. Other Whisper folders in "
                                             "models/stt show here too."}),
                "language": ("STRING", {"default": "en", "tooltip": "Two-letter code, or empty to detect it."}),
                "format": (["lines", "srt", "H3 prompt"], {"default": "lines",
                           "tooltip": "lines: 0:01.20 - 0:03.40  text. srt: subtitles. H3 prompt: 'At 1.2 seconds "
                                      "they say, \"...\"' sentences to paste into a timed prompt."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "auto = the GPU when there is one."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("transcript", "words_json")
    FUNCTION = "transcribe"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = "What was said, and when: a timed transcript (lines, subtitles or H3 timed-prompt sentences) plus every word's time."

    def transcribe(self, audio, model, language="en", format="lines", device="auto"):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0:
            return ("", "[]")
        x16 = resample(np.asarray(wave, dtype=np.float32).mean(0), sr, 16000)
        ws = words(x16, model, language, device)
        segs = segments(ws)
        if format == "srt":
            text = "\n\n".join(f"{i + 1}\n{_clock(s['start'], True)} --> {_clock(s['end'], True)}\n{s['text']}"
                               for i, s in enumerate(segs))
        elif format == "H3 prompt":
            text = " ".join(f"At {s['start']:.1f} seconds they say, \"{s['text']}\"" for s in segs)
        else:
            text = "\n".join(f"{_clock(s['start'])} - {_clock(s['end'])}  {s['text']}" for s in segs)
        return (text, json.dumps(ws))


NODE_CLASS_MAPPINGS = {"LCTranscribe": LCTranscribe}
NODE_DISPLAY_NAME_MAPPINGS = {"LCTranscribe": "LC Transcribe 📝"}
