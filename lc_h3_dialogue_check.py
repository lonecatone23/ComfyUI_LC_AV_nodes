"""
LC H3 Dialogue Check (BETA)
---------------------------
H3 sometimes renders speech as gibberish that sounds like talking. This node listens to the finished clip with
Whisper, writes down what was actually said, and compares it with the dialogue in your prompt (the text in quotes).
A low match means garbled speech: redo the shot. It also lists the pauses, which are the clean places to cut or
start the next segment, and gives the transcript back as timed prompt lines ("At 0 seconds they say, ..."), so a
recording you sync with LC H3 Audio Sync can carry its words and their timing into the prompt: H3 moves the mouth
when the prompt says the words are spoken.

Uses faster-whisper (already a ComfyUI dependency for many packs). The model downloads once on first use to
models/stt/ (small = about 460 MB). Idea from Deno2026's Whisper nodes (ideas only, written fresh for LC_AV).
"""
from __future__ import annotations

import difflib
import os
import re

import torch

import folder_paths

SIZES = ["tiny", "base", "small", "medium", "large-v3"]
_models = {}


def _model(size):
    from faster_whisper import WhisperModel, download_model
    if size in _models:
        return _models[size]
    root = os.path.join(folder_paths.models_dir, "stt", f"faster-whisper-{size}")
    if not os.path.exists(os.path.join(root, "model.bin")):
        print(f"[LC_AV] Dialogue Check: downloading Whisper {size} to {root} (one time)")
        download_model(size, output_dir=root)
    # CPU on purpose: a clip takes seconds there, it leaves the GPU to H3, and faster-whisper's GPU build wants CUDA 12
    # libraries (cublas64_12.dll) that a CUDA 13 ComfyUI doesn't ship
    m = WhisperModel(root, device="cpu", compute_type="int8")
    _models.clear()
    _models[size] = m
    return m


def quoted(prompt):
    """The dialogue in a prompt: everything inside straight or curly double quotes, or H3's <d>[English] ... </d>
    tags (the layout LC H3 Segment writes)."""
    parts = re.findall(r'<d>\s*(?:\[[^\]]*\])?\s*(.*?)\s*</d>|"([^"]+)"|“([^”]+)”', prompt or "", re.S)
    return " ".join(a or b or c for a, b, c in parts).strip()


def _words(text):
    return re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'"))  # curly We’ll = we'll


def match(expected, heard):
    """0-1 word-level similarity (1 = every word in order)."""
    a, b = _words(expected), _words(heard)
    if not a:
        return 1.0 if not b else 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


class LCH3DialogueCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The clip's sound (from the H3 audio decode or a loaded video)."}),
                "model_size": (SIZES, {"default": "small",
                                       "tooltip": "Whisper model. small is the sweet spot for English dialogue "
                                                  "(about 460 MB, downloads once to models/stt/)."}),
                "min_match": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Below this the speech counts as garbled. 1.0 = every word heard "
                                                   "in order, 0.6 = most of the words, 0.3 or less = gibberish."}),
                "min_pause": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 5.0, "step": 0.1,
                                        "tooltip": "Silence at least this long (seconds) is listed as a cut point."}),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True,
                                      "tooltip": "The positive prompt. The text in quotes is the dialogue it should say."}),
                "language": ("STRING", {"default": "en", "tooltip": "Language code (en, es, fr...). Blank = detect."}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "BOOLEAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcript", "match", "passed", "cut_points", "prompt_line", "report")
    OUTPUT_TOOLTIPS = (
        "What Whisper heard.",
        "0-1 word match against the quoted dialogue in the prompt (1.0 when no prompt is wired).",
        "False when the match is below min_match: redo the shot.",
        "Pause times in seconds, comma separated: clean places to cut or start the next segment.",
        'The transcript as timed prompt lines: At 0 seconds they say, "...". Paste it into the prompt and change "they" to the character.',
        "A short readable summary.",
    )
    FUNCTION = "run"
    CATEGORY = "LC_AV/H3"
    OUTPUT_NODE = True
    DESCRIPTION = ("Listens to an H3 clip with Whisper: what was said, how well it matches the quoted dialogue in the "
                   "prompt (garbled speech check), the pauses to cut on, and the transcript as a prompt line.")

    def run(self, audio, model_size, min_match, min_pause, prompt="", language="en"):
        import torchaudio
        wav = audio["waveform"]
        if wav.dim() == 3:
            wav = wav[0]
        mono = wav.float().mean(0)
        sr = int(audio["sample_rate"])
        if sr != 16000:
            mono = torchaudio.functional.resample(mono, sr, 16000)
        segs, _info = _model(model_size).transcribe(mono.cpu().numpy(), language=(language or "").strip() or None,
                                                     word_timestamps=True, vad_filter=True, beam_size=5)  # VAD: no "Thank you" made up from near-silence
        segs = list(segs)
        words = [w for s in segs for w in (s.words or []) if re.search(r"\w", w.word)]  # "..." is Whisper for no speech
        heard = " ".join(w.word.strip() for w in words).strip()
        cuts = []
        for a, b in zip(words, words[1:]):
            if b.start - a.end >= min_pause:
                cuts.append(round((a.end + b.start) / 2, 2))
        expected = quoted(prompt)
        score = match(expected, heard) if expected else 1.0
        passed = score >= min_match
        # timed, the way H3 was trained ("At 2 seconds she says, ..."): H3 moves the mouth when the prompt says the
        # words are spoken, so a synced track needs its lines placed in time
        spoken = [(s.start, " ".join(w.word.strip() for w in (s.words or []) if re.search(r"\w", w.word)).strip()) for s in segs]
        line = " ".join(f'At {round(t)} seconds they say, "{txt}"' for t, txt in spoken if txt)
        report = (f"{'PASSED' if passed else 'GARBLED? redo the shot'}  match {score:.2f} (needs {min_match:.2f})\n"
                  f"Heard: {heard or '(no speech)'}\n"
                  + (f"Expected: {expected}\n" if expected else "No quoted dialogue in the prompt, nothing to match.\n")
                  + f"Cut points (s): {', '.join(map(str, cuts)) or 'none'}")
        return {"ui": {"text": [report]},
                "result": (heard, float(score), passed, ", ".join(map(str, cuts)), line, report)}


NODE_CLASS_MAPPINGS = {"LCH3DialogueCheck": LCH3DialogueCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"LCH3DialogueCheck": "LC H3 Dialogue Check 🗣️ (BETA)"}
