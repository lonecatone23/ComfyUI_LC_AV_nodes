"""
LC TTS engines: one small interface that LC Dialogue talks to, so speech engines plug in without LC Dialogue knowing
their details. Each engine checks its own requirements only when it runs; a missing engine never stops LC AV loading.

An engine is an object with:
    name                      shown in LC Dialogue's info
    emotions()                emotion names it understands (the script's [Alice|angry:0.6] tags)
    speak(text, voice, mix, strength, seed, rate) -> (waveform np.float32 [channels, samples], sample_rate)
        voice     AUDIO dict (the speaker's reference clip) or None
        mix       {emotion: weight}, plus "valence" / "arousal" (turned into the engine's own emotions)
        strength  overall emotion strength (1.0 = as written)
        rate      None or 0..4 (slow .. fast)
    speaker_vector(audio)     a unit voice-print vector (LC Voice Match)

Engines so far: IndexTTS-2, built in (vendor/indextts via lc_indextts.py; code and model under bilibili's IndexTTS
license, see vendor/indextts/UPSTREAM.md). No other node pack needed.
"""

from __future__ import annotations

import math

import numpy as np

ENGINE_TYPE = "LC_TTS_ENGINE"

INDEX_EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
MAX_WEIGHT = 1.2  # IndexTTS-2's own ceiling per emotion
# other names people write, so old scripts keep working
ALIASES = {"fear": "afraid", "scared": "afraid", "fearful": "afraid", "disgust": "disgusted", "surprise": "surprised",
           "anger": "angry", "mad": "angry", "joy": "happy", "sadness": "sad", "excited": "happy", "tired": "melancholic",
           "bitter": "disgusted", "relaxed": "calm"}
# where each emotion sits on the valence / arousal pad (same spots LC Emotion Mix draws)
SPOTS = {"happy": (0.65, 0.35), "surprised": (0.3, 0.75), "angry": (-0.45, 0.65), "afraid": (-0.75, 0.3),
         "disgusted": (-0.6, 0.05), "sad": (-0.6, -0.5), "melancholic": (-0.25, -0.8), "calm": (0.45, -0.6)}
RATE_SPEED = (0.8, 0.9, 1.0, 1.1, 1.2)  # rate 0..4


def pad_to_mix(valence, arousal):
    """A spot on the valence / arousal pad -> IndexTTS emotion weights. The nearest emotions by direction take most of
    it; the distance from the middle is how strong."""
    v, a = float(valence), float(arousal)
    r = min(1.0, math.hypot(v, a))
    if r < 0.05:
        return {}
    out = {}
    for name, (sv, sa) in SPOTS.items():
        cos = (v * sv + a * sa) / (math.hypot(v, a) * math.hypot(sv, sa))
        if cos > 0.5:
            out[name] = ((cos - 0.5) / 0.5) ** 2
    top = max(out.values()) if out else 1.0
    return {k: round(r * 0.9 * x / top, 3) for k, x in out.items() if x / top > 0.15}


def to_index_mix(mix, strength=1.0):
    """LC Dialogue's mix (aliases, valence / arousal) -> the 8 IndexTTS weights, each 0..1.2."""
    mix = dict(mix or {})
    pad = pad_to_mix(mix.pop("valence", 0.0), mix.pop("arousal", 0.0))
    out = {}
    for k, w in list(mix.items()) + list(pad.items()):
        k = ALIASES.get(k.lower(), k.lower())
        if k in INDEX_EMOTIONS:
            out[k] = out.get(k, 0.0) + float(w)
    return {k: max(0.0, min(MAX_WEIGHT, w * float(strength))) for k, w in out.items() if w * float(strength) > 0}


class IndexEngine:
    name = "IndexTTS-2"

    def __init__(self, settings):
        self.settings = settings  # model, emotion_alpha, temperature, low_vram

    def emotions(self):
        return list(INDEX_EMOTIONS) + sorted(ALIASES)

    def speak(self, text, voice, mix, strength=1.0, seed=0, rate=None):
        from . import lc_indextts

        if voice is None:
            raise ValueError("IndexTTS-2 clones a voice: give every speaker a voice clip.")
        weights = to_index_mix(mix, strength)
        text = text.replace("[", "(").replace("]", ")").strip()  # the text TTS-Audio-Suite used to get (and strip)
        if not text:
            raise ValueError("IndexTTS-2 got an empty line to speak.")
        s = self.settings
        y, sr = lc_indextts.synthesize(text, voice, emotion=weights or None, emotion_alpha=s["emotion_alpha"],
                                       temperature=s["temperature"], seed=int(seed) % (2 ** 32) or 1,
                                       model=s["model"], low_vram=s["low_vram"])
        x = y[None].astype(np.float32)
        if rate is not None and RATE_SPEED[max(0, min(4, int(rate)))] != 1.0:
            from . import lc_dsp as librosa_lite

            speed = RATE_SPEED[max(0, min(4, int(rate)))]
            x = np.stack([librosa_lite.time_stretch(c, rate=speed, n_fft=2048) for c in x]).astype(np.float32)
        return x, sr

    def speak_with(self, text, voice, emotion_text=None, emotion_audio=None, seed=1, rate=None, strength=None):
        """Text emotion (IndexTTS-2's Qwen reader; True = the line itself) or an emotion reference clip (an LC Emotion
        Mix clip like 'Haley bitter'). strength scales how much of the clip's feeling is copied (1 = the engine's)."""
        from . import lc_indextts

        s = self.settings
        alpha = s["emotion_alpha"] * (1.0 if strength is None else float(strength))
        text = text.replace("[", "(").replace("]", ")").strip()
        y, sr = lc_indextts.synthesize(text, voice, emotion_text=emotion_text, emotion_audio=emotion_audio,
                                       emotion_alpha=max(0.0, min(1.0, alpha)), temperature=s["temperature"],
                                       seed=int(seed) % (2 ** 32) or 1, model=s["model"], low_vram=s["low_vram"])
        x = y[None].astype(np.float32)
        if rate is not None and RATE_SPEED[max(0, min(4, int(rate)))] != 1.0:
            from . import lc_dsp as librosa_lite

            speed = RATE_SPEED[max(0, min(4, int(rate)))]
            x = np.stack([librosa_lite.time_stretch(c, rate=speed, n_fft=2048) for c in x]).astype(np.float32)
        return x, sr

    def speaker_vector(self, audio):
        """CAM++ (IndexTTS-2's own speaker model, Apache-2.0 3D-Speaker) voice print, unit length."""
        from . import lc_indextts

        return lc_indextts.speaker_vector(audio)


class LCIndexTTSEngine:
    @classmethod
    def INPUT_TYPES(cls):
        from .lc_indextts import model_choices

        models = model_choices()
        default = models[0]
        return {"required": {
            "model": (models, {"default": default, "tooltip": "IndexTTS-2 is the one we benched: it keeps the cloned "
                                                             "voice the same person under emotion."}),
            "emotion_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "How much of the emotion mix comes through. Lower it if strong "
                                                   "mixes start to slur the words."}),
            "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05,
                                      "tooltip": "0.8 is the model's own setting. Higher = more varied reads, more "
                                                 "mistakes."}),
            "low_vram": ("BOOLEAN", {"default": False, "tooltip": "Moves parts of the model off the GPU between steps."}),
        }}

    RETURN_TYPES = (ENGINE_TYPE,)
    RETURN_NAMES = ("engine",)
    FUNCTION = "load"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = ("IndexTTS-2 speech for LC Dialogue: clones each speaker from a short clip and mixes its 8 emotions "
                   "(happy, angry, sad, afraid, disgusted, melancholic, surprised, calm) per line. Built in: the "
                   "model (about 8 GB with its helpers) downloads to models/IndexTTS-2 on first use.")

    def load(self, model, emotion_alpha=1.0, temperature=0.8, low_vram=False):
        return (IndexEngine({"model": model, "emotion_alpha": float(emotion_alpha),
                             "temperature": float(temperature), "low_vram": bool(low_vram)}),)


NODE_CLASS_MAPPINGS = {"LCIndexTTSEngine": LCIndexTTSEngine}
NODE_DISPLAY_NAME_MAPPINGS = {"LCIndexTTSEngine": "LC IndexTTS Engine 🗣️"}
