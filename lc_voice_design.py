"""
LC voice design
---------------
LC Voice Design: describe a voice, Maya1 speaks a sample in it, the clip is saved to models/voices/<name>.wav so
LC Dialogue can use the name as a speaker (save it with LC Save Audio's voice_name).

LC Emotion Editor: a test voice speaks a line in a feeling (valence / arousal pad plus named emotions). Saved with LC
Save Audio under any name ('bitter', or 'Haley bitter' for Haley only), a script's [Haley|bitter] copies that feeling.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from .lc_tts_engine import ENGINE_TYPE, INDEX_EMOTIONS



def _voices_dir():
    import folder_paths

    d = os.path.join(folder_paths.models_dir, "voices")
    os.makedirs(d, exist_ok=True)
    return d


def _voice_names():
    """Speaker names in models/voices (file names without extension)."""
    try:
        from .lc_dialogue import _voice_files

        return sorted(dict.fromkeys(os.path.splitext(os.path.basename(f))[0] for f in _voice_files()), key=str.lower)
    except Exception:
        return []


def _safe(name):
    name = re.sub(r"[^\w\- ]+", "", name).strip()
    if not name:
        raise ValueError("Give it a name (letters, numbers, spaces, - or _).")
    return name


def _trim(w, sr, floor_db=-45.0, pad=0.05):
    if w.size == 0:
        return w
    level = np.abs(w).max(0)
    on = np.nonzero(level > 10 ** (floor_db / 20) * max(1e-4, float(level.max())))[0]
    if on.size == 0:
        return w
    return w[:, max(0, on[0] - int(pad * sr)):min(w.shape[-1], on[-1] + int(pad * sr))]


# Maya1 only knows the words it was trained on (its README). Anything else ("alluring", "stern") makes it babble, so the
# description is built from these and nothing more, in Maya's own sentence shape.
SEXES = ["Female", "Male"]
AGES = ["20s", "30s", "40s", "50s"]
ACCENTS = ["American", "British", "Australian", "Indian", "Middle Eastern"]
PITCHES = ["normal", "high", "low"]
TIMBRES = ["smooth", "warm", "gravelly", "raspy"]
PACINGS = ["conversational", "fast", "slow"]
TONES = ["energetic", "happy", "curious", "calm", "angry"]
INTENSITIES = ["medium", "high", "low"]
# The 8 tested voices: between them every word Maya was trained on (sex, age, accent, pitch, timbre, tone, intensity,
# pacing), each one matched its description.
PRESETS = {
    "Custom": None,
    "F_American_20yo": ("Female", "20s", "American", "high", "smooth", "energetic", "high", "fast"),
    "F_British_30yo": ("Female", "30s", "British", "normal", "warm", "happy", "medium", "conversational"),
    "F_Indian_40yo": ("Female", "40s", "Indian", "normal", "smooth", "curious", "medium", "conversational"),
    "F_Australian_50yo": ("Female", "50s", "Australian", "low", "raspy", "calm", "low", "slow"),
    "M_American_20yo": ("Male", "20s", "American", "normal", "smooth", "energetic", "high", "fast"),
    "M_British_30yo": ("Male", "30s", "British", "low", "warm", "happy", "medium", "conversational"),
    "M_MiddleEastern_40yo": ("Male", "40s", "Middle Eastern", "low", "gravelly", "angry", "medium", "conversational"),
    "M_American_50yo": ("Male", "50s", "American", "low", "raspy", "calm", "low", "slow"),
}
PRESET_TIP = ("A tested voice. It replaces the dropdowns below; Custom uses them.\n"
              "F_American_20yo: high pitch, smooth, energetic, high intensity, fast\n"
              "F_British_30yo: normal pitch, warm, happy, medium intensity, conversational\n"
              "F_Indian_40yo: normal pitch, smooth, curious, medium intensity, conversational\n"
              "F_Australian_50yo: low pitch, raspy, calm, low intensity, slow\n"
              "M_American_20yo: normal pitch, smooth, energetic, high intensity, fast\n"
              "M_British_30yo: low pitch, warm, happy, medium intensity, conversational\n"
              "M_MiddleEastern_40yo: low pitch, gravelly, angry, medium intensity, conversational\n"
              "M_American_50yo: low pitch, raspy, calm, low intensity, slow")
# One take that moves through feelings: the saved voice carries its range into everything cloned from it.
RANGE_TEXT = ("Okay, okay, I get it. <laugh> No, seriously, that was the funniest thing I have seen all week! <sigh> But "
              "then she just left. She didn't even say goodbye. <angry> And you know what? I am done waiting for people "
              "who never show up! <whisper> Come here. Don't tell anyone, but I kept the letter. <gasp> Wait, is that "
              "them? <excited> They actually came back!")
# Scripts for the take. Each one moves through feelings with Maya's tags, one every sentence or two: everything cloned
# from the voice inherits the range of this one take. <scream> is left out (it cracks).
SCRIPTS = {
    "Range": RANGE_TEXT,
    "Storyteller": ("Gather around, this one is true. <whisper> It was late, and the whole house was asleep. Then I heard "
                    "it. <gasp> A knock at the door, three times, slow. <sigh> I almost didn't answer. <excited> But when I "
                    "opened it, there was my brother, home after six long years! <cry> I couldn't say a word. <laugh> He "
                    "just grinned and asked what was for dinner."),
    "Argument": ("<angry> You told me you would be there, and you weren't! Do you know how long I waited? <sigh> No, don't. "
                 "I don't want another excuse. <cry> I just wanted you to care, even a little. <whisper> Look at me. Tell "
                 "me the truth this time. <chuckle> Okay. Okay, fine. That is honestly the worst excuse I have ever heard."),
    "Good news": ("<excited> I got it! I actually got the job! <giggle> Sorry, I can't stop smiling. <gasp> Wait, they want "
                  "me to start on Monday? That is three days from now! <sigh> Okay. Breathe. I can do this. <laugh> Who am "
                  "I kidding, I am going to be a mess. <whisper> But honestly? I have never been this happy."),
    "Custom": None,
}
FEELING_TAGS = ("<laugh>", "<sigh>", "<angry>", "<whisper>", "<gasp>", "<excited>", "<cry>", "<giggle>", "<chuckle>")


def maya_description(sex, age, accent, pitch, timbre, tone, intensity, pacing):
    """'Female voice in her 20s with American accent. High pitch, smooth timbre, energetic tone at high intensity,
    fast pacing.' (the shape of Maya1's own examples)."""
    who = "her" if sex == "Female" else "his"
    return (f"{sex} voice in {who} {age} with {accent} accent. {pitch.capitalize()} pitch, {timbre} timbre, "
            f"{tone} tone at {intensity} intensity, {pacing} pacing.")


class LCVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_name": ("STRING", {"default": "Alita", "tooltip": "The speaker name for LC Dialogue. Wire "
                                          "voice_name and audio into LC Save Audio to keep it as "
                                          "models/voices/<name>.wav, or into LC Load Voice for this run only."}),
                "preset": (list(PRESETS), {"default": "Custom", "tooltip": PRESET_TIP}),
                "sex": (SEXES, {"default": "Female"}),
                "age": (AGES, {"default": "20s"}),
                "accent": (ACCENTS, {"default": "American"}),
                "pitch": (PITCHES, {"default": "normal"}),
                "timbre": (TIMBRES, {"default": "smooth"}),
                "tone": (TONES, {"default": "energetic", "tooltip": "How lively the person is. Energetic gives the "
                                 "most range to clone from; calm gives a flat voice."}),
                "intensity": (INTENSITIES, {"default": "medium"}),
                "pacing": (PACINGS, {"default": "conversational"}),
                "script": (list(SCRIPTS), {"default": "Range", "tooltip": "What the voice reads. Each script moves "
                           "through feelings in one take, so every clone of the voice gets that range. Custom reads "
                           "the box below: keep a feeling tag every sentence or two, or the voice comes out flat."}),
                "sample_text": ("STRING", {"multiline": True, "default": RANGE_TEXT,
                    "tooltip": "What the voice reads, in one take. Move through feelings with Maya's tags (<laugh> "
                               "<sigh> <angry> <whisper> <gasp> <excited> <cry> <giggle> <chuckle>): everything cloned "
                               "from this voice inherits its range. About 30 seconds."}),
                "seed": ("INT", {"default": 8675309, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": "fixed",
                                 "tooltip": "Each seed is a different person for the same description."}),
                "check": ("BOOLEAN", {"default": True, "tooltip": "Listens to each take (Whisper, models/stt). When "
                                      "Maya reads the description out loud or babbles, it tries the next seed (up to 4 "
                                      "takes). The info line says which seed passed."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "info", "voice_name")
    OUTPUT_TOOLTIPS = ("The designed voice.", "The take, the seed that passed and the description.",
                       "The name: wire it with the audio into LC Save Audio (keep it) or LC Load Voice (this run).")
    FUNCTION = "design"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = ("Design a voice (Maya1) from the words Maya was trained on, in one take that moves through feelings, "
                   "and save it as a speaker for LC Dialogue.")

    @staticmethod
    def _heard(w, sr, text):
        """Share of the sample text's words Whisper hears in the whole take (0..1), or None without a Whisper model."""
        r = LCVoiceDesign._listen(w, sr, text, window=False)
        return None if r is None else r[0]

    @staticmethod
    def _listen(w, sr, text, window=True):
        """(score, start_s, end_s): how much of the sample text Whisper hears (0..1), and where the text starts and
        ends in the take. Maya sometimes reads part of the description first ("description equals young girl..."):
        with window=True only the stretch from the text's first words to its last word is scored, so that part can
        be cut off. None without a Whisper model."""
        from .lc_av_media import resample
        from .lc_transcribe import _local_names
        from .lc_transcribe import words as whisper_words

        names = _local_names()  # only a Whisper already on disk: the check never starts a download
        if not names:
            print("[LC Voice Design] no Whisper model in models/stt (run LC Transcribe once): take check skipped.")
            return None
        x = resample(np.ascontiguousarray(w.mean(0)), sr, 16000)
        words = [(re.sub(r"[^a-z]", "", ww["word"].lower()), ww["start"], ww["end"])
                 for ww in whisper_words(x, names[0], "en", "auto")]
        words = [("okay" if t[0] == "ok" else t[0], t[1], t[2]) for t in words if t[0]]  # Whisper writes "OK"
        want = re.sub(r"[^a-z ]", " ", re.sub(r"<[^>]*>", " ", text).lower()).split()  # tags aren't words
        if not want:
            return (1.0, 0.0, w.shape[-1] / sr)
        lo, hi = 0, len(words)
        if window and words:
            head = want[:2]
            for i in range(len(words)):  # first place the text's opening words are heard
                if [t[0] for t in words[i:i + len(head)]] == head or (words[i][0] == want[0] and i + 1 < len(words)
                                                                       and words[i + 1][0] in want[:4]):
                    lo = i
                    break
            for j in range(len(words) - 1, lo - 1, -1):  # last place the text's final word is heard
                if words[j][0] == want[-1]:
                    hi = j + 1
                    break
        heard = [t[0] for t in words[lo:hi]]
        pool, hit = list(heard), 0
        for wd in want:
            if wd in pool:
                pool.remove(wd)
                hit += 1
        extra = max(0, len(heard) - len(want))  # words that were never asked for (a read-out description)
        score = max(0.0, (hit - extra) / len(want))
        start = words[lo][1] if words and lo < len(words) else 0.0
        end = words[hi - 1][2] if words and hi > 0 else w.shape[-1] / sr
        return (score, start, end)

    def design(self, voice_name="Alita", preset="Custom", sex="Female", age="20s", accent="American", pitch="normal", timbre="smooth",
               tone="energetic", intensity="medium", pacing="conversational", script="Range", sample_text=RANGE_TEXT, seed=8675309,
               check=True, **_old):
        import torch

        from .lc_maya import SR as MAYA_SR
        from .lc_maya import park, speak

        if PRESETS.get(preset):
            sex, age, accent, pitch, timbre, tone, intensity, pacing = PRESETS[preset]
        if SCRIPTS.get(script):
            sample_text = SCRIPTS[script]
        flat = not any(t in sample_text.lower() for t in FEELING_TAGS)
        desc = maya_description(sex, age, accent, pitch, timbre, tone, intensity, pacing)
        tries, notes, best = (4 if check else 1), [], None
        for t in range(tries):
            s = int(seed) + t
            w = speak(desc, sample_text, seed=s)[None, :]  # the Maya1 pack's defaults: 4000 tokens, 0.4 / 0.9 / 1.1
            sr = MAYA_SR
            import comfy.model_management as mm
            mm.throw_exception_if_processing_interrupted()  # stopped mid-take: Maya hands back empty audio
            if w.size == 0 or w.shape[-1] < sr // 10:
                notes.append(f"seed {s}: no audio")
                continue
            w = _trim(w, sr)
            if check:
                park()  # small cards: Maya waits in system RAM while Whisper listens
            heard = self._listen(w, sr, sample_text) if check else None
            if heard is None:
                best = (1.0, s, w, "")
                notes.append(f"seed {s}")
                break
            score, a, b = heard
            cut = ""
            if a > 0.4:  # something was said before the text (a read-out description): cut it off
                w = w[:, max(0, int((a - 0.12) * sr)):]
                cut = f", cut {a:.1f} s of extra speech before the text"
                b -= max(0.0, a - 0.12)
            if b + 0.35 < w.shape[-1] / sr:  # and anything said after it
                w = w[:, :int((b + 0.35) * sr)]
            if best is None or score > best[0]:
                best = (score, s, w, cut)
            if score >= 0.8:
                notes.append(f"seed {s} ok ({score:.0%} of the words{cut})")
                break
            notes.append(f"seed {s} off script ({score:.0%})")
        if best is None:
            raise RuntimeError("Maya gave no audio on any seed (" + ", ".join(notes) + "). Check models/maya1-TTS/maya1 "
                               "and try again.")
        score, seed, w, cut = best
        info = f"{w.shape[-1] / sr:.1f} s | " + ", ".join(notes)
        if check and score < 0.8:
            info += f" | no take passed: kept the best (seed {seed}, {score:.0%}). Try another seed or a fuller description"
        if flat:
            info += (" | no feeling tags in the script (<laugh> <sigh> <angry> ...): the voice comes out flat, and so "
                     "does everything cloned from it")
        info += f" | Maya1, seed {seed}: {desc}"  # the recipe, so the voice can be made again
        name = (voice_name or _old.get("name") or "").strip()  # name: the widget before 0.6.0's release
        return ({"waveform": torch.from_numpy(np.ascontiguousarray(w))[None], "sample_rate": sr}, info, name)


class LCEmotionMix:
    """Part 2 of the voice workflow: build a feeling on the pad, hear it on a test voice, save it by name with the
    node's save emotion button. A saved emotion is only the feeling (mix + pad), never a voice."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (ENGINE_TYPE, {"tooltip": "LC IndexTTS Engine."}),
                "voices": ("LC_VOICES", {"tooltip": "A test voice from LC Load Voice, only to hear the feeling. The "
                                         "saved emotion has nothing of this voice. With several on the wire, the last "
                                         "one speaks."}),
                "voice_name": ("STRING", {"default": "bitter", "tooltip": "The name save emotion keeps it under. "
                                          "'bitter' works for everyone ([Alice|bitter], [Bob|bitter]); 'Haley bitter' "
                                          "is Haley's own and wins for her."}),
                "mix": ("STRING", {"default": "sad:0.5, angry:0.3", "tooltip": "Named emotions with weights: "
                                   "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm. Click a name "
                                   "below the pad to add it. The weights are shares: IndexTTS-2 holds the whole mix at 0.8 or less."}),
                "valence": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Unpleasant (-1) to pleasant (+1). Drag the pad."}),
                "arousal": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Calm (-1) to energetic (+1). Drag the pad."}),
                "line": ("STRING", {"multiline": True, "default": "I can't believe you kept this from me. After "
                                    "everything we went through, you just stood there and said nothing.",
                                    "tooltip": "What the test voice says, to hear the feeling."}),
                "seed": ("INT", {"default": 8675309, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": "fixed",
                                 "tooltip": "Each seed is a different take of the same feeling."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("preview", "mix")
    OUTPUT_TOOLTIPS = ("The line plain, a short pause, then in the feeling: to hear what the mix does.",
                       "The mix as a script tag: [Haley|sad:0.5,angry:0.3].")
    FUNCTION = "mix_it"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = ("Builds a feeling (valence / arousal pad plus named emotions) and plays it on a test voice. Save "
                   "emotion keeps it by name as an asset: 'bitter' works for everyone in LC Dialogue, 'Haley bitter' "
                   "only for Haley. Saved and premade emotions show as purple buttons: click one to load it.")

    def mix_it(self, engine, voices=None, voice_name="bitter", mix="", valence=0.0, arousal=0.0, line="", seed=8675309,
               **_old):
        import torch

        from .lc_dialogue import _fit, _parse, _trim
        from .lc_voice_load import MAX_SECONDS, cap

        tag = self._tag(mix, valence, arousal)
        line = (line or _old.get("test_line") or "").strip()
        if not line:
            raise ValueError("LC Emotion Editor: give it a line to speak.")
        if not voices:
            raise ValueError("LC Emotion Editor: wire LC Load Voice into voices (any test voice).")
        clip = cap(list(voices.values())[-1]["audio"], MAX_SECONDS)  # the test voice: only to hear the feeling
        feeling = _parse(f"[x|{tag}] {line}", engine.emotions())[0]["pieces"][0][1] if tag else {}
        sr, takes = None, {}
        for key, m in (("plain", {}), ("felt", feeling)):
            w, wsr = engine.speak(line, clip, m, 1.0, int(seed), None)
            sr = sr or wsr
            takes[key] = _trim(_fit(w, wsr, sr, 1), sr)
        pause = np.zeros((1, int(0.8 * sr)), np.float32)
        preview = np.concatenate([takes["plain"], pause, takes["felt"]], axis=-1)
        prev = {"waveform": torch.from_numpy(np.ascontiguousarray(preview))[None], "sample_rate": sr}
        return (prev, tag)

    @staticmethod
    def _tag(mix, valence=0.0, arousal=0.0):
        parts = {}
        for item in [x.strip() for x in re.split(r"[,\n]", mix or "") if x.strip()]:
            k, _, v = item.partition(":")
            parts[k.strip().lower()] = float(v) if v.strip() else 1.0
        if valence:
            parts["valence"] = round(float(valence), 2)
        if arousal:
            parts["arousal"] = round(float(arousal), 2)
        return ",".join(f"{k}:{v:g}" for k, v in parts.items())


class LCVoiceMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "engine": (ENGINE_TYPE, {"tooltip": "LC IndexTTS Engine: its speaker model (CAM++) does the listening."}),
            "reference": ("AUDIO", {"tooltip": "The voice it should sound like (your clip)."}),
            "audio": ("AUDIO", {"tooltip": "What came out (a clone, a dialogue line...)."}),
        }}

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("match", "info")
    FUNCTION = "match"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = "How alike two voices are (1.0 = the same voice), measured by a speaker encoder. Checks a clone against its clip."

    def match(self, engine, reference, audio):
        if not hasattr(engine, "speaker_vector"):
            raise ValueError(f"{engine.name} has no speaker encoder (LC IndexTTS Engine does).")
        a, b = engine.speaker_vector(reference), engine.speaker_vector(audio)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        word = "same voice" if cos >= 0.85 else "close" if cos >= 0.7 else "a different voice"
        return (cos, f"{cos:.3f} ({word})")


NODE_CLASS_MAPPINGS = {"LCVoiceDesign": LCVoiceDesign, "LCEmotionMix": LCEmotionMix, "LCVoiceMatch": LCVoiceMatch}
NODE_DISPLAY_NAME_MAPPINGS = {"LCVoiceDesign": "LC Voice Design 🎙️ (BETA)", "LCEmotionMix": "LC Emotion Editor 🎨",
                              "LCVoiceMatch": "LC Voice Match 🔍"}


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/lc_av/voice_design_presets")
    async def _lc_voice_design_presets(request):
        keys = ("sex", "age", "accent", "pitch", "timbre", "tone", "intensity", "pacing")
        return web.json_response({"presets": {k: dict(zip(keys, v)) for k, v in PRESETS.items() if v},
                                  "scripts": {k: v for k, v in SCRIPTS.items() if v}})

    @PromptServer.instance.routes.get("/lc_av/emotions")
    async def _lc_emotions(request):
        from .lc_dialogue import saved_emotions

        saved = sorted(saved_emotions().values(), key=lambda e: e["name"].lower())
        return web.json_response({"built_in": list(INDEX_EMOTIONS), "designed": [e["name"] for e in saved],
                                  "saved": {e["name"]: e for e in saved}})

    @PromptServer.instance.routes.post("/lc_av/emotions/save")
    async def _lc_emotions_save(request):
        from .lc_dialogue import user_emotions_dir

        try:
            b = await request.json()
            name = re.sub(r"\s+", " ", str(b.get("name") or "")).strip()
            if not name or not re.fullmatch(r"[\w\- ]+", name):
                return web.json_response({"error": "The name needs letters, numbers, spaces, - or _."})
            if name.lower() in {e.lower() for e in INDEX_EMOTIONS}:
                return web.json_response({"error": f"'{name}' is a built-in emotion. Pick another name."})
            e = {"name": name, "mix": str(b.get("mix") or ""), "valence": round(float(b.get("valence") or 0), 2),
                 "arousal": round(float(b.get("arousal") or 0), 2)}
            if not e["mix"].strip() and not e["valence"] and not e["arousal"]:
                return web.json_response({"error": "Nothing to save: the mix is empty and the dot is in the middle."})
            d = user_emotions_dir()
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(e, f, indent=1)
            return web.json_response({"ok": True, "name": name})
        except Exception as ex:
            return web.json_response({"error": str(ex)})
except Exception as _e:  # no server (CPU tests): the node still works
    pass
