"""
LC Load Voice
-------------
Loads a voice clip and gives it the speaker name the script uses ([Haley] in LC Dialogue). Picking a file fills
voice_name with the file's name (Haley.wav -> Haley); change it when the file has a bad name. Chain several: each one
adds its voice to the cast it gets, and the last one's voices output carries the whole cast on one wire into LC
Dialogue. Nothing is saved: a one-shot voice is just a loaded or uploaded file. Clips longer than max_seconds are cut
(IndexTTS-2 has trouble with long references; 10 to 20 s of one person talking clones best).
"""

from __future__ import annotations

import os

VOICES_TYPE = "LC_VOICES"  # {speaker name (lower case): {"name", "audio"}}
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
MAX_SECONDS = 30.0  # the longest reference any LC voice node hands to the engine


def _choices():
    from .lc_dialogue import _voice_files

    try:
        import folder_paths

        inp = folder_paths.get_input_directory()
        extra = sorted(f for f in os.listdir(inp) if f.lower().endswith(AUDIO_EXT))
    except Exception:
        extra = []
    return list(dict.fromkeys(_voice_files() + extra)) or ["(no audio files: upload one)"]


def _path(name):
    from .lc_dialogue import _find_voice

    p = _find_voice(os.path.splitext(name)[0]) or _find_voice(name)
    if p:
        return p
    import folder_paths

    cand = os.path.join(folder_paths.get_input_directory(), name)
    return cand if os.path.isfile(cand) else None


def stem(name):
    """The voice name a file gives: 'voices/Haley bitter.wav' -> 'Haley bitter'."""
    return os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0].strip()


def cap(audio, seconds):
    """The first `seconds` of a ComfyUI AUDIO (a voice reference longer than that is cut)."""
    wv, sr = audio["waveform"], int(audio["sample_rate"])
    n = int(round(float(seconds) * sr))
    if wv.shape[-1] <= n:
        return audio
    return {"waveform": wv[..., :n].contiguous(), "sample_rate": sr}


class LCLoadVoice:
    @classmethod
    def INPUT_TYPES(cls):
        files = _choices()
        return {
            "required": {
                "voice_name": ("STRING", {"default": stem(files[0]), "tooltip": "The speaker name, as the script "
                                          "writes it: [Haley] uses this voice. Picking a file fills it with the "
                                          "file's name; type over it to rename. Empty = the file's name."}),
                "audio": (files, {"audio_upload": True,
                                  "tooltip": "A voice clip: your premade voices in models/voices and files in the "
                                             "input folder. Upload brings in a file from anywhere (it lands in the "
                                             "input folder). 10 to 20 s of one person talking clones best."}),
                "max_seconds": ("FLOAT", {"default": 20.0, "min": 3.0, "max": MAX_SECONDS, "step": 1.0,
                                          "tooltip": "Longer clips are cut to this. Past about 30 s the voice "
                                                     "engine has trouble."}),
            },
            "optional": {
                "voices": (VOICES_TYPE, {"tooltip": "The cast so far, from another LC Load Voice. This voice is added "
                                         "to it (a voice with the same name is replaced)."}),
            },
        }

    RETURN_TYPES = (VOICES_TYPE, "AUDIO", "STRING")
    RETURN_NAMES = ("voices", "audio", "voice_name")
    OUTPUT_TOOLTIPS = ("The cast: wire it into LC Dialogue's voices (or the next LC Load Voice).",
                       "This voice's clip (cut to max_seconds).",
                       "This voice's name: wire it with the audio into LC Save Audio to keep the voice in models/voices.")
    FUNCTION = "load"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = ("Loads a voice clip and names it for the script (the file's name unless you type one). Chain "
                   "several for a whole cast on one wire into LC Dialogue.")

    @classmethod
    def VALIDATE_INPUTS(cls, **kw):
        return True  # files come and go in models/voices and input; checked when the node runs

    def load(self, voice_name="", audio="", max_seconds=20.0, voices=None, **_old):
        from .lc_dialogue import _load_audio

        file = audio or _old.get("file", "")  # file: the picker's name before release
        p = _path(file) if file else None
        if p is None:
            raise FileNotFoundError(f"LC Load Voice: pick or upload a voice file ({file or 'none picked'} is not in "
                                    "models/voices or the input folder).")
        name = (voice_name or "").strip() or stem(file)
        clip = cap(_load_audio(p), min(float(max_seconds), MAX_SECONDS))
        cast = dict(voices or {})
        cast[name.lower()] = {"name": name, "audio": clip}
        return (cast, clip, name)


NODE_CLASS_MAPPINGS = {"LCLoadVoice": LCLoadVoice}
NODE_DISPLAY_NAME_MAPPINGS = {"LCLoadVoice": "LC Load Voice 🎙️"}
