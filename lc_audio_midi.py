"""
LC Audio to MIDI
----------------
Turns a single melody line (voice, whistle, lead synth, bass) into MIDI notes. pyin (librosa's, ported to numpy in lc_dsp) tracks the pitch; a run
of the same note longer than min_note_ms becomes one MIDI note, with velocity from its loudness. The notes are saved
as a standard .mid file (format 0, one track) in the output folder. Chords and full mixes come out messy: give it
one instrument at a time.
"""

from __future__ import annotations

import json
import math
import os
import struct

import numpy as np

from .lc_av_media import audio_parts

_PPQ = 480


def _vlq(v):
    v = int(v)
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def midi_bytes(notes, bpm=120.0):
    """notes: [{midi, start, end, velocity}] in seconds -> a format 0 .mid file as bytes."""
    tick = bpm / 60.0 * _PPQ  # ticks per second
    events = []
    for nt in notes:
        on = int(round(nt["start"] * tick))
        off = max(on + 1, int(round(nt["end"] * tick)))
        events.append((on, 1, bytes([0x90, nt["midi"], nt["velocity"]])))
        events.append((off, 0, bytes([0x80, nt["midi"], 0])))
    events.sort(key=lambda e: (e[0], e[1]))  # note-offs first when they share a tick
    us = int(round(60_000_000 / bpm))
    track = bytearray()
    track += b"\x00\xff\x51\x03" + us.to_bytes(3, "big")
    track += b"\x00\xff\x58\x04\x04\x02\x18\x08"  # 4/4
    last = 0
    for t, _, data in events:
        track += _vlq(t - last) + data
        last = t
    track += b"\x00\xff\x2f\x00"
    head = b"MThd" + struct.pack(">IHHH", 6, 0, 1, _PPQ)
    return head + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)


def _hz(text, default):
    from . import lc_dsp as librosa_lite

    s = str(text).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(librosa_lite.note_to_hz(s))
    except Exception:
        return float(librosa_lite.note_to_hz(default))


class LCAudioToMidi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fmin": ("STRING", {"default": "C2", "tooltip": "Lowest note to look for, as a note (C2, A3) or Hz. "
                                                                "C2 = 65 Hz, low male voice and bass."}),
                "fmax": ("STRING", {"default": "C6", "tooltip": "Highest note to look for. C6 = 1047 Hz, covers "
                                                                "voices and most leads."}),
                "min_note_ms": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 1000.0, "step": 5.0,
                                          "tooltip": "Shorter blips are dropped. Raise it if you get many tiny "
                                                     "notes."}),
                "tempo": ("FLOAT", {"default": 120.0, "min": 20.0, "max": 300.0, "step": 0.5,
                                    "tooltip": "Tempo written into the .mid file. Notes keep their real timing "
                                               "either way; match the song so the grid lines up in your DAW."}),
                "filename_prefix": ("STRING", {"default": "audio/LC_midi",
                                               "tooltip": "Folder/name inside the output folder."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("notes", "path", "info")
    FUNCTION = "convert"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Single melody line to MIDI notes, saved as a .mid file. One instrument or voice at a time."

    def convert(self, audio, fmin="C2", fmax="C6", min_note_ms=80.0, tempo=120.0, filename_prefix="audio/LC_midi"):
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0 or float(np.abs(wave).max()) < 1e-6:
            return ("[]", "", "no sound")
        from . import lc_dsp as librosa_lite
        from scipy.ndimage import median_filter
        from scipy.signal import resample_poly

        y = np.asarray(wave, dtype=np.float32).mean(axis=0)
        work = 22050
        if sr != work:
            g = math.gcd(int(sr), work)
            y = resample_poly(y, work // g, sr // g).astype(np.float32)
        lo = _hz(fmin, "C2")
        hi = _hz(fmax, "C6")
        if hi <= lo * 1.1:
            hi = lo * 4
        hop = 256
        f0, voiced, _ = librosa_lite.pyin(y, fmin=lo, fmax=hi, sr=work, frame_length=2048, hop_length=hop)
        rms = librosa_lite.rms(y=y, frame_length=2048, hop_length=hop)[0][:len(f0)]
        midi = np.where(voiced & np.isfinite(f0), np.round(librosa_lite.hz_to_midi(np.nan_to_num(f0, nan=1.0))), -1)
        midi = median_filter(midi.astype(np.int64), size=5, mode="nearest")  # drop 1-2 frame pitch flickers
        dt = hop / work
        min_frames = max(1, int(round(float(min_note_ms) / 1000.0 / dt)))
        runs = []
        i = 0
        while i < len(midi):
            j = i
            while j + 1 < len(midi) and midi[j + 1] == midi[i]:
                j += 1
            if midi[i] >= 0 and j - i + 1 >= min_frames:
                runs.append((int(midi[i]), i, j + 1, float(rms[i:j + 1].mean())))
            i = j + 1
        notes = []
        if runs:
            loud = max(r[3] for r in runs)
            for m, a, b, level in runs:
                db = 20 * math.log10(max(level, 1e-9) / max(loud, 1e-9))
                vel = int(np.clip(round(127 * (1 + db / 40.0)), 1, 127))
                notes.append({"midi": int(np.clip(m, 0, 127)), "name": librosa_lite.midi_to_note(m, unicode=False),
                              "start": round(a * dt, 3), "end": round(b * dt, 3), "velocity": vel})
        import folder_paths

        folder, name, counter, _, _ = folder_paths.get_save_image_path(filename_prefix,
                                                                       folder_paths.get_output_directory())
        path = os.path.join(folder, f"{name}_{counter:05}_.mid")
        with open(path, "wb") as f:
            f.write(midi_bytes(notes, float(tempo)))
        span = (f"range {librosa_lite.midi_to_note(min(n['midi'] for n in notes), unicode=False)}-"
                f"{librosa_lite.midi_to_note(max(n['midi'] for n in notes), unicode=False)}") if notes else "no notes"
        info = (f"{len(notes)} notes ({span}) | voiced {float(np.mean(voiced)) * 100:.0f}% of "
                f"{len(y) / work:.2f} s | {float(tempo):g} BPM | {os.path.basename(path)}")
        return (json.dumps(notes), path, info)


NODE_CLASS_MAPPINGS = {"LCAudioToMidi": LCAudioToMidi}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioToMidi": "LC Audio to MIDI 🎹"}
