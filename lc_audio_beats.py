"""
LC Audio Beats
--------------
Finds the tempo, the beats and the onsets (every hit or note start) in a song (librosa's tracker, ported to numpy in lc_dsp), and gives them as
seconds and as video frame numbers at your fps. envelope is the loudness per video frame (0..1, JSON list), handy to
drive zooms, flashes or any value that should pulse with the music.
"""

from __future__ import annotations

import json

import numpy as np

from .lc_av_media import audio_parts


def _csv(values, fmt):
    return ",".join(fmt.format(v) for v in values)


class LCAudioBeats:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "Your video's frame rate, for beat_frames and envelope."}),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider",
                                          "tooltip": "Onsets only. Higher finds softer hits too; lower keeps "
                                                     "only the strong ones."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("bpm", "beat_times", "beat_frames", "onsets", "envelope", "info")
    FUNCTION = "analyze"
    CATEGORY = "LC AV/audio"
    DESCRIPTION = "Tempo, beats and onsets of a song, as seconds and video frame numbers, plus a loudness envelope."

    def analyze(self, audio, fps=24.0, sensitivity=0.5):
        wave, sr = audio_parts(audio)
        fps = float(fps)
        if wave is None or sr <= 0 or wave.size == 0 or float(np.abs(wave).max()) < 1e-6:
            frames = 0 if wave is None or sr <= 0 else int(np.ceil(wave.shape[-1] / sr * fps))
            return (0.0, "", "", "", json.dumps([0.0] * frames), "no sound")
        from . import lc_dsp as librosa_lite

        y = np.asarray(wave, dtype=np.float32).mean(axis=0)
        tempo, beats = librosa_lite.beat_track(y=y, sr=sr, units="time")
        bpm = float(np.atleast_1d(tempo)[0])
        delta = max(0.005, 0.14 * (1.0 - float(sensitivity)))
        onsets = librosa_lite.onset_detect(y=y, sr=sr, units="time", delta=delta, backtrack=False)
        beat_frames = [int(round(t * fps)) for t in beats]
        n_frames = int(np.ceil(y.shape[-1] / sr * fps))
        edges = np.round(np.arange(n_frames + 1) * sr / fps).astype(np.int64)
        edges = np.minimum(edges, y.shape[-1])
        csum = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
        cnt = np.maximum(edges[1:] - edges[:-1], 1)
        rms = np.sqrt((csum[edges[1:]] - csum[edges[:-1]]) / cnt)
        env = rms / max(float(rms.max()), 1e-12)
        info = (f"{bpm:.1f} BPM | {len(beats)} beats, {len(onsets)} onsets | {y.shape[-1] / sr:.2f} s = "
                f"{n_frames} frames at {fps:g} fps")
        return (bpm, _csv(beats, "{:.3f}"), _csv(beat_frames, "{}"), _csv(onsets, "{:.3f}"),
                json.dumps([round(float(v), 3) for v in env]), info)


NODE_CLASS_MAPPINGS = {"LCAudioBeats": LCAudioBeats}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAudioBeats": "LC Audio Beats 🥁"}
