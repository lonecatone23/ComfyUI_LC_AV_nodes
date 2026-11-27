"""
LC Voice Shaper
--------------
Basic voice controls for speech: pitch, voice size (formants), how lively the pitch moves, speed, tremble, breathiness,
plus warmth, softness and air. The WORLD vocoder (pyworld, MIT) splits the voice into its pitch, its vocal tract shape
and its breath, so each moves on its own: a deeper voice sounds like a bigger person, not a slowed-down tape, and a
lighter voice like a smaller person, not a chipmunk. The EQ is three plain RBJ filters (scipy).
Works on speech; music or sound effects in the clip get mangled, so split first (LC Audio Split) and shape only the
speech stem.
"""

from __future__ import annotations

import math

import numpy as np

from .lc_av_media import audio_parts, make_audio

FRAME_MS = 5.0

# Presets set the sliders they name; anything left out stays at its neutral value. Age is more than pitch: older voices
# tremble, get breathier, slower and lose their top end; children are higher, smaller, quicker and livelier.
NEUTRAL = dict(pitch=0.0, voice_size=1.0, liveliness=1.0, speed=1.0, warmth_db=0.0, softness_db=0.0, air_db=0.0,
               tremble=0.0, breathiness=0.0)
PRESETS = {
    "Custom": None,
    "Flat": dict(NEUTRAL),  # every slider back to the center: the voice as it came in
    "Deeper (baritone)": dict(pitch=-3.0, voice_size=0.92, liveliness=0.9, warmth_db=2.0, air_db=-1.0),
    "Bass": dict(pitch=-6.0, voice_size=0.86, liveliness=0.8, warmth_db=3.0, softness_db=1.0, air_db=-2.0),
    "Tenor (lighter man)": dict(pitch=2.0, voice_size=1.04, liveliness=1.1, warmth_db=-1.0, air_db=1.0),
    "Softer, higher (woman)": dict(pitch=2.0, voice_size=1.06, liveliness=1.1, warmth_db=-1.0, softness_db=3.0, air_db=1.5),
    "Lower, fuller (woman)": dict(pitch=-2.0, voice_size=0.95, liveliness=0.95, warmth_db=1.5, softness_db=1.0),
    "Young girl (from a woman)": dict(pitch=5.0, voice_size=1.18, liveliness=1.35, speed=1.05, warmth_db=-3.0,
                                      softness_db=1.0, air_db=2.0, breathiness=0.1),
    "Young boy (from a man)": dict(pitch=9.0, voice_size=1.22, liveliness=1.3, speed=1.05, warmth_db=-4.0, air_db=2.0),
    "Elderly woman": dict(pitch=-2.0, voice_size=0.97, liveliness=0.85, speed=0.88, warmth_db=-1.0, softness_db=2.0,
                          air_db=-3.0, tremble=0.45, breathiness=0.35),
    "Elderly man": dict(pitch=-1.0, voice_size=0.96, liveliness=0.8, speed=0.88, warmth_db=-1.0, softness_db=1.0,
                        air_db=-3.0, tremble=0.5, breathiness=0.3),
    "Calmer (less lively)": dict(liveliness=0.6, warmth_db=1.0, softness_db=1.5),
    "More expressive": dict(liveliness=1.5, air_db=0.5),
}



def _pitch_range(x64, sr):
    """The speaker's own pitch range, so the tracker never loses the voice (a floor above it or a ceiling below it
    turns into static). A quick wide DIO pass, then the 5th / 95th percentile with headroom."""
    import pyworld as pw
    f0, t = pw.dio(x64, sr, f0_floor=50.0, f0_ceil=1100.0, frame_period=FRAME_MS)
    f0 = pw.stonemask(x64, f0, t, sr)
    v = f0[f0 > 0]
    if v.size < 10:
        return 50.0, 800.0
    lo, hi = np.percentile(v, [5, 95])
    return float(max(40.0, lo * 0.7)), float(min(1100.0, max(hi * 1.5, lo * 3.0)))


def _move_pitch(f0, t, pitch, lively):
    """Pitch moves the speaker's middle; liveliness scales the melody around it. Only the smooth melody (~50 ms) is
    scaled: the tracker's tiny jumps at word starts and ends stay as spoken, or scaling them turns them into clicks.
    The melody gets bridged across unvoiced gaps so a word picks up where the last one ended, and stays within about
    an octave of the middle (a dive far below that turns the voice into single pulses you can hear)."""
    v = f0 > 0
    lf = np.log(np.where(v, f0, 1.0))
    mid = float(np.median(lf[v]))
    idx = np.arange(len(f0))
    cont = np.interp(idx, idx[v], lf[v])  # continuous melody through the gaps
    k = 11
    pad = np.pad(cont, k // 2, mode="edge")
    smooth = np.convolve(pad, np.ones(k) / k, mode="valid")
    detail = np.clip(lf - smooth, -0.12, 0.12)  # micro movement, kept as spoken (+-2 semitones max)
    melody = np.clip((smooth - mid) * float(lively), -0.6, 0.6)  # +-10 semitones around the middle
    new = np.exp(mid + melody + detail + math.log(2) * float(pitch) / 12.0)
    out = f0.copy()
    out[v] = np.maximum(new[v], 45.0)
    return out


def _world(x, sr, pitch, size, lively, speed, tremble, breath):
    """One channel through WORLD: analyse, move pitch / vocal tract / breath / timing, then synthesize."""
    try:
        import pyworld as pw
    except ImportError as e:
        raise RuntimeError("LC Voice Shaper needs the pyworld package (MIT). Install it in ComfyUI's Python "
                           "(python -m pip install pyworld) and restart.") from e
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    fmin, fmax = _pitch_range(x64, sr)
    f0, t = pw.harvest(x64, sr, f0_floor=float(fmin), f0_ceil=float(fmax), frame_period=FRAME_MS)
    sp = pw.cheaptrick(x64, f0, t, sr)
    ap = pw.d4c(x64, f0, t, sr)
    voiced = f0 > 0
    if voiced.sum() >= 5:
        f0 = _move_pitch(f0, t, pitch, lively)
    if tremble > 0:  # age tremor: ~5.5 wobbles a second, up to +-4 % at 1, with a slow drift so it isn't a vibrato
        tt = t.astype(np.float64)
        f0[voiced] *= 1.0 + 0.04 * float(tremble) * np.sin(2 * np.pi * 5.5 * tt[voiced] + 0.8 * np.sin(2 * np.pi * 0.7 * tt[voiced]))
    if size != 1.0:  # vocal tract: above 1 = a smaller person (formants up), below 1 = a bigger one (formants down)
        k = sp.shape[1]
        src = np.arange(k) / float(size)
        i0 = np.clip(np.floor(src).astype(int), 0, k - 1)
        i1 = np.clip(i0 + 1, 0, k - 1)
        w = np.clip(src - i0, 0.0, 1.0)
        sp = sp[:, i0] * (1 - w) + sp[:, i1] * w
        sp[:, src > k - 1] = sp[:, -1:].min()  # nothing above the old top: keep it quiet
        ap = ap[:, i0] * (1 - w) + ap[:, i1] * w
    if speed != 1.0:  # talking speed: fewer or more frames, the same pitch
        n = len(f0)
        m = max(2, int(round(n / float(speed))))
        pos = np.linspace(0, n - 1, m)
        near = np.clip(np.round(pos).astype(int), 0, n - 1)
        lo = np.floor(pos).astype(int)
        hi = np.clip(lo + 1, 0, n - 1)
        fr = (pos - lo)[:, None]
        sp = sp[lo] * (1 - fr) + sp[hi] * fr
        ap = ap[lo] * (1 - fr) + ap[hi] * fr
        old = f0
        both = (old[lo] > 0) & (old[hi] > 0)  # blend pitch only between two voiced frames; else take the nearest
        f0 = np.where(both, old[lo] * (1 - fr[:, 0]) + old[hi] * fr[:, 0], old[near])
    breath_noise = None
    if breath > 0:  # air: noise shaped by the voice's own envelope, only from about 1.2 kHz up (full by 3.5 kHz) and
        # only while the voice sounds. Noise in the low band is what turned into static, so it stays out of there.
        b = float(breath)
        # first take the voice's own rough noise out of the sounded parts (WORLD resynthesizes it as a hiss/static),
        # so the added breath is the only air there. Consonants keep theirs. Fully clean from 0.5 up.
        ap[f0 > 0] *= max(0.0, 1.0 - 2.0 * b)
        f = np.linspace(0.0, sr / 2.0, sp.shape[1])
        w = np.clip((f - 1200.0) / 2300.0, 0.0, 1.0) ** 0.7
        on = (f0 > 0).astype(np.float64)
        on = np.convolve(on, np.ones(9) / 9.0, mode="same")[:, None]  # soft edges, no clicks at word starts
        sp_n = sp * (w ** 2)[None, :] * on + 1e-16  # WORLD takes the log: never exactly 0
        breath_noise = pw.synthesize(np.zeros_like(f0), np.ascontiguousarray(sp_n), np.ones_like(ap), sr,
                                     frame_period=FRAME_MS) * (1.6 * b)
        # real breath puffs with every vocal fold cycle: a steady hiss on top reads as a separate noise, so the air is
        # pulsed at the voice's own pitch (open phase loud, closed phase quiet)
        hop = sr * FRAME_MS / 1000.0
        ts = np.arange(len(breath_noise)) / hop
        fs = np.interp(ts, np.arange(len(f0)), np.where(f0 > 0, f0, 0.0))
        phase = np.cumsum(2.0 * np.pi * fs / sr)
        env = 0.25 + 0.75 * (0.5 + 0.5 * np.cos(phase)) ** 2
        breath_noise = breath_noise * np.where(fs > 0, env / 0.53, 1.0)  # 0.53 = mean of env: same loudness
        sp = sp * (1.0 - 0.6 * b * w)[None, :]  # the buzz up top gives way to the air
    y = pw.synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(np.clip(ap, 0.0, 1.0)), sr,
                      frame_period=FRAME_MS)
    if breath_noise is not None:
        n = min(len(y), len(breath_noise))
        y = y[:n] + breath_noise[:n]
    return y.astype(np.float32)


def _rbj(kind, sr, f0, gain_db, q):
    """RBJ cookbook biquad: lowshelf, highshelf or peak. Returns (b, a)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * min(f0, sr * 0.45) / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:
        sa = 2 * math.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
            a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
        else:  # highshelf
            b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
            a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return np.array(b) / a[0], np.array(a) / a[0]


def _eq(x, sr, warmth_db, softness_db, air_db):
    from scipy.signal import lfilter

    for kind, f, g, q in (("lowshelf", 200.0, warmth_db, 0.7), ("peak", 3800.0, -softness_db, 0.8),
                          ("highshelf", min(6000.0, sr * 0.25), air_db, 0.7)):  # TTS is 22-24 kHz: little above 8 kHz
        if g:
            b, a = _rbj(kind, sr, f, float(g), q)
            x = lfilter(b, a, x, axis=-1)
    return x.astype(np.float32)


class LCVoiceShape:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Speech. Split music and effects off first (LC Audio Split)."}),
                "voice_name": ("STRING", {"default": "", "tooltip": "The speaker name for LC Dialogue. Wire "
                                          "voice_name into LC Save Audio: it saves the voice as "
                                          "models/voices/<name>.wav (replaces one with that name)."}),
                "preset": (list(PRESETS), {"default": "Custom",
                           "tooltip": "A starting point. Picking one sets the sliders it uses; Custom uses the sliders."}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                                    "tooltip": "Semitones. -3 to -6 deepens a man toward baritone / bass, +2 to +4 "
                                               "lifts a voice. Past 6 it starts to sound processed."}),
                "voice_size": ("FLOAT", {"default": 1.0, "min": 0.75, "max": 1.35, "step": 0.01,
                                         "tooltip": "The size of the person: below 1 = bigger chest and throat (deeper, "
                                                    "fuller), above 1 = smaller (lighter, younger). 0.9 to 1.1 sounds "
                                                    "natural. Pair it with pitch: deeper pitch + smaller than 1."}),
                "liveliness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                         "tooltip": "How much the pitch moves: 0 = flat robot, 1 = as spoken, 1.5 = "
                                                    "more expressive."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01,
                                    "tooltip": "Talking speed without changing pitch. 0.9 = a touch slower."}),
                "tremble": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "The shake in an elderly voice. 0.3 to 0.5 for old age, 1 is a "
                                                 "frightened quiver."}),
                "breathiness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Air in the voice. Subtle under 0.6, clear at 1.0. Deep voices "
                                                     "need it higher."}),
                "warmth_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                                        "tooltip": "Low end around 200 Hz: + is fuller and warmer, - is thinner."}),
                "softness_db": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 12.0, "step": 0.5,
                                          "tooltip": "Turns down the harsh 3-5 kHz edge: + is softer and gentler."}),
                "air_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                                     "tooltip": "Top end above 6 kHz: + is brighter and crisper, - is duller."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "info", "voice_name")
    OUTPUT_TOOLTIPS = ("The shaped voice.", "What was changed.",
                       "The name to save it under: wire it into LC Save Audio's voice_name.")
    FUNCTION = "shape"
    CATEGORY = "LC AV/voice"
    DESCRIPTION = ("Pitch, voice size, liveliness, speed, warmth, softness and air for speech: deepen a man's voice, "
                   "soften and lift a woman's, make a voice younger or older.")

    def shape(self, audio, preset="Custom", pitch=0.0, voice_size=1.0, liveliness=1.0, speed=1.0, warmth_db=0.0,
              softness_db=0.0, air_db=0.0, tremble=0.0, breathiness=0.0, voice_name="", **_old):
        voice_name = (voice_name or _old.get("save_as") or "").strip()  # save_as: the name before 0.6.0's release
        wave, sr = audio_parts(audio)
        if wave is None or sr <= 0 or wave.size == 0:
            return (audio, "no sound", voice_name)
        if PRESETS.get(preset):
            v = {**NEUTRAL, **PRESETS[preset]}
            pitch, voice_size, liveliness, speed = v["pitch"], v["voice_size"], v["liveliness"], v["speed"]
            warmth_db, softness_db, air_db = v["warmth_db"], v["softness_db"], v["air_db"]
            tremble, breathiness = v["tremble"], v["breathiness"]
        x = np.asarray(wave, dtype=np.float32)
        peak_in = float(np.abs(x).max()) or 1.0
        info = []
        if pitch or voice_size != 1.0 or liveliness != 1.0 or speed != 1.0 or tremble or breathiness:
            chans = [_world(ch, sr, pitch, voice_size, liveliness, speed, tremble, breathiness)
                     for ch in x]
            n = min(len(c) for c in chans)
            x = np.stack([c[:n] for c in chans])
            info.append(f"pitch {pitch:+g} st, size {voice_size:g}, liveliness {liveliness:g}, speed {speed:g}"
                        + (f", tremble {tremble:g}" if tremble else "") + (f", breathiness {breathiness:g}" if breathiness else ""))
        if warmth_db or softness_db or air_db:
            x = _eq(x, sr, warmth_db, softness_db, air_db)
            info.append(f"warmth {warmth_db:+g} dB, softness {softness_db:+g} dB, air {air_db:+g} dB")
        peak = float(np.abs(x).max())
        if peak > 0.999:  # keep the original headroom instead of clipping
            x = x * (min(peak_in, 0.98) / peak)
        msg = (preset + ": " if PRESETS.get(preset) else "") + ("; ".join(info) or "unchanged")
        return (make_audio(x.astype(np.float32), sr), msg, voice_name)


NODE_CLASS_MAPPINGS = {"LCVoiceShape": LCVoiceShape}
NODE_DISPLAY_NAME_MAPPINGS = {"LCVoiceShape": "LC Voice Shaper 🦜 (BETA)"}


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/lc_av/voice_shape_presets")
    async def _lc_voice_shape_presets(request):
        # every slider for every named preset, so the node can move its sliders when a preset is picked
        return web.json_response({"neutral": NEUTRAL, "presets": {k: {**NEUTRAL, **v} for k, v in PRESETS.items() if v}})
except Exception:  # no server (CPU tests)
    pass
