"""
LC H3 Segment / LC H3 Segment Save / LC H3 Assemble
---------------------------------------------------
Makes a long MiniMax H3 video one short segment per queue run, from an
LC Vision Director plan (output/LC_H3/<project>/plan.json).

- Segment: builds the conditioning + empty AV latent for one segment. A
  CONTINUE segment anchors the previous segment's saved tail (last
  `overlap` frames, video and audio) at frame 0 as a MiniMax H3 guide, so the
  motion carries straight on. Wire its outputs into a normal sampler.
- Segment Save: decodes the sampled segment, saves a preview mp4, the part
  that goes into the final video (overlap cut off), and the new tail.
- Assemble: joins the saved parts into one mp4 with ffmpeg (nothing is held
  in RAM), with a tiny audio fade at every join.

Files per segment in the project folder: seg_001.mp4 (whole segment),
seg_001_part.mp4 (for the final video), seg_001_tail.mp4 + seg_001_tail.wav
(the handoff to the next segment) and seg_001.json (settings, seed, prompt).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch

import folder_paths
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _encode_ref_audio, _resize

from .lc_av_media import audio_parts
from .lc_h3_refs import REFS_TYPE, subset as _refs_subset
from .lc_save_video import _ffmpeg_bin, _write_wav

PLAN_TYPE = "LC_H3_PLAN"
NL = chr(10)
SEGMENT_TYPE = "LC_H3_SEGMENT"
FPS = 24
TIME_PER_SECOND = 40.0  # H3 time axis: 5/3 per video frame at 24 fps = 1 per audio latent frame at 40 fps
_STAMP_RE = re.compile(r"(\d+(?:\.\d+)?)(\s*seconds)")


def _patch_ref_gap():
    """H3 lays references on its timeline right before frame 0, so the video reads them as "the moment just
    before" and opens on them. A reference block carrying "lc_time_gap" pushes the target (and its guide
    frames) that much later on the time axis, leaving a gap after the references. Blocks without the key
    are untouched, so stock H3 behaves exactly as before."""
    try:
        import comfy.ldm.minimax.model as mm
    except Exception:
        return
    if getattr(mm.PackedLayout, "_lc_gap", False):
        return
    orig = mm.PackedLayout.__init__

    def init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        orig(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=keyframes, refs=refs)
        gap = max((float(b.get("lc_time_gap", 0.0)) for b in (refs or ())), default=0.0)
        if gap > 0:
            for start, stop, kind in self.segments:
                if kind in ("cond", "cond_audio", "audio", "video"):
                    self.position_ids[start:stop, 0] += gap

    mm.PackedLayout.__init__ = init
    mm.PackedLayout._lc_gap = True


_patch_ref_gap()


def project_dir(project: str) -> str:
    """Same folder rule as LC Vision Director."""
    safe = re.sub(r"[^A-Za-z0-9 _.-]", "_", (project or "").strip()) or "untitled"
    return os.path.join(folder_paths.get_output_directory(), "LC_H3", safe)


def _seg_path(folder: str, index: int, suffix: str) -> str:
    return os.path.join(folder, f"seg_{index:03d}{suffix}")


def _is_done(folder: str, index: int) -> bool:
    return os.path.exists(_seg_path(folder, index, ".json")) and os.path.exists(_seg_path(folder, index, "_part.mp4"))


def _resolve_index(plan: dict, index: int) -> int:
    count = len(plan.get("shots", []))
    if index > 0:
        return min(index, count)
    folder = project_dir(plan.get("project", ""))
    for i in range(1, count + 1):
        if not _is_done(folder, i):
            return i
    return count


def _frames(plan: dict):
    seg = int(round(plan["segment_s"] * FPS))
    ovl = int(round(plan["overlap_s"] * FPS))
    return seg, ovl


def _read_mp4_frames(path: str, width: int, height: int) -> torch.Tensor:
    """mp4 -> IMAGE tensor [N, H, W, 3] via ffmpeg (scaled to the canvas)."""
    cmd = [_ffmpeg_bin(), "-v", "error", "-i", path, "-vf", f"scale={width}:{height}:flags=lanczos",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def _encode(frames_u8, dest, crf, pix_fmt="yuv420p", audio=None):
    """rgb frames (+ optional audio) -> h264 mp4. No -shortest: AAC padding would make
    ffmpeg drop the last video frame, and a lost frame per segment adds up."""
    n, h, w, _c = frames_u8.shape
    tmp = tempfile.mkdtemp(prefix="lc_h3_")
    try:
        raw = os.path.join(tmp, "frames.rgb")
        frames_u8.tofile(raw)
        cmd = [_ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", raw]
        wave, sr = audio_parts(audio)
        if wave is not None and sr > 0:
            wav = os.path.join(tmp, "a.wav")
            _write_wav(wav, wave, sr)
            cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", "libx264", "-crf", str(int(crf)), "-pix_fmt", pix_fmt, "-movflags", "+faststart", dest]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _read_wav(path: str) -> dict:
    import wave as wavmod
    with wavmod.open(path, "rb") as w:
        ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).T.astype(np.float32) / 32767.0
    return {"waveform": torch.from_numpy(pcm).unsqueeze(0), "sample_rate": sr}


def _color_match(frames: torch.Tensor, ref: torch.Tensor, strength: float) -> torch.Tensor:
    """Pull the tail's per-channel mean / contrast toward segment 1 (drift guard)."""
    if strength <= 0 or ref is None:
        return frames
    fm, fs = frames.mean(dim=(0, 1, 2)), frames.std(dim=(0, 1, 2)) + 1e-5
    rm, rs = ref.mean(dim=(0, 1, 2)), ref.std(dim=(0, 1, 2)) + 1e-5
    matched = (frames - fm) / fs * rs + rm
    return (frames + (matched - frames) * strength).clamp(0, 1)


def _shift_stamps(text: str, offset: float) -> str:
    return _STAMP_RE.sub(lambda m: f"{float(m.group(1)) + offset:.2f}{m.group(2)}", text)


def _scene_start(plan: dict, i: int, folder: str = "") -> int:
    """The segment a CONTINUE run grew from: the last CUT at or before i (segment 1 counts as a cut). A segment
    saved as a CUT counts as one even when the plan says CONTINUE."""
    shots = plan.get("shots") or []

    def is_cut(k):
        saved = _load_json(_seg_path(folder, k, ".json")) if folder else None
        return (saved or {}).get("mode", shots[k - 1].get("mode")) != "CONTINUE"

    k = i
    while k > 1 and not is_cut(k):
        k -= 1
    return k


def _look(frames_u8: np.ndarray) -> dict:
    """Average Lab color + contrast of some frames: the numbers the drift check compares."""
    x = frames_u8[:, ::4, ::4].reshape(-1, 3).astype(np.float32) / 255.0
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    xyz = x @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]], dtype=np.float32)
    xyz /= np.array([0.9505, 1.0, 1.089], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    lab_l = 116 * f[:, 1] - 16
    lab_a = 500 * (f[:, 0] - f[:, 1])
    lab_b = 200 * (f[:, 1] - f[:, 2])
    return {"L": round(float(lab_l.mean()), 2), "a": round(float(lab_a.mean()), 2), "b": round(float(lab_b.mean()), 2),
            "contrast": round(float(lab_l.std()), 2)}


FACE_SAME, FACE_CLOSE = 0.55, 0.40  # best of 5 references: the same person scores about 0.8, a drifted face 0.25 to 0.35
FACE_OTHER = 0.25  # under this it is somebody else in the scene (a barista scores 0.0 to 0.22), not her drifting
MIN_FACE_FRAMES = 5  # of the ~21 sampled frames
_FACE_MODELS = {}


def _face_models():
    """OpenCV YuNet (face finder) + SFace (face match), from models/facerestore. None when missing."""
    if "pair" not in _FACE_MODELS:
        pair = None
        try:
            import cv2
            base = os.path.join(folder_paths.models_dir, "facerestore")
            det_p = os.path.join(base, "face_detection_yunet_2023mar.onnx")
            rec_p = os.path.join(base, "face_recognition_sface_2021dec.onnx")
            if os.path.exists(det_p) and os.path.exists(rec_p):
                pair = (cv2.FaceDetectorYN.create(det_p, "", (320, 320), 0.7), cv2.FaceRecognizerSF.create(rec_p, ""))
        except Exception as e:
            print(f"[LC H3] face check off: {e}")
        _FACE_MODELS["pair"] = pair
    return _FACE_MODELS["pair"]


def _frontal_faces(rgb: np.ndarray):
    """[(face row, frontal score)] for every clear front-facing face (0.5% of the frame or more). A face turned
    away, or hair / a reflection read as one, gives a meaningless match, so those are left out."""
    pair = _face_models()
    if pair is None:
        return []
    det = pair[0]
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    out = []
    for f in faces if faces is not None else []:
        # landmarks: right eye (4,5), left eye (6,7), nose (8,9); confidence at 14
        eye_dist = abs(f[6] - f[4]) + 1e-3
        if f[2] * f[3] < 0.005 * w * h or f[14] < 0.8 or eye_dist < 0.25 * f[2]:
            continue
        frontal = 1.0 - abs(f[8] - (f[4] + f[6]) / 2) / eye_dist
        level = 1.0 - abs(f[7] - f[5]) / eye_dist  # eyes level, not a tilted head
        if frontal < 0.55:
            continue
        out.append((f, float(f[14]) * frontal * max(0.0, level) * float(np.sqrt(f[2] * f[3] / (w * h)))))
    return out


def _face_features(rgb: np.ndarray) -> list:
    """SFace features of every clear front-facing face in the frame (other people included)."""
    pair = _face_models()
    if pair is None:
        return []
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return [pair[1].feature(pair[1].alignCrop(bgr, f)) for f, _ in _frontal_faces(rgb)]


def _best_face_frames(frames_u8: np.ndarray, count: int = 5, apart: int = 12) -> list:
    """Indexes of the frames with the clearest front-facing face, at least `apart` frames from each other, so the
    references cover a few moments (expressions, hood up and down) instead of one."""
    scored = []
    for k in range(0, len(frames_u8), 3):
        faces = _frontal_faces(frames_u8[k])
        if faces:
            scored.append((max(sc for _, sc in faces), k))
    picked = []
    for sc, k in sorted(scored, reverse=True):
        if all(abs(k - p) >= apart for p in picked):
            picked.append(k)
        if len(picked) == count:
            break
    return picked


def _best_face_frame(frames_u8: np.ndarray):
    """The single clearest front-facing face frame, or None."""
    picked = _best_face_frames(frames_u8, 1)
    return picked[0] if picked else None


def _face_refs(frames_u8: np.ndarray) -> np.ndarray | None:
    """Up to 5 reference faces (SFace features) from segment 1: the largest clear face of its best frames."""
    pair = _face_models()
    if pair is None:
        return None
    feats = []
    for k in _best_face_frames(frames_u8):
        faces = _frontal_faces(frames_u8[k])
        f = max(faces, key=lambda t: t[0][2] * t[0][3])[0]
        bgr = np.ascontiguousarray(frames_u8[k][..., ::-1])
        feats.append(pair[1].feature(pair[1].alignCrop(bgr, f)).reshape(-1))
    return np.stack(feats) if feats else None


def _frame_match(pair, refs: np.ndarray, rgb: np.ndarray):
    """Best match of any clear face in the frame against any reference, or None with no clear face."""
    best = None
    for ft in _face_features(rgb):
        for r in refs:
            m = float(pair[1].match(r.reshape(1, -1).astype(np.float32), ft, 0))  # 0 = cosine
            best = m if best is None else max(best, m)
    return best


def _face_score(frames_u8: np.ndarray, refs_path: str):
    """How much the main person matches segment 1: per frame the best match of any face against any reference
    (so a passer-by or a new expression doesn't count against her), then the median over the segment. Frames where
    the only clear face is somebody else (under FACE_OTHER, e.g. the barista while she has her back to us) are
    left out instead of scored as drift."""
    pair = _face_models()
    if pair is None or not os.path.exists(refs_path):
        return None
    refs = np.load(refs_path)
    scores, others = [], 0
    for k in range(0, len(frames_u8), 6):
        best = _frame_match(pair, refs, frames_u8[k])
        if best is None:
            continue
        if best < FACE_OTHER:
            others += 1
        else:
            scores.append(best)
    if len(scores) < MIN_FACE_FRAMES:
        return {"match": None, "frames": len(scores), "others": others}  # from behind, too small or someone else: nothing fair to compare
    return {"match": round(float(np.median(scores)), 3), "low": round(float(np.percentile(scores, 10)), 3),
            "frames": len(scores), "others": others}


def _needs_reanchor(folder: str, i: int, width: int, height: int) -> str:
    """Why segment i (a CONTINUE) should get the memory too, or "" when its guide frames carry the face fine."""
    prev = _load_json(_seg_path(folder, i - 1, ".json")) or {}
    m = (prev.get("face") or {}).get("match")
    if m is not None and m < FACE_CLOSE:
        return f"segment {i - 1}'s face drifted ({m:.2f})"
    tail = _seg_path(folder, i - 1, "_tail.mp4")
    if _face_models() is None or not os.path.exists(tail):
        return ""
    fr = (_read_mp4_frames(tail, width, height) * 255.0).round().to(torch.uint8).numpy()
    refs_path = _seg_path(folder, 1, "_face_refs.npy")
    if os.path.exists(refs_path):  # the main person's face, not just any face (a barista facing us doesn't carry the look on)
        pair, refs = _face_models(), np.load(refs_path)
        seen = sum(1 for k in range(0, len(fr), 5) if (_frame_match(pair, refs, fr[k]) or -1.0) >= FACE_OTHER)
    else:
        seen = sum(1 for k in range(0, len(fr), 5) if _frontal_faces(fr[k]))
    return "" if seen >= 3 else f"no clear face at the end of segment {i - 1}"


def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


HEAD_SIDE = 384  # memory head clip size: small, and only the face, hair and hood, never the place


def _head_crop(frames_u8: np.ndarray) -> np.ndarray | None:
    """The memory clip cropped to the main person's head (one fixed square over the whole clip), resized to
    HEAD_SIDE. A full-frame memory carries its background, so every cut opened in segment 1's location."""
    boxes = []
    for fr in frames_u8:
        faces = _frontal_faces(fr)
        if faces:
            f = max(faces, key=lambda t: t[0][2] * t[0][3])[0]
            boxes.append((f[0], f[1], f[0] + f[2], f[1] + f[3]))
    if not boxes:
        return None
    b = np.array(boxes, dtype=np.float32)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    h, w = frames_u8.shape[1:3]
    side = min(min(h, w), max(x1 - x0, y1 - y0) * 1.9)  # head, hair and hood, a little shoulder
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.05 * side
    left = int(min(max(0, cx - side / 2), w - side))
    top = int(min(max(0, cy - side / 2), h - side))
    crop = frames_u8[:, top:top + int(side), left:left + int(side)]
    t = torch.from_numpy(crop.astype(np.float32) / 255.0).movedim(-1, 1)
    t = torch.nn.functional.interpolate(t, size=(HEAD_SIDE, HEAD_SIDE), mode="bicubic", align_corners=False).clamp(0, 1)
    return (t.movedim(1, -1) * 255.0).round().to(torch.uint8).numpy()


def _any_faces(rgb: np.ndarray) -> list:
    """Every confident face box (x0, y0, x1, y1), profiles included."""
    pair = _face_models()
    if pair is None:
        return []
    det = pair[0]
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    return [(f[0], f[1], f[0] + f[2], f[1] + f[3]) for f in (faces if faces is not None else [])
            if f[14] >= 0.7 and f[2] * f[3] >= 0.003 * w * h]


def _solo_head_crop(frames_u8: np.ndarray) -> np.ndarray | None:
    """A close-up's subject: the largest face in each frame (profiles too), centred on the median box, so a few frames
    where someone else is bigger can't drag the crop between two people."""
    boxes = []
    for fr in frames_u8:
        faces = _any_faces(fr)
        if faces:
            boxes.append(max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
    if not boxes:
        return None
    b = np.array(boxes, dtype=np.float32)
    cx, cy = np.median((b[:, 0] + b[:, 2]) / 2), np.median((b[:, 1] + b[:, 3]) / 2)
    size = float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])))
    h, w = frames_u8.shape[1:3]
    side = min(min(h, w), size * 1.8)
    cy += 0.05 * side
    left = int(min(max(0, cx - side / 2), w - side))
    top = int(min(max(0, cy - side / 2), h - side))
    crop = frames_u8[:, top:top + int(side), left:left + int(side)]
    t = torch.from_numpy(crop.astype(np.float32) / 255.0).movedim(-1, 1)
    t = torch.nn.functional.interpolate(t, size=(HEAD_SIDE, HEAD_SIDE), mode="bicubic", align_corners=False).clamp(0, 1)
    return (t.movedim(1, -1) * 255.0).round().to(torch.uint8).numpy()


def _style_for(style: str, name: str, names: dict) -> str:
    """The STYLE block for a shot of one person: drop the clauses that describe the others ("a woman in her late
    thirties ..." in his close-up), keep the rest (camera, look)."""
    others = {n: g for n, g in names.items() if n != name}
    if not others:
        return style
    words = set(others) | {g for g in others.values() if g != names.get(name)}
    parts = re.split(r"(?<=[.;])\s+", style)
    keep = [x for x in parts if not any(re.search(rf"\b{re.escape(wd)}\b", x, re.I) for wd in words)]
    return " ".join(keep).strip() or style


def _read_mp4_native(path: str) -> torch.Tensor:
    """mp4 -> IMAGE tensor at the file's own size."""
    # ffmpeg -i prints the stream size (ffprobe isn't always there, imageio-ffmpeg ships ffmpeg only)
    info = subprocess.run([_ffmpeg_bin(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    w, h = (int(v) for v in re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info).groups())
    return _read_mp4_frames(path, w, h)


def _sharpest(frames_u8: np.ndarray) -> int:
    """Index of the crispest frame in the middle half: a good memory picture of the cast."""
    n = len(frames_u8)
    best, best_i = -1.0, n // 2
    for k in range(n // 4, max(n // 4 + 1, 3 * n // 4), 2):
        g = frames_u8[k, ::2, ::2].astype(np.float32).mean(axis=2)
        score = float(np.var(np.diff(g, axis=0)) + np.var(np.diff(g, axis=1)))
        if score > best:
            best, best_i = score, k
    return best_i


def _portrait_crop(frame: np.ndarray) -> np.ndarray:
    """3:4 crop around the busiest (most detailed) columns, where the person usually is. A full frame
    from the video reads to H3 as a frame to open on; a portrait crop reads as a picture of someone."""
    h, w = frame.shape[:2]
    cw = min(w, int(h * 0.75))
    g = frame.astype(np.float32).mean(axis=2)
    energy = np.abs(np.diff(g, axis=0)).sum(axis=0)
    energy = np.convolve(energy, np.ones(cw, dtype=np.float32), mode="valid")
    x = int(np.argmax(energy)) if len(energy) else 0
    return frame[:, x:x + cw]


def _ref_size(h: int, w: int, width: int, height: int):
    """Same sizing as MiniMax H3 Reference to Video's 'match': down to the canvas area, keep aspect."""
    scale = min(1.0, math.sqrt((width * height) / (w * h)))
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)


_SPOKEN = re.compile(r'[\u201c"]([^\u201d"]+)[\u201d"]')
_MOUTH = re.compile(r",?\s*(?:with\s+)?(?:(?:his|her|their)\s+)?(?:mouth|lips)\s+(?:slightly\s+)?(?:open|opening|parted|parting)"
                    r"(?:\s+(?:as if about to speak|in [^,.;]+))?|,?\s*as if about to speak", re.I)


def _silent(text: str) -> str:
    """A segment without a line: nobody speaks. "The only voice is the quoted line" with no quoted line, or a mouth
    "slightly open as if about to speak", makes H3 invent talking (gibberish)."""
    text = re.sub(r"The only voice is the quoted line\.?", "Nobody speaks.", text)
    return _MOUTH.sub("", text)


_SAYS = re.compile(r"\b(says|say|said|asks|whispers|replies|answers|shouts|calls|adds|tells)\b[^.;]*$", re.I)
_SUBJECT = re.compile(r"\b(?:the\s+(?:man|woman|girl|boy|barista|waiter|waitress)|she|he|[A-Z][a-z]+)\b(?!['’]s)")
_NOT_NAMES = {"At", "The", "A", "An", "Her", "His", "Their", "It", "Then", "As", "When", "While", "After", "Before",
              "And", "But", "With", "In", "On", "She", "He", "They"}


def _speaker(text: str, end: int):
    """Who says the quote that starts at `end`: the subject of its sentence ("Claire squeezes his hand and says" ->
    Claire, "At 1.85 seconds, she says" -> she), or None when no say-verb leads into it (a sign that reads "OPEN")."""
    start = max(text.rfind(c, 0, end) for c in (".", ";", "\n", "seconds,"))
    clause = text[start + 1:end]
    if not _SAYS.search(clause):
        return None
    for m in _SUBJECT.finditer(clause):
        w = m.group(0)
        if w in _NOT_NAMES and w not in ("She", "He"):
            continue
        return start + 1 + m.start(), start + 1 + m.end(), w.lower()
    return None


def _dialogue(text: str, style: str = "") -> str:
    """Quoted lines as <d>[English] ... </d>, and the person who says each one tagged (S1), (S2) the way H3 was
    trained ("Claire (S1) squeezes his hand and says <d>[English] ... </d>"), so the right mouth moves. Speakers are
    numbered in the order the style block names them, then in the order they speak."""
    quotes = list(_SPOKEN.finditer(text))
    who = [_speaker(text, m.start()) for m in quotes]
    names = [w[2] for w in who if w]
    low = style.lower()
    order = sorted(set(names), key=lambda k: (low.find(k) if k in low else 10 ** 6, names.index(k)))
    ids = {k: i + 1 for i, k in enumerate(order)}
    out, last = [], 0
    for m, w in zip(quotes, who):
        if w and w[0] >= last:
            out.append(text[last:w[1]] + f" (S{ids[w[2]]})" + text[w[1]:m.start()])
        else:
            out.append(text[last:m.start()])
        out.append(f"<d>[English] {m.group(1).strip()} </d>")
        last = m.end()
    return "".join(out) + text[last:]


def _h3_sections(style: str, action: str, ref: str | None, people: int = 1, focus=None) -> str:
    """The prompt in H3's trained layout: integrated_multimodal_description / overall_soundscape / non_diegetic_music,
    quoted lines as <d>[English] ... </d>, and with a reference the subject_definitions / summary / retention_analysis
    / detailed_description block. ref: None, "picture", "video" or "video_audio" (the memory)."""
    keep, sound = [], ""
    for line in action.split("\n"):
        if line.strip().lower().startswith("sound:"):
            sound = line.split(":", 1)[1].strip()
        else:
            keep.append(line)
    body = _dialogue("\n".join(keep).strip(), style)
    sound = re.sub(r"\s*No music\.?", "", sound).strip()  # the music section says it
    desc = (style + NL + NL + body).strip()
    tail = NL + NL + "overall_soundscape: " + (sound or "quiet natural ambience.") + NL + NL + "non_diegetic_music: none"
    if ref is None:
        return "integrated_multimodal_description: " + desc + tail
    who = "<Picture 1>" if ref == "picture" else "<Video 1>"
    if people > 1:
        subj, defs = "<Subject 1> and <Subject 2>", f"<Subject 1> and <Subject 2> are the two people shown in {who}."
        keep_line = (f"{subj} ({who}): fully_preserved - their faces, hair and clothing are retained. Only the people "
                     f"come from {who}; the place, the poses and the framing are new.")
    else:
        subj, defs = "<Subject 1>", f"<Subject 1> is the person shown in {who}."
        keep_line = (f"<Subject 1> ({who}): fully_preserved - the face, hair and clothing of <Subject 1> are retained. "
                     f"Only the person comes from {who}; the place, the pose and the framing are new.")
    follow = subj
    if people > 1 and focus:  # a close-up of one of the two: name which one, and keep the other out of frame
        defs += f" <Subject 1> is the {focus[1]}, {focus[0]}."
        follow = "<Subject 1> alone; only <Subject 1> is in frame"
    task = "[reference generation]"
    if ref == "video_audio":
        keep_line += NL + f"<Audio 1>: reference - the voice{'s' if people > 1 else ''} of {subj}."
        task = "[reference generation + audio reference]"
    return ("subject_definitions:" + NL + defs + NL + NL
            + f"summary: {task} The target video follows {follow}." + NL + NL
            + "retention_analysis:" + NL + keep_line + NL + NL + "detailed_description: " + desc + tail)


def _mentions(text: str, name: str) -> bool:
    name = (name or "").strip()
    if not name:
        return False
    core = re.sub(r"^(the|a|an)\s+", "", name, flags=re.I)
    if re.search(rf"\b{re.escape(core)}\b", text, re.I):
        return True
    head = core.split()[-1] if core.split() else ""  # "the pier" names "the fishing pier"
    return len(core.split()) > 1 and len(head) > 3 and bool(re.search(rf"\bthe\s+{re.escape(head)}\b", text, re.I))


_PEOPLE = re.compile(r"\b(she|he|her|him|his|they|them|their|couple|both|everyone|people|man|woman|girl|boy|friends|crowd)\b", re.I)


def _shot_refs(refs: dict, shot: dict, style: str, genders: dict | None = None) -> dict | None:
    """The reference pictures one shot needs, renumbered from <Picture 1>: the people named in its action (everyone
    when it names nobody but clearly has people in it), the settings and props it names (the only setting when there is just one), style always."""
    pics = refs.get("pictures") or []
    if not pics:
        return None
    action = shot.get("action", "")
    text = action + NL + style
    cast = sorted({p["name"] for p in pics if p["role"].startswith("cast") and p["name"]})
    who = [n for n in cast if _mentions(action, n)]
    if not who and _PEOPLE.search(action):
        who = cast
        # only "she" (or only "he"): the people of that sex, from the story's cast list ("Lonecat, a woman ...")
        she = bool(re.search(r"\b(she|her|hers)\b", action, re.I))
        he = bool(re.search(r"\b(he|him|his)\b", action, re.I))
        if genders and she != he:
            want = {"woman", "girl", "lady"} if she else {"man", "boy", "guy"}
            picked = [n for n in cast if (genders.get(n) or "") in want]
            who = picked or cast
    settings = sorted({p["name"] for p in pics if p["role"] == "setting"})
    where = [n for n in settings if _mentions(text, n)] or (settings if len(settings) == 1 else [])
    props = [p["name"] for p in pics if p["role"] == "prop" and _mentions(text, p["name"])]
    keep = {n.strip().lower() for n in who + where + props}
    sub = _refs_subset(refs, keep)
    return sub if sub["pictures"] else None


def _h3_sections_refs(style: str, action: str, pics: list, memory: str | None = None, people: int = 1) -> str:
    """H3's trained reference layout for the Reference Assembler's pictures: one <Subject k> per named person (their
    face / full body pictures), and retention lines for each setting, prop and style picture. memory: "video" or
    "video_audio" when segment 1's memory clip rides along (no cast pictures in this shot), as <Video 1> / <Audio 1>."""
    keep, sound = [], ""
    for line in action.split("\n"):
        if line.strip().lower().startswith("sound:"):
            sound = line.split(":", 1)[1].strip()
        else:
            keep.append(line)
    body = _dialogue("\n".join(keep).strip(), style)
    sound = re.sub(r"\s*No music\.?", "", sound).strip()
    desc = (style + NL + NL + body).strip()
    tail = NL + NL + "overall_soundscape: " + (sound or "quiet natural ambience.") + NL + NL + "non_diegetic_music: none"
    tag = lambda p: f"<Picture {p['n']}>"  # noqa: E731
    join = lambda t: t[0] if len(t) == 1 else ", ".join(t[:-1]) + " and " + t[-1]  # noqa: E731
    people_pics = {}
    for p in pics:
        if p["role"].startswith("cast"):
            people_pics.setdefault(p["name"] or "the person", []).append(p)
    defs, kept, subjects = [], [], []
    for k, (name, ps) in enumerate(people_pics.items(), 1):
        subj = f"<Subject {k}>"
        subjects.append(subj)
        tags = [tag(p) for p in ps]
        defs.append(f"{subj} is {name}, the person shown in {join(tags)}.")
        what = "face, hair and clothing" if any(p["role"] == "cast full body" for p in ps) else "face and hair"
        note = "; ".join(p["note"].strip() for p in ps if p.get("note", "").strip())
        kept.append(f"{subj} ({', '.join(tags)}): fully_preserved - the {what} of {subj} are retained"
                    + (f" ({note})" if note else "") + ". Only the person comes from " + join(tags) + "; not the background.")
    if memory and not people_pics:
        subj = "<Subject 1> and <Subject 2>" if people > 1 else "<Subject 1>"
        subjects.append(subj)
        defs.append(f"{subj} {'are the two people' if people > 1 else 'is the person'} shown in <Video 1>.")
        kept.append(f"{subj} (<Video 1>): fully_preserved - {'their faces' if people > 1 else 'the face'}, hair and clothing are retained. "
                    "Only the people come from <Video 1>; the place, the poses and the framing are new.")
        if memory == "video_audio":
            kept.append(f"<Audio 1>: reference - the voice{'s' if people > 1 else ''} of {subj}.")
    for p in pics:
        name = p["name"] or ""
        if p["role"] == "setting":
            kept.append(f"{tag(p)} ({name or 'the place'}): reference - only the place and its light are used, none of its people.")
        elif p["role"] == "prop":
            kept.append(f"{tag(p)} ({name or 'the object'}): fully_preserved - the shape, colors and any printed text of the {name or 'object'} are retained.")
        elif p["role"] == "style":
            kept.append(f"{tag(p)}: reference - only the colors, light and grain are used.")
    follow = join(subjects) if subjects else "the scene"
    task = "[reference generation + audio reference]" if memory == "video_audio" and not people_pics else "[reference generation]"
    return (("subject_definitions:" + NL + NL.join(defs) + NL + NL if defs else "")
            + f"summary: {task} The target video follows {follow}." + NL + NL
            + "retention_analysis:" + NL + NL.join(kept) + NL + NL + "detailed_description: " + desc + tail)


_NAMED = re.compile(r"\b([A-Z][a-z]+),\s+an?\s+(man|woman|boy|girl|guy|lady)\b")


def _cast_names(plan_or_story) -> dict:
    """{name: "man" / "woman" ...} from the story ("Daniel, a man in his forties ..."). Empty when it names nobody."""
    story = plan_or_story.get("story", "") if isinstance(plan_or_story, dict) else str(plan_or_story or "")
    return {m.group(1): m.group(2) for m in _NAMED.finditer(story)}


def _solo(text: str, names: dict) -> str | None:
    """The one named person a shot is of (its Camera line: "close-up of Daniel"), or None for a shot of several."""
    cam = next((ln for ln in text.split("\n") if ln.strip().lower().startswith("camera:")), "")
    if not cam or re.search(r"\b(both|two|them|couple|together|everyone)\b", cam, re.I):
        return None
    hits = [n for n in names if re.search(rf"\b{n}\b", cam)]
    return hits[0] if len(hits) == 1 else None


def _cast_path(folder: str, name: str, ext: str) -> str:
    return os.path.join(folder, f"cast_{re.sub(r'[^A-Za-z0-9]', '', name)}{ext}")


def _cast_size(folder: str, width: int, height: int) -> int:
    """Most clear front-facing faces seen in one frame of segment 1's memory clip (1 when unsure). Two or more:
    the memory keeps the whole frame, so a cut to either person has both of them to copy."""
    path = _seg_path(folder, 1, "_memory_clip.mp4")
    if _face_models() is None or not os.path.exists(path):
        return 1
    fr = (_read_mp4_frames(path, width, height) * 255.0).round().to(torch.uint8).numpy()
    return max([1] + [len(_frontal_faces(fr[k])) for k in range(0, len(fr), 4)])


class LCH3Segment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {"tooltip": "From LC Vision Director 🎬."}),
                "clip": ("CLIP", {"tooltip": "MiniMax H3 text encoder."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 60, "tooltip": "Which segment to make. 0 = the next one not saved yet. Set a number to redo that segment."}),
                "width": ("INT", {"default": 960, "min": 256, "max": 2048, "step": 32, "tooltip": "960 x 544 is a good low VRAM size. Keep it the same for every segment of a project."}),
                "height": ("INT", {"default": 544, "min": 256, "max": 2048, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Passed through to your sampler and saved with the segment. Change it to redo a segment."}),
                "carry_on_line": ("BOOLEAN", {"default": True, "tooltip": "CONTINUE segments: adds a line telling H3 the opening is the previous moment carrying straight on."}),
                "color_match": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Pulls the carried frames' color toward the start of this scene (the last CUT) to fight drift. 0 = off. Turn it on (0.2 to 0.4) when Assemble's drift check flags a scene."}),
                "memory": (["clip", "on", "off"], {"default": "clip", "tooltip": "Keeps the cast the same. clip: about 1 s of segment 1 (picture and sound) goes in as a video reference so faces, hair, clothes and voices stay the same, on a CUT written into the story, and as a gentle nudge on any CONTINUE that starts with no clear face (or after a segment whose face drifted). on: a still portrait from segment 1 (or memory_image). off: the style text alone."}),
                "memory_gap": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5, "tooltip": "Seconds of empty time H3 sees between the memory and the start of the cut. Without a gap H3 treats the memory as the moment just before the cut and opens on its pose. 5 s worked in tests. 0 = stock H3."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "Start image for segment 1 (and for CUT segments when wired)."}),
                "memory_image": ("IMAGE", {"tooltip": "Your own cast picture (a character sheet or a still) to use as the memory instead of segment 1's frame. Used for every segment, segment 1 included."}),
                "prompt_format": (["sections (H3 trained)", "plain"], {"default": "sections (H3 trained)", "tooltip": "sections: the layout H3 was trained on (description / soundscape / non_diegetic_music: none, dialogue as <d>[English] ... </d>, and the reference block for the memory). plain: style + action text."}),
                "redo_after": (["just this segment", "this + the CONTINUE ones after it"], {"default": "this + the CONTINUE ones after it", "tooltip": "When index redoes a segment that is already saved. The old take is always kept in the project's takes folder. this + the CONTINUE ones after it: the segments that carry straight on from this one no longer join, so they are set aside too; set index back to 0 and queue to make them again. A CUT after it stays."}),
                "refs": (REFS_TYPE, {"tooltip": "From LC H3 Reference Assembler: your cast, setting, prop and style pictures. Each segment gets only the ones its shot names."}),
                "refs_on": (["cuts + re-anchors", "every segment"], {"default": "cuts + re-anchors", "tooltip": "When the references are sent. cuts + re-anchors: CUT segments, and a CONTINUE that lost the face (a reference on a plain CONTINUE pulled the opening away from the carried frames in tests). every segment: always."}),
            },
        }

    # prompt last: added after the first workflows were saved (outputs only ever go on the end)
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", SEGMENT_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "seed", "segment", "info", "prompt")
    FUNCTION = "build"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = "One segment of an LC Vision Director plan: conditioning + latent for your sampler. CONTINUE segments start from the previous segment's saved ending."

    @classmethod
    def IS_CHANGED(cls, plan=None, index=0, **kwargs):
        if not isinstance(plan, dict):
            # plan comes in through a link, and ComfyUI doesn't hand linked inputs to IS_CHANGED: always run, or a
            # second Run with a fixed seed would replay the cached segment instead of making the next one
            return float("nan")
        i = _resolve_index(plan, index)
        prev = _seg_path(project_dir(plan.get("project", "")), i - 1, "_tail.mp4")
        return f"{i}:{os.path.getmtime(prev) if os.path.exists(prev) else 0}"

    def build(self, plan, clip, vae, audio_vae, index, width, height, seed, carry_on_line, color_match, memory="clip", memory_gap=5.0,
              first_frame=None, memory_image=None, prompt_format="sections (H3 trained)",
              redo_after="this + the CONTINUE ones after it", refs=None, refs_on="cuts + re-anchors"):
        shots = plan.get("shots") or []
        if not shots:
            raise ValueError("[LC H3] The plan has no segments.")
        i = _resolve_index(plan, index)
        shot = shots[i - 1]
        folder = project_dir(plan["project"])
        os.makedirs(folder, exist_ok=True)
        seg_frames, ovl_frames = _frames(plan)
        cont = shot["mode"] == "CONTINUE" and i > 1
        # memory: one picture of the cast (segment 1's crispest frame, or memory_image)
        mem = None
        if memory == "on":
            if memory_image is not None:
                mem = memory_image[:1]
            elif i > 1 and os.path.exists(_seg_path(folder, 1, "_memory.png")):
                from PIL import Image
                mem = torch.from_numpy(np.asarray(Image.open(_seg_path(folder, 1, "_memory.png")).convert("RGB"), dtype=np.float32) / 255.0)[None]
        # memory rides on CUT segments only: a CONTINUE segment already carries the cast in its guide frames,
        # and a reference there pulls the opening away from them (the join got worse in tests).
        # A bare <Picture 1> reads as "open on this picture" to H3, so the memory is declared the way
        # MiniMax's prompt guide declares a reference subject: subject_definitions + retention_analysis.
        # memory "clip": about 1 s of segment 1 (video + its sound) as <Audio 1> / <Video 1>. A moving face
        # carries who she is without one frozen pose to copy (the H3-Multishot / JoyEcho memory idea).
        mem_clip = None
        cast = 1
        focus = None  # (name, "man" / "woman"): a shot of one person while the memory shows two
        # a CONTINUE segment carries the face through its guide frames, unless those frames show none (walked away,
        # turned round) or the segment before already drifted: then it gets the memory too, or it invents a new face
        reanchor = ""
        if memory == "clip" and cont:
            reanchor = _needs_reanchor(folder, i, width, height)
        if memory == "clip" and (not cont or reanchor) and i > 1 and os.path.exists(_seg_path(folder, 1, "_memory_clip.mp4")):
            head = _seg_path(folder, 1, "_memory_head.mp4")
            cast = _cast_size(folder, width, height)
            # one person: the head-only clip (no background to copy); two or more: the whole frame, or every cut
            # would get the one face the head crop centred on
            cfr = _read_mp4_native(head) if os.path.exists(head) and cast < 2 else _read_mp4_frames(_seg_path(folder, 1, "_memory_clip.mp4"), width, height)
            n_ok = cfr.shape[0]
            while n_ok > 5 and n_ok % 17 != 5:
                n_ok -= 1
            mem_clip = (cfr[:n_ok], _read_wav(_seg_path(folder, 1, "_memory_clip.wav")) if os.path.exists(_seg_path(folder, 1, "_memory_clip.wav")) else None)
            names = _cast_names(plan)
            solo = _solo(shot["action"], names)
            if solo and os.path.exists(_cast_path(folder, solo, ".mp4")):
                own = _read_mp4_native(_cast_path(folder, solo, ".mp4"))
                wav = _cast_path(folder, solo, ".wav")
                mem_clip = (own, _read_wav(wav) if os.path.exists(wav) else None)
                cast = 1
            if solo:
                focus = (solo, names[solo])
        if cont:
            mem = None
        # your references (LC H3 Reference Assembler): the ones this shot names, on CUTs and re-anchors by default
        shot_refs = None
        if refs and (not cont or refs_on == "every segment" or (cont and (reanchor if memory == "clip" else _needs_reanchor(folder, i, width, height)))):
            shot_refs = _shot_refs(refs, shot, plan.get("style", ""), _cast_names(plan))
        if shot_refs and any(p["role"].startswith("cast") for p in shot_refs["pictures"]):
            mem_clip, mem = None, None  # your cast pictures take over from segment 1's memory
        latent, frame_count = _empty_av_latent(width, height, seg_frames)

        action = shot["action"]
        if cont and carry_on_line:
            action = (f"From 0.00 to {plan['overlap_s']:.2f} seconds, the shot carries straight on from the previous "
                      f"moment with the same pose, position and camera. ") + action
        style = plan.get("style", "")
        solo_now = _solo(shot["action"], _cast_names(plan))
        if solo_now:
            style = _style_for(style, solo_now, _cast_names(plan))
        if not _SPOKEN.search(action):
            style, action = _silent(style), _silent(action)
        if mem_clip is not None:
            who = ("The two people look exactly as in <Video 1>: same faces, hair and clothing" if cast > 1 else
                   "The main person looks exactly as in <Video 1> (a close-up of their face): same face, hair and clothing")
            prompt = (who + ". The place, the pose and the framing come from the text, not from <Video 1>"
                      + ((". Their voices sound like <Audio 1>." if cast > 1 else ". Their voice sounds like <Audio 1>.") if mem_clip[1] is not None else ".")
                      + NL + NL + style + NL + NL + action)
        elif mem is not None:
            prompt = (
                "subject_definitions:" + NL + "<Subject 1> is the person shown in <Picture 1>." + NL + NL
                + "summary: [reference generation] The target video follows <Subject 1>." + NL + NL
                + "retention_analysis:" + NL + "<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair and "
                "clothing of <Subject 1> are retained. Only the person comes from <Picture 1>; the place, the pose "
                "and the framing are new." + NL + NL + "detailed_description: " + style + NL + NL + action)
        else:
            prompt = (style + NL + NL + action).strip()
        if prompt_format.startswith("sections"):
            ref_kind = ("video_audio" if mem_clip[1] is not None else "video") if mem_clip is not None else "picture" if mem is not None else None
            if shot_refs:
                prompt = _h3_sections_refs(style, action, shot_refs["pictures"], ref_kind if mem_clip is not None else None, cast)
            else:
                prompt = _h3_sections(style, action, ref_kind, cast, focus if cast > 1 else None)
        elif shot_refs:
            prompt = shot_refs["text"] + NL + NL + prompt

        images, keyframes, ref_items, refs_out = [], [], [], []
        for p in (shot_refs or {}).get("pictures", []):  # pictures come first in H3's reference order
            pic = p["image"][:1]
            rw, rh = _ref_size(pic.shape[1], pic.shape[2], width, height)
            pic_r = _resize(pic, rw, rh, "disabled")
            ref_items.append({"type": "image", "data": pic_r})
            refs_out.append({"kind": "image", "latent_h": rh // 16, "latent_w": rw // 16, "latent": vae.encode(pic_r),
                             "lc_time_gap": memory_gap * TIME_PER_SECOND})
        if mem_clip is not None:
            vfr, vaud = mem_clip
            if vaud is not None:
                ref_items.append({"type": "audio"})
            qwen = vfr[list(range(0, vfr.shape[0], FPS // 2))]
            ref_items.append({"type": "video", "data": qwen, "timestamps": [k / 2.0 for k in range(qwen.shape[0])]})
            z = vae.encode(vfr)
            a_lat, a_t = _encode_ref_audio(audio_vae, vaud) if vaud is not None else (None, 0)
            refs_out.append({"kind": "video_audio" if a_t else "video", "latent_t": z.shape[2], "latent_h": vfr.shape[1] // 16,
                         "latent_w": vfr.shape[2] // 16, "ref_audio_t": a_t, "latent": z, "audio_latent": a_lat,
                         "lc_time_gap": memory_gap * TIME_PER_SECOND})
        elif mem is not None:
            rw, rh = _ref_size(mem.shape[1], mem.shape[2], width, height)
            mem_r = _resize(mem, rw, rh, "disabled")
            ref_items.append({"type": "image", "data": mem_r})
            refs_out.append({"kind": "image", "latent_h": rh // 16, "latent_w": rw // 16, "latent": vae.encode(mem_r),
                         "lc_time_gap": memory_gap * TIME_PER_SECOND})
        if cont:
            tail_mp4 = _seg_path(folder, i - 1, "_tail.mp4")
            tail_wav = _seg_path(folder, i - 1, "_tail.wav")
            if not os.path.exists(tail_mp4):
                raise ValueError(f"[LC H3] Segment {i} continues segment {i - 1}, which isn't saved yet. Make segment {i - 1} first.")
            tail = _read_mp4_frames(tail_mp4, width, height)[:ovl_frames]
            ref_path = _seg_path(folder, _scene_start(plan, i - 1, folder), "_tail.mp4")
            ref = _read_mp4_frames(ref_path, width, height) if color_match > 0 and os.path.exists(ref_path) else None
            tail = _color_match(tail, ref, color_match)
            kf = {"resolved_frame_index": 0, "latent": vae.encode(tail)}
            if os.path.exists(tail_wav):
                a_lat, _ = _encode_ref_audio(audio_vae, _read_wav(tail_wav))
                kf["audio_latent"] = a_lat
            keyframes.append(kf)
        elif first_frame is not None:
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "latent": vae.encode(img)})

        if ref_items:
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        elif images:
            tokens = clip.tokenize(prompt, images=images)
        else:
            tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
        if refs_out:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": refs_out})

        segment = {"project": plan["project"], "folder": folder, "index": i, "count": len(shots), "mode": "CONTINUE" if cont else "CUT",
                   "seed": int(seed), "prompt": prompt, "width": width, "height": height, "frames": frame_count,
                   "overlap_frames": ovl_frames, "memory": "clip" if mem_clip is not None else (mem is not None), "reanchor": reanchor, "scene_start": _scene_start(plan, i, folder), "lead_frames": 0, "redo_after": redo_after.startswith("this +"),
                   "refs": [f"<Picture {p['n']}> {p['role']} {p['name']}".strip() for p in (shot_refs or {}).get("pictures", [])],
                   "from_tail_mtime": os.path.getmtime(_seg_path(folder, i - 1, "_tail.mp4")) if cont else None}
        info = (f"Segment {i} of {len(shots)} | {segment['mode']} | {frame_count} frames | seed {seed}"
                + (f" | starts from the end of segment {i - 1}" if cont else "") + (" | memory clip" if mem_clip is not None else " | memory on" if mem is not None else "")
                + (f" (face re-anchored: {reanchor})" if mem_clip is not None and reanchor else "")
                + (" | refs: " + ", ".join(f"<Picture {p['n']}> {p['name'] or p['role']}" for p in shot_refs["pictures"]) if shot_refs else ""))
        print(f"[LC H3] {info}")
        return (cond, latent, int(seed), segment, info, prompt)


def _cast_refs(folder: str, name: str) -> str | None:
    """Path to one person's face references (from their first solo shot), made from their saved head clip when an
    older project has the clip but no references yet. None when there is nothing to compare with."""
    path = _cast_path(folder, name, "_refs.npy")
    if os.path.exists(path):
        return path
    clip = _cast_path(folder, name, ".mp4")
    if not os.path.exists(clip) or _face_models() is None:
        return None
    frames = (_read_mp4_native(clip) * 255.0).round().to(torch.uint8).numpy()
    refs = _face_refs(frames)
    if refs is None:
        return None
    np.save(path, refs)
    return path


def _solo_name(segment: dict) -> str | None:
    """The one named person this segment is of ("Camera: close-up of Daniel"), from the project's plan."""
    try:
        plan = json.load(open(os.path.join(segment["folder"], "plan.json"), encoding="utf-8"))
    except Exception:
        return None
    return _solo(segment.get("prompt", ""), _cast_names(plan))


def _save_cast_memory(segment: dict, frames_u8: np.ndarray, wave, sr: int):
    """A shot of one named person ("Camera: close-up of Daniel"): keep about 3 s of their head and their voice, the
    loudest stretch (their line), as that person's memory. Later shots of them get it, so the face and voice match
    and nobody else is pulled into their close-up. Only the first solo shot of each person is kept."""
    folder = segment["folder"]
    try:
        plan = json.load(open(os.path.join(folder, "plan.json"), encoding="utf-8"))
    except Exception:
        return
    name = _solo(segment.get("prompt", ""), _cast_names(plan))
    if not name or os.path.exists(_cast_path(folder, name, ".mp4")):
        return
    n, win = frames_u8.shape[0], 73  # 3 s, on H3's 17k+5 frame grid
    if n < win:
        return
    c0 = 0
    if wave is not None and sr:
        per = sr / FPS
        energy = [float(np.abs(wave[:, int(k * per):int((k + win) * per)]).mean()) for k in range(0, n - win + 1, 6)]
        c0 = int(np.argmax(energy)) * 6
    head = _solo_head_crop(frames_u8[c0:c0 + win])
    if head is None:
        return
    _encode(head, _cast_path(folder, name, ".mp4"), 8, "yuv444p")
    refs = _face_refs(frames_u8[c0:c0 + win])  # their own face, for checking their later shots
    if refs is not None:
        np.save(_cast_path(folder, name, "_refs.npy"), refs)
    if wave is not None and sr:
        _write_wav(_cast_path(folder, name, ".wav"), wave[:, int(c0 * sr / FPS):int((c0 + win) * sr / FPS)], sr)
    print(f"[LC H3] Saved {name}'s memory (face and voice) from segment {segment['index']}.")


_TAKE_FILES = (".mp4", "_part.mp4", "_tail.mp4", "_tail.wav", ".json", "_latent.safetensors", "_up.mp4")


def _keep_take(folder: str, i: int, move: bool = False) -> str:
    """Put a saved segment's files in takes/ as seg_00N_takeK (copied, or moved so the segment counts as not made
    yet). Returns the take's name."""
    takes = os.path.join(folder, "takes")
    os.makedirs(takes, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(takes, f"seg_{i:03d}_take{k}.mp4")):
        k += 1
    for suf in _TAKE_FILES:
        src = _seg_path(folder, i, suf)
        if os.path.exists(src):
            dst = os.path.join(takes, f"seg_{i:03d}_take{k}{suf}")
            shutil.move(src, dst) if move else shutil.copy2(src, dst)
    return f"seg_{i:03d}_take{k}"


def _redo(segment: dict) -> str:
    """A segment made again: keep the old take, and with redo_after also set aside the CONTINUE segments that start
    from its ending (they no longer join), so index 0 makes them next."""
    folder, i = segment["folder"], segment["index"]
    if not _is_done(folder, i):
        return ""
    note = f" | redo: old take kept as takes/{_keep_take(folder, i)}"
    if os.path.exists(_seg_path(folder, i, "_up.mp4")):  # the old upscale is of the old take: make it again
        os.remove(_seg_path(folder, i, "_up.mp4"))
        note += " (its upscale too: run LC H3 Upscale for it again)"
    if segment.get("redo_after"):
        nxt = []
        k = i + 1
        while k <= segment["count"] and _is_done(folder, k):
            with open(_seg_path(folder, k, ".json"), encoding="utf-8") as f:
                if json.load(f).get("mode") != "CONTINUE":
                    break
            _keep_take(folder, k, move=True)
            nxt.append(k)
            k += 1
        if nxt:
            note += (f" | segment{'s' if len(nxt) > 1 else ''} {', '.join(map(str, nxt))} set aside (they continue from this one):"
                     f" set index to 0 and queue {len(nxt)} more")
    return note


def _write_segment(segment: dict, frames_u8: np.ndarray, wave, sr: int, crf: int) -> str:
    """Write one segment's files (preview, part, tail, json with the face and color checks). Returns the info line."""
    folder, i, ovl = segment["folder"], segment["index"], segment["overlap_frames"]
    os.makedirs(folder, exist_ok=True)
    redo_note = _redo(segment)
    n = frames_u8.shape[0]
    per_frame = sr / FPS if sr else 0

    def _cut(a, b):
        return {"waveform": torch.from_numpy(wave[:, int(a * per_frame):int(b * per_frame)]).unsqueeze(0), "sample_rate": sr} if wave is not None else None

    start = ovl if segment["mode"] == "CONTINUE" else 0
    _encode(frames_u8, _seg_path(folder, i, ".mp4"), crf, audio=_cut(0, n))
    _encode(frames_u8[start:], _seg_path(folder, i, "_part.mp4"), crf, audio=_cut(start, n))
    # the tail is re-encoded as the next segment's guide, so keep it near lossless
    _encode(frames_u8[n - ovl:], _seg_path(folder, i, "_tail.mp4"), 8, "yuv444p")
    if wave is not None:
        _write_wav(_seg_path(folder, i, "_tail.wav"), wave[:, int((n - ovl) * per_frame):], sr)
    if i == 1:
        from PIL import Image
        bi = _best_face_frame(frames_u8)
        bi = _sharpest(frames_u8) if bi is None else bi
        best = frames_u8[bi]
        Image.fromarray(np.ascontiguousarray(_portrait_crop(best))).save(_seg_path(folder, i, "_memory.png"))
        Image.fromarray(best).save(_seg_path(folder, i, "_face_ref.png"))
        refs = _face_refs(frames_u8)
        if refs is not None:
            np.save(_seg_path(folder, i, "_face_refs.npy"), refs)
        c0 = min(max(0, bi - 11), max(0, n - 22))  # the memory clip is centred on the best face too
        _encode(frames_u8[c0:c0 + 22], _seg_path(folder, i, "_memory_clip.mp4"), 8, "yuv444p")
        head = _head_crop(frames_u8[c0:c0 + 22])
        if head is not None:
            _encode(head, _seg_path(folder, i, "_memory_head.mp4"), 8, "yuv444p")
        if wave is not None:
            _write_wav(_seg_path(folder, i, "_memory_clip.wav"), wave[:, int(c0 * per_frame):int((c0 + 22) * per_frame)], sr)
    _save_cast_memory(segment, frames_u8, wave, sr)
    solo = _solo_name(segment) if i > 1 else None
    own = _cast_refs(folder, solo) if solo else None
    face = _face_score(frames_u8, own or _seg_path(folder, 1, "_face_refs.npy"))
    if face is not None and own:
        face["who"] = solo  # matched against this person's own face, not segment 1's
    meta = {**segment, "decoded_frames": n, "part_frames": n - start, "look": _look(frames_u8[n - ovl:]),
            "face": face, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(_seg_path(folder, i, ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)

    info = f"Saved segment {i} of {segment['count']} ({n - start} new frames) to {folder}"
    if face and i > 1 and face.get("match") is None:
        info += (". No clear face of the main person to check (the clear face was someone else's)" if face.get("others", 0) >= MIN_FACE_FRAMES
                 else ". No clear face to check (seen from behind or too small)")
    elif face and i > 1:
        verdict = "same person" if face["match"] >= FACE_SAME else "close" if face["match"] >= FACE_CLOSE else "⚠️ looks like someone else, try another seed"
        info += f". Face match{' to ' + face['who'] if face.get('who') else ''} {face['match']:.2f} ({verdict})"
    if i < segment["count"]:
        info += f". Next: segment {i + 1}."
    else:
        info += ". All segments done: run LC H3 Assemble."
    return info + redo_note


class LCH3SegmentSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment": (SEGMENT_TYPE, {"tooltip": "From LC H3 Segment 🎞️."}),
                "samples": ("LATENT", {"tooltip": "The sampled AV latent from your sampler."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 40, "tooltip": "mp4 quality. Lower = better and bigger. 16 is close to lossless."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = "Decodes and saves one segment: a preview mp4, the part for the final video (overlap cut off) and the ending the next segment starts from."

    def save(self, segment, samples, vae, audio_vae, crf):
        latent = samples["samples"]
        if not getattr(latent, "is_nested", False):
            raise ValueError("[LC H3] Segment Save expects a MiniMax H3 AV latent.")
        video_lat, audio_lat = latent.unbind()[:2]
        # the sampler's own latent, for LC H3 Upscale later (it works from the latent, not the mp4)
        os.makedirs(segment["folder"], exist_ok=True)
        comfy.utils.save_torch_file({"video": video_lat.contiguous().cpu(), "audio": audio_lat.contiguous().cpu()},
                                    _seg_path(segment["folder"], segment["index"], "_latent.safetensors"))
        images = vae.decode(video_lat)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        audio = vae_decode_audio(audio_vae, samples)
        wave, sr = audio_parts(audio)
        lead = int(segment.get("lead_frames", 0))
        if lead:
            images = images[lead:]
            if wave is not None:
                wave = wave[:, int(round(lead * sr / FPS)):]
                audio = {"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": sr}

        frames_u8 = (images.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        info = _write_segment(segment, frames_u8, wave, sr, crf)
        print(f"[LC H3] {info}")
        return {"ui": {"text": [info]}, "result": (images, audio, info)}


def _read_audio(path: str, sr: int = 48000) -> np.ndarray:
    """mp4 audio -> float32 [2, T] at sr (zeros if the file has no audio)."""
    cmd = [_ffmpeg_bin(), "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy() if raw else np.zeros((2, 0), np.float32)


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame[::8, ::8].astype(np.float32).mean(axis=2)


class LCH3Assemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {"tooltip": "From LC Vision Director 🎬 (a saved plan loads instantly)."}),
                "blend_frames": ("INT", {"default": 8, "min": 0, "max": 30, "tooltip": "CONTINUE joins: crossfades this many frames of the old ending into the remade start (picture and sound), so the join doesn't pop. 0 = hard join."}),
                "seam_fade_ms": ("INT", {"default": 15, "min": 0, "max": 200, "tooltip": "CUT joins: a tiny audio fade so the cut doesn't click."}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 40}),
            },
            "optional": {
                "after": ("STRING", {"forceInput": True, "tooltip": "Wire LC H3 Segment Save's info here, so the join runs after the new segment is saved (in the same workflow)."}),
                "version": (["normal", "upscaled"], {"default": "normal", "tooltip": "upscaled: joins the LC H3 Upscale copies instead (every segment needs one) into <project>_full_upscaled.mp4."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "seam_report")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("Joins every saved segment into one mp4, crossfading each CONTINUE join over the shared overlap, "
                   "and reports how smooth every join is. Streams frames to ffmpeg, never the whole video in RAM.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def assemble(self, plan, blend_frames, seam_fade_ms, crf, after=None, version="normal"):
        folder = project_dir(plan["project"])
        segs = []
        for i in range(1, len(plan.get("shots", [])) + 1):
            if not _is_done(folder, i):
                break
            with open(_seg_path(folder, i, ".json"), encoding="utf-8") as f:
                segs.append(json.load(f))
        if not segs:
            raise ValueError(f"[LC H3] No saved segments in {folder}.")
        up = version == "upscaled"
        if up:
            missing = [s_["index"] for s_ in segs if not s_.get("up") or not os.path.exists(_seg_path(folder, s_["index"], "_up.mp4"))]
            if missing:
                raise ValueError(f"[LC H3] Not upscaled yet: segment{'s' if len(missing) > 1 else ''} {', '.join(map(str, missing))}. "
                                 "Run LC H3 Upscale for them first (index 0 takes the next one), or set version to normal.")
            w, h = segs[0]["up"]["width"], segs[0]["up"]["height"]
        else:
            w, h = segs[0]["width"], segs[0]["height"]
        sr = 48000
        spf = sr / FPS
        ff = _ffmpeg_bin()
        dest = os.path.join(folder, f"{os.path.basename(folder)}_full{'_upscaled' if up else ''}.mp4")
        tmp = tempfile.mkdtemp(prefix="lc_h3_")
        video = os.path.join(tmp, "v.mp4")
        enc = subprocess.Popen([ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-crf", str(int(crf)), "-pix_fmt", "yuv420p", video], stdin=subprocess.PIPE)
        audio_out, steps, joins = [], [], []
        state = {"written": 0, "prev": None}

        def emit(frames):
            for fr in frames:
                g = _gray(fr)
                if state["prev"] is not None:
                    steps.append(float(np.abs(g - state["prev"]).mean()))
                state["prev"] = g
            enc.stdin.write(np.ascontiguousarray(frames).tobytes())
            state["written"] += len(frames)

        def fade(a, d, out):
            if d and a.shape[1] > d:
                a = a.copy()
                ramp = np.linspace(1, 0, d, dtype=np.float32) if out else np.linspace(0, 1, d, dtype=np.float32)
                if out:
                    a[:, -d:] *= ramp
                else:
                    a[:, :d] *= ramp
            return a

        try:
            held_v, held_a = None, None  # the previous segment's last frames / audio, held back for blending
            for k, seg in enumerate(segs):
                i = seg["index"]
                full = _seg_path(folder, i, "_up.mp4" if up else ".mp4")
                frames = (_read_mp4_frames(full, w, h) * 255.0).round().to(torch.uint8).numpy()
                aud = _read_audio(full, sr)
                n, ovl = len(frames), int(seg.get("overlap_frames", 0))
                nxt = segs[k + 1] if k + 1 < len(segs) else None
                if seg["mode"] == "CONTINUE" and k > 0:
                    # the old ending and the remade start show the same moments: crossfade them, then go on with new frames
                    nb = len(held_v) if held_v is not None else 0
                    if nb:
                        t = (np.arange(1, nb + 1, dtype=np.float32) / (nb + 1))[:, None, None, None]
                        emit((held_v * (1 - t) + frames[ovl - nb:ovl] * t).round().astype(np.uint8))
                        new_a = aud[:, int(round((ovl - nb) * spf)):int(round(ovl * spf))]
                        ln = min(new_a.shape[1], held_a.shape[1])
                        ta = np.linspace(0, 1, ln, dtype=np.float32)
                        audio_out.append(held_a[:, :ln] * np.cos(ta * np.pi / 2) + new_a[:, :ln] * np.sin(ta * np.pi / 2))
                    start = ovl
                    joins.append((i, "CONTINUE", state["written"]))
                else:
                    start = 0
                    if k > 0:
                        d = int(sr * seam_fade_ms / 1000)
                        audio_out[-1] = fade(audio_out[-1], d, True)
                        aud = fade(aud, d, False)
                        joins.append((i, "CUT", state["written"]))
                # hold back this segment's last frames when the next segment will blend into them
                keep = min(blend_frames, int(nxt.get("overlap_frames", 0))) if nxt is not None and nxt["mode"] == "CONTINUE" else 0
                body_end = n - keep
                emit(frames[start:body_end])
                audio_out.append(aud[:, int(round(start * spf)):int(round(body_end * spf))])
                held_v = frames[body_end:] if keep else None
                held_a = aud[:, int(round(body_end * spf)):int(round(n * spf))] if keep else None
            enc.stdin.close()
            enc.wait()
            wav = os.path.join(tmp, "a.wav")
            _write_wav(wav, np.concatenate(audio_out, axis=1), sr)
            subprocess.run([ff, "-y", "-v", "error", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-movflags", "+faststart", dest], check=True, capture_output=True)
        finally:
            if enc.poll() is None:
                enc.kill()
            shutil.rmtree(tmp, ignore_errors=True)

        # seam check: the biggest frame-to-frame jump around each join vs the normal motion nearby
        lines, report = [], []
        for i, mode, pos in joins:
            # the steps across the join (or across the blend) vs the 10 steps either side of it
            lo = max(0, pos - (blend_frames if mode == "CONTINUE" else 0) - 1)
            win = steps[lo:pos]
            around = steps[max(0, lo - 10):lo] + steps[pos:pos + 10]
            if not win or not around:
                continue
            ratio = max(win) / max(1e-3, float(np.median(around)))
            flag = mode == "CONTINUE" and ratio > 1.6
            report.append({"segment": i, "mode": mode, "time_s": round(pos / FPS, 2), "jump": round(ratio, 2), "flag": flag})
            if mode == "CONTINUE":
                lines.append(f"{pos / FPS:6.2f} s  segment {i - 1} -> {i}: {ratio:.2f}x normal motion" + ("  ⚠️ check this join" if flag else ""))
            else:
                lines.append(f"{pos / FPS:6.2f} s  segment {i - 1} -> {i}: cut")
        stale = []
        for s_ in segs:
            prev_tail = _seg_path(folder, s_["index"] - 1, "_tail.mp4")
            made_from = s_.get("from_tail_mtime")
            if s_["mode"] == "CONTINUE" and made_from and os.path.exists(prev_tail) and abs(os.path.getmtime(prev_tail) - made_from) > 1:
                stale.append(s_["index"])
        # drift check: each segment's ending vs the ending of the segment its scene started with
        looks = {s["index"]: s for s in segs if s.get("look")}
        drift_lines, drift = [], []
        for s in segs:
            k = s.get("scene_start", s["index"])
            if s["mode"] != "CONTINUE" or s["index"] not in looks or k not in looks:
                continue
            a, b = looks[k]["look"], s["look"]
            de = math.sqrt((a["L"] - b["L"]) ** 2 + (a["a"] - b["a"]) ** 2 + (a["b"] - b["b"]) ** 2)
            warm = (b["a"] - a["a"]) + (b["b"] - a["b"])
            flag = de > 6.0
            drift.append({"segment": s["index"], "scene_start": k, "delta_e": round(de, 2), "warmer_by": round(warm, 2),
                          "brightness": round(b["L"] - a["L"], 2), "flag": flag})
            drift_lines.append(f"segment {s['index']} vs {k}: color shift {de:.1f}"
                               + (f" (warmer +{warm:.1f})" if warm > 2 else f" (cooler {warm:.1f})" if warm < -2 else "")
                               + ("  ⚠️ drifting: try color_match 0.3 on the next segment" if flag else ""))
        # face check: how much each segment's face matches segment 1's
        faces, face_lines = [], []
        for s in segs:
            fc = s.get("face")
            if s["index"] == 1 or not fc or fc.get("match") is None:
                continue
            m = fc["match"]
            flag = m < FACE_CLOSE
            faces.append({"segment": s["index"], "mode": s["mode"], "match": m, "low": fc.get("low"), "flag": flag, "who": fc.get("who")})
            face_lines.append(f"segment {s['index']} ({s['mode'].lower()}{', ' + fc['who'] if fc.get('who') else ''}): {m:.2f} "
                              + ("same person" if m >= FACE_SAME else "close" if m >= FACE_CLOSE else "⚠️ looks like someone else, redo with another seed"))
        with open(os.path.join(folder, "seams.json"), "w", encoding="utf-8") as f:
            json.dump({"joins": report, "drift": drift, "faces": faces}, f, indent=1)
        seam_text = ("Join smoothness (1.00x = as smooth as the rest of the video, over 1.60x is flagged):\n" + "\n".join(lines)) if lines else "No joins yet."
        if stale:
            seam_text += NL + NL + "⚠️ " + ", ".join(f"segment {k}" for k in stale) + (
                " continued from an older version of the segment before it (that one was redone). Redo "
                + ("it" if len(stale) == 1 else "them") + " so the join lines up.")
        if drift_lines:
            seam_text += ("\n\nColor drift inside each scene (under 3 = invisible, over 6 is flagged; a scene that walks "
                          "into new light on purpose will read high):\n" + "\n".join(drift_lines))
        if face_lines:
            seam_text += (NL + NL + f"Face match to segment 1, or to that person's first close-up for a shot of one person (0 to 1; {FACE_SAME:.2f} and up = same person, "
                          f"{FACE_CLOSE:.2f} to {FACE_SAME:.2f} = close, under {FACE_CLOSE:.2f} = someone else):" + NL + NL.join(face_lines))
        written = state["written"]
        info = f"Joined {len(segs)} segment(s) into {dest} ({written} frames, {written / FPS:.2f} s)"
        print(f"[LC H3] {info}\n{seam_text}")
        return {"ui": {"text": [info + "\n\n" + seam_text]}, "result": (dest, seam_text)}


def _face_track(frames_u8: np.ndarray) -> list:
    """Per frame the main person's face box (x0, y0, x1, y1) or None, smoothed over time and filled across short
    gaps. The main person = the largest face, any angle (profiles included: their face still needs fixing)."""
    pair = _face_models()
    if pair is None:
        return [None] * len(frames_u8)
    det = pair[0]
    raw = []
    for fr in frames_u8:
        bgr = np.ascontiguousarray(fr[..., ::-1])
        h, w = bgr.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(bgr)
        if faces is None or not len(faces):
            raw.append(None)
            continue
        f = max(faces, key=lambda r: r[2] * r[3])
        raw.append(None if f[2] * f[3] < 0.003 * w * h else (float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3])))
    # fill gaps up to 12 frames from the nearest box, then a 7-frame moving average so the mask doesn't jitter
    filled = list(raw)
    for k in range(len(raw)):
        if raw[k] is None:
            near = [(abs(j - k), raw[j]) for j in range(max(0, k - 12), min(len(raw), k + 13)) if raw[j] is not None]
            filled[k] = min(near)[1] if near else None
    out = []
    for k in range(len(filled)):
        win = [b for b in filled[max(0, k - 3):k + 4] if b is not None]
        out.append(tuple(float(np.mean([b[c] for b in win])) for c in range(4)) if win and filled[k] is not None else None)
    return out


def _face_mask_latent(track: list, frames: int, lat_t: int, lat_h: int, lat_w: int, px_h: int, px_w: int,
                      grow: float = 1.7) -> torch.Tensor:
    """The face boxes as a soft video-latent mask [1, 1, T, H, W]: 1 = redraw, 0 = keep. Each latent frame covers
    1 pixel frame (the first) then 4 (FRAME_PER_TOKEN), and takes the union of its frames' boxes."""
    mask = torch.zeros((1, 1, lat_t, lat_h, lat_w))
    ys = (torch.arange(lat_h).float() + 0.5) * px_h / lat_h
    xs = (torch.arange(lat_w).float() + 0.5) * px_w / lat_w
    for t in range(lat_t):
        f0, f1 = (0, 1) if t == 0 else (1 + 4 * (t - 1), min(frames, 1 + 4 * t))
        for b in track[f0:f1]:
            if b is None:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            rx, ry = (b[2] - b[0]) * grow / 2, (b[3] - b[1]) * grow / 2 * 1.15  # a little taller: hair and chin
            d = ((xs[None, :] - cx) / rx) ** 2 + ((ys[:, None] - cy) / ry) ** 2
            soft = (1.0 - (d - 0.6) / 0.4).clamp(0, 1)  # solid core, feathered rim
            mask[0, 0, t] = torch.maximum(mask[0, 0, t], soft)
    return mask


def _crop_boxes(track: list, w: int, h: int, grow: float) -> list:
    """One square crop (x0, y0, side) per frame around the face, heavily smoothed so the redrawn close-up doesn't
    shake, and kept inside the frame."""
    known = [b for b in track if b is not None]
    fill = []
    last = known[0]
    for b in track:
        last = b if b is not None else last
        fill.append(last)
    cx = np.array([(b[0] + b[2]) / 2 for b in fill])
    cy = np.array([(b[1] + b[3]) / 2 for b in fill])
    sd = np.array([max(b[2] - b[0], b[3] - b[1]) * grow * 1.35 for b in fill])

    def smooth(v, k=15):
        pad = np.pad(v, (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    cx, cy, sd = smooth(cx), smooth(cy), smooth(sd)
    out = []
    for x, y, side in zip(cx, cy, sd):
        side = float(min(side, w, h))
        x0 = float(min(max(0.0, x - side / 2), w - side))
        y0 = float(min(max(0.0, y - side / 2), h - side))
        out.append((x0, y0, side))
    return out


def _crop_frames(frames: torch.Tensor, crop: list, size: int) -> torch.Tensor:
    """IMAGE [N, H, W, 3] -> the per-frame crops resized to size x size."""
    out = []
    for k, (x0, y0, side) in enumerate(crop):
        x0i, y0i, si = int(round(x0)), int(round(y0)), max(16, int(round(side)))
        c = frames[k, y0i:y0i + si, x0i:x0i + si].movedim(-1, 0)[None]
        out.append(torch.nn.functional.interpolate(c, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)[0].movedim(0, -1))
    return torch.stack(out)


# Face Pass / Face Paste: not registered. Tests 2026-10-13 showed H3's reference is too weak to move the face's
# identity in a partial redraw; kept here for a later try.
class LCH3FacePass:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("STRING", {"default": "parka_test", "tooltip": "The LC_H3 project (output/LC_H3/<project>)."}),
                "index": ("INT", {"default": 2, "min": 2, "max": 60, "tooltip": "The saved segment to fix. Segment 1 is the face everyone is compared to, so it is never redrawn."}),
                "clip": ("CLIP", {"tooltip": "MiniMax H3 text encoder."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
                "mode": (["crop", "mask"], {"default": "crop", "tooltip": "crop: follow the face with a steady box, upscale it to crop_size and redraw only that (the face gets far more pixels). Needs LC H3 Face Paste after the sampler. mask: redraw the face inside the full frame (wire straight into LC H3 Segment Save)."}),
                "crop_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 32, "tooltip": "crop mode: the size the face crop is redrawn at."}),
                "grow": ("FLOAT", {"default": 1.7, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "How far around the face the redraw reaches. 1.7 takes in the hair line and the chin."}),
                "memory_gap": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", SEGMENT_TYPE, "STRING")
    RETURN_NAMES = ("positive", "latent", "segment", "info")
    FUNCTION = "build"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("Second pass for a saved segment: redraws only the main person's face (a soft mask that follows it) "
                   "with segment 1's face as the reference. Wire into your sampler with a low denoise (about 0.4), then "
                   "LC H3 Segment Save. Sound and everything outside the mask stay as they were.")

    @classmethod
    def IS_CHANGED(cls, project="", index=2, **kwargs):
        p = _seg_path(project_dir(project), index, ".mp4")
        return os.path.getmtime(p) if os.path.exists(p) else ""

    def build(self, project, index, clip, vae, audio_vae, mode, crop_size, grow, memory_gap):
        folder = project_dir(project)
        meta = _load_json(_seg_path(folder, index, ".json"))
        src = _seg_path(folder, index, ".mp4")
        if not meta or not os.path.exists(src):
            raise ValueError(f"[LC H3] Segment {index} of '{project}' isn't saved yet.")
        w, h = meta["width"], meta["height"]
        # keep the base version once, so a face pass can always be compared or undone
        for suffix in (".mp4", ".json", "_part.mp4", "_tail.mp4", "_tail.wav"):
            base = _seg_path(folder, index, "_base" + suffix)
            if os.path.exists(_seg_path(folder, index, suffix)) and not os.path.exists(base):
                shutil.copy2(_seg_path(folder, index, suffix), base)
        frames = _read_mp4_frames(_seg_path(folder, index, "_base.mp4"), w, h)
        n = frames.shape[0]
        while n > 5 and n % 17 != 5:
            n -= 1
        frames = frames[:n]
        frames_u8 = (frames * 255.0).round().to(torch.uint8).numpy()
        track = _face_track(frames_u8)
        covered = sum(1 for b in track if b is not None)
        if not covered:
            raise ValueError(f"[LC H3] No face found in segment {index}: nothing for the face pass to fix.")
        crop = None
        if mode == "crop":
            crop = _crop_boxes(track, w, h, grow)
            frames = _crop_frames(frames, crop, crop_size)
            track = [None if bx is None else ((bx[0] - c[0]) * crop_size / c[2], (bx[1] - c[1]) * crop_size / c[2],
                                              (bx[2] - c[0]) * crop_size / c[2], (bx[3] - c[1]) * crop_size / c[2])
                     for bx, c in zip(track, crop)]
            w = h = crop_size
        video = vae.encode(frames)
        audio = {"waveform": torch.from_numpy(_read_audio(_seg_path(folder, index, "_base.mp4"), 32000)).unsqueeze(0), "sample_rate": 32000}
        a_lat, _ = _encode_ref_audio(audio_vae, audio)
        vmask = _face_mask_latent(track, n, video.shape[2], video.shape[3], video.shape[4], h, w, 1.5 if crop else grow)
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, a_lat)),
                  "noise_mask": comfy.nested_tensor.NestedTensor((vmask.expand(1, video.shape[1], -1, -1, -1).contiguous(),
                                                                  torch.zeros_like(a_lat)))}

        # conditioning: the segment's own prompt, plus segment 1's head clip as the face reference
        prompt = meta.get("prompt", "")
        ref_items, refs = [], []
        head = _seg_path(folder, 1, "_memory_head.mp4")
        if os.path.exists(head):
            hv = _read_mp4_native(head)
            m = hv.shape[0]
            while m > 5 and m % 17 != 5:
                m -= 1
            hv = hv[:m]
            ref_items.append({"type": "video", "data": hv[list(range(0, m, FPS // 2))], "timestamps": [k / 2.0 for k in range(len(range(0, m, FPS // 2)))]})
            z = vae.encode(hv)
            refs.append({"kind": "video", "latent_t": z.shape[2], "latent_h": hv.shape[1] // 16, "latent_w": hv.shape[2] // 16,
                         "ref_audio_t": 0, "latent": z, "audio_latent": None, "lc_time_gap": memory_gap * TIME_PER_SECOND})
            prompt = ("The main person looks exactly as in <Video 1> (a close-up of their face): same face, hair and clothing. "
                      "The place, the pose and the framing come from the text, not from <Video 1>." + NL + NL
                      + ("A close-up of the main person's face filling the frame; the expression, mouth and head movement follow "
                         "the video." + NL + NL if crop else "") + prompt)
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items) if ref_items else clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if refs:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": refs})

        segment = {k: meta[k] for k in ("project", "folder", "index", "count", "mode", "seed", "prompt", "width", "height",
                                        "frames", "overlap_frames", "scene_start") if k in meta}
        segment.update({"lead_frames": 0, "face_pass": mode, "from_tail_mtime": meta.get("from_tail_mtime"),
                        "crop": crop, "crop_size": crop_size, "base": _seg_path(folder, index, "_base.mp4")})
        info = (f"Face pass ({mode}) for segment {index}: face found in {covered} of {n} frames"
                + (f", redrawn at {crop_size} x {crop_size}" if crop else f", mask grows {grow}x")
                + (". Wire the sampler into LC H3 Face Paste." if crop else ""))
        print(f"[LC H3] {info}")
        return (cond, latent, segment, info)


class LCH3FacePaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment": (SEGMENT_TYPE, {"tooltip": "From LC H3 Face Pass 🙂 (crop mode)."}),
                "samples": ("LATENT", {"tooltip": "The sampled face crop from your sampler."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 40}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "paste"
    OUTPUT_NODE = True
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = "Blends the redrawn face crop back into the segment (only the face, with a soft edge) and saves the segment."

    def paste(self, segment, samples, vae, crf):
        crop = segment.get("crop")
        if not crop:
            raise ValueError("[LC H3] Face Paste needs LC H3 Face Pass in crop mode.")
        lat = samples["samples"]
        faces = vae.decode(lat.unbind()[0] if getattr(lat, "is_nested", False) else lat)
        if faces.ndim == 5:
            faces = faces.reshape(-1, *faces.shape[-3:])
        w, h = segment["width"], segment["height"]
        base = _read_mp4_frames(segment["base"], w, h)
        n = min(len(base), len(faces), len(crop))
        size = faces.shape[1]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        # soft oval: solid over the face, fading out well before the crop's edge so no box shows
        alpha = (1.0 - ((xx / 0.78) ** 2 + ((yy + 0.02) / 0.86) ** 2 - 0.55) / 0.45).clamp(0, 1)[None, None]
        out = base.clone()
        for k in range(n):
            x0, y0, side = crop[k]
            x0i, y0i, si = int(round(x0)), int(round(y0)), max(16, int(round(side)))
            si = min(si, w - x0i, h - y0i)
            f = torch.nn.functional.interpolate(faces[k].movedim(-1, 0)[None], size=(si, si), mode="bicubic", align_corners=False).clamp(0, 1)[0]
            a = torch.nn.functional.interpolate(alpha, size=(si, si), mode="bilinear", align_corners=False)[0]
            region = out[k, y0i:y0i + si, x0i:x0i + si].movedim(-1, 0)
            out[k, y0i:y0i + si, x0i:x0i + si] = (region * (1 - a) + f * a).movedim(0, -1)
        frames_u8 = (out[:n].clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
        wave = _read_audio(segment["base"], 48000)
        info = _write_segment({k: v for k, v in segment.items() if k not in ("crop", "base")} | {"face_pass": "crop"}, frames_u8,
                              wave if wave.shape[1] else None, 48000, crf)
        print(f"[LC H3] {info}")
        return {"ui": {"text": [info]}, "result": (out[:n], info)}


NODE_CLASS_MAPPINGS = {
    "LCH3Segment": LCH3Segment,
    "LCH3SegmentSave": LCH3SegmentSave,
    "LCH3Assemble": LCH3Assemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCH3Segment": "LC H3 Segment (BETA) 🎞️",
    "LCH3SegmentSave": "LC H3 Segment Save (BETA) 💾",
    "LCH3Assemble": "LC H3 Assemble (BETA) 🧩",
}
