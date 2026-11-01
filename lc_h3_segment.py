"""
LC H3 Segment / LC H3 Segment Save / LC H3 Assemble
---------------------------------------------------
Makes a long MiniMax H3 video one short segment per queue run, from an
LC Vision Showrunner plan (output/LC_H3/<project>/plan.json).

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
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _encode_ref_audio, _resize

from .lc_av_media import audio_parts
from .lc_h3_refs import REFS_TYPE, subset as _refs_subset
from .lc_save_video import _ffmpeg_bin, _write_wav, comfy_tags, metadata_args

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
    """Same folder rule as LC Vision Showrunner."""
    safe = re.sub(r"[^A-Za-z0-9 _.-]", "_", (project or "").strip())
    if not safe.strip(" ."):  # "" / "." / ".." would be LC_H3 itself or the output folder
        safe = "untitled"
    return os.path.join(folder_paths.get_output_directory(), "LC_H3", safe)


_WORK_SEEN = set()


def work_dir(folder: str) -> str:
    """Everything a project makes along the way lives in <project>/takes/, so the top of the folder is only
    project.json, plan.json and the finished movie. An older project's loose files (and an earlier work/ folder) move
    in there the first time it is touched."""
    takes = os.path.join(folder, "takes")
    if folder not in _WORK_SEEN and os.path.isdir(folder):
        name = os.path.basename(folder)
        loose = [os.path.join(folder, f) for f in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, f)) and (re.match(r"seg_\d{3}", f) or f.startswith("cast_")
                                                                   or f == "seams.json" or f.startswith(name + "_full"))]
        old_work = os.path.join(folder, "work")
        if os.path.isdir(old_work):
            loose += [os.path.join(old_work, f) for f in os.listdir(old_work)]
        if loose:
            os.makedirs(takes, exist_ok=True)
            for f in loose:
                os.replace(f, os.path.join(takes, os.path.basename(f)))
            print(f"[LC H3] Tidied {os.path.basename(folder)}: {len(loose)} working files moved into takes/.")
        if os.path.isdir(old_work) and not os.listdir(old_work):
            os.rmdir(old_work)
        _WORK_SEEN.add(folder)
    if os.path.isdir(folder):
        os.makedirs(takes, exist_ok=True)
    return takes


def _seg_path(folder: str, index: int, suffix: str) -> str:
    return os.path.join(work_dir(folder), f"seg_{index:03d}{suffix}")


def _is_done(folder: str, index: int) -> bool:
    return os.path.exists(_seg_path(folder, index, ".json")) and os.path.exists(_seg_path(folder, index, "_part.mp4"))


def _warn_ui(project: str, message: str):
    """A problem the chain carries on through (the join failed): a sticky toast in the browser that queued the run."""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("lc_h3_join_failed", {"project": project, "message": message},
                                        getattr(PromptServer.instance, "client_id", None))
    except Exception:
        pass


def _tell_ui(data: dict):
    """Segment Save -> the browser: how many segments are left, so make: all remaining knows whether to queue the next."""
    try:
        from server import PromptServer
        # only the browser that queued this run: every open tab with the workflow would queue the next segment
        PromptServer.instance.send_sync("lc_h3_segment_saved", data, getattr(PromptServer.instance, "client_id", None))
    except Exception:
        pass


def _resolve_index(plan: dict, index: int) -> int:
    count = len(plan.get("shots", []))
    folder = project_dir(plan.get("project", ""))
    if index > 0:
        i = want = min(index, count)
        shots = plan.get("shots", [])
        # a CONTINUE needs the segment before it: step back to the first one of that run that isn't made yet
        while i > 1 and shots[i - 1].get("mode") == "CONTINUE" and not _is_done(folder, i - 1):
            i -= 1
        if i != want:
            print(f"[LC H3] index {index}: segment {want} carries on from segment {i}, which isn't made yet, so segment {i} is made first."
                  " Set index to 0 to make the segments in order.")
        return i
    for i in range(1, count + 1):
        if not _is_done(folder, i):
            return i
    return count


def _latent_tail(path: str, frames: int, width: int, height: int) -> dict | None:
    """The previous segment's tail sliced straight out of its saved latent, video and audio, instead of the mp4 -> VAE
    round trip (idea from NikoDemon80's H3 Motion Context). None = no usable latent: use the mp4 tail."""
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
    if not os.path.exists(path):
        return None
    try:
        sd = comfy.utils.load_torch_file(path)
        video, audio = sd["video"], sd["audio"]
    except Exception as e:  # a torn or older latent file: the mp4 tail still works
        print(f"[LC H3] latent tail: can't read {os.path.basename(path)} ({e}), using the mp4 tail")
        return None
    if video.ndim == 4:
        video, audio = video[None], audio[None]
    if video.shape[3] * 16 != height or video.shape[4] * 16 != width:
        print(f"[LC H3] latent tail: size changed ({video.shape[4] * 16}x{video.shape[3] * 16}), using the mp4 tail")
        return None
    steps, covered = 0, 0
    while covered < frames:
        covered += FRAME_PER_TOKEN[steps % 5]
        steps += 1
    start = video.shape[2] - steps
    if covered != frames or start < 0 or start % 5:
        print(f"[LC H3] latent tail: {frames} frames don't line up with the latent grid, using the mp4 tail")
        return None
    a_steps = int(round(frames * FRAME_RESCALE))
    print(f"[LC H3] latent tail: {steps} video steps ({frames} frames) + {a_steps} audio steps from the saved latent")
    return {"resolved_frame_index": 0, "latent": video[:, :, start:].float().contiguous(),
            "audio_latent": audio[..., -a_steps:].float().contiguous()}


def _frames(plan: dict):
    seg = int(round(plan["segment_s"] * FPS))
    ovl = int(round(plan["overlap_s"] * FPS))
    return seg, ovl


def _middle_frame(path: str, width: int, height: int, frames: int) -> torch.Tensor | None:
    """The middle frame of an mp4 (frames long) as IMAGE [1, H, W, 3], or None."""
    raw = subprocess.run([_ffmpeg_bin(), "-v", "error", "-ss", f"{max(0, frames // 2) / FPS:.3f}", "-i", path,
                          "-vf", f"scale={width}:{height}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    if len(raw) < width * height * 3:
        return None
    return torch.from_numpy(np.frombuffer(raw[:width * height * 3], np.uint8).reshape(1, height, width, 3).astype(np.float32) / 255.0)


def _place_of(action: str) -> str:
    line = next((ln for ln in str(action or "").split("\n") if ln.strip().lower().startswith("place:")), "")
    return _norm_words(line.split(":", 1)[1]) if ":" in line else ""


def _place_key(shot: dict) -> str:
    """Where a segment is: the Shot List's place name when the plan has one, else its Place line."""
    return "@" + _norm_words(shot["place_name"]) if shot.get("place_name") else _place_of(shot.get("action", ""))


def _same_place(a: str, b: str) -> bool:
    import difflib

    if a.startswith("@") or b.startswith("@"):  # place names: exact
        return a == b
    return bool(a and b) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.75)


def _place_groups(plan: dict) -> list:
    """A place number per shot (None = the shot doesn't say). Place names group exactly. Without them, Place lines
    group by wording, and a one-place story keeps one place: when one group holds at least two thirds of the shots and
    the rest are odd ones out (a Place line the planner filled with camera words), those join it."""
    shots = plan.get("shots", [])
    keys = [_place_key(sh) for sh in shots]
    reps, ids = [], []
    for k in keys:
        if not k:
            ids.append(None)
            continue
        g = next((n for n, r in enumerate(reps) if _same_place(k, r)), None)
        if g is None:
            reps.append(k)
            g = len(reps) - 1
        ids.append(g)
    if not any(sh.get("place_name") for sh in shots) and len(reps) > 1:
        sizes = [ids.count(g) for g in range(len(reps))]
        big = max(range(len(reps)), key=lambda g: sizes[g])
        if sizes[big] * 3 >= 2 * sum(sizes) and all(sizes[g] == 1 for g in range(len(reps)) if g != big):
            ids = [None if g is None else big for g in ids]
    return ids


_OUTDOOR_RE = re.compile(r"\b(outside|outdoors?|street|sidewalk|curb|road|highway|alley|parking|exterior|plaza|"
                         r"beach|park|field|forest|desert|bridge)\b", re.I)
_MOVING_RE = re.compile(r"\b(track(s|ing)?|follow(s|ing)?|chas(e|es|ing)|pan(s|ning)?|run(s|ning)?|sprint(s|ing)?|"
                        r"rac(e|es|ing)|jump(s|ing)?|driv(e|es|ing)|drove|pulls? up|screech(es|ing)?|speed(s|ing)?|"
                        r"tears? away|dives?|flee(s|ing)?|crash(es|ing)?)\b", re.I)


def _moving_outdoors(shot: dict) -> bool:
    """An outdoor shot where things move (a van pulling up, people running): a layout anchor would freeze it into the
    anchor's frame, so it gets the anchor as setting (the place and its light) instead."""
    where = (shot.get("place_name") or "") + " " + _place_of(shot.get("action", ""))
    return bool(_OUTDOOR_RE.search(where)) and bool(_MOVING_RE.search(shot.get("action", "")))


_CLOSE_RE = re.compile(r"\b(extreme\s+)?close[- ]?ups?\b", re.I)


def _is_close_up(action: str) -> bool:
    cam = next((ln for ln in str(action or "").split("\n") if ln.strip().lower().startswith("camera:")), "")
    return bool(_CLOSE_RE.search(cam))


def _scene_anchor(plan: dict, folder: str, i: int, width: int, height: int, empty_first: bool = False):
    """hold_scene: the first saved shot AT THE SAME PLACE that isn't a close-up (that place's establishing shot), as
    (segment number, its middle frame, its part's mtime). Later shots there keep the room: who sits where, the table,
    the window. A new place gets its own anchor the first time it is made (the street = anchor 1, the café = anchor 2,
    and back to the street uses 1 again). empty_first: a shot whose Camera line names none of the cast wins (a
    setting anchor should show the place, not people who would walk into the new shot). None when there is none yet."""
    groups = _place_groups(plan)
    here = groups[i - 1]
    names = _cast_names(plan)
    order = list(range(1, i))
    if empty_first:
        def _cam(k):
            return next((ln for ln in plan["shots"][k - 1].get("action", "").split("\n") if ln.strip().lower().startswith("camera:")), "")
        order.sort(key=lambda k: any(re.search(rf"\b{re.escape(n)}\b", _cam(k), re.I) for n in names))
    for k in order:
        sh = plan["shots"][k - 1]
        if here is not None and groups[k - 1] != here:
            continue
        if not _is_close_up(sh.get("action", "")) and _is_done(folder, k):
            part = _seg_path(folder, k, "_part.mp4")
            with open(_seg_path(folder, k, ".json"), encoding="utf-8") as f:
                frames = int(json.load(f).get("part_frames", 85))
            img = _middle_frame(part, width, height, frames)
            if img is not None:
                return k, img, os.path.getmtime(part)
    return None


def _room_from(folder: str, i: int):
    """Segment i's Room from (plan.json, set with ✎ on the Proof Sheet): None = auto, 0 = none, k = segment k."""
    try:
        with open(os.path.join(folder, "plan.json"), encoding="utf-8") as f:
            v = json.load(f)["shots"][i - 1].get("anchor_from")
        return None if v is None else int(v)
    except Exception:
        return None


def _anchor_of(folder: str, k: int, width: int, height: int):
    """Segment k's middle frame as the room picture, like _scene_anchor, or None when it isn't made."""
    if not _is_done(folder, k):
        return None
    part = _seg_path(folder, k, "_part.mp4")
    with open(_seg_path(folder, k, ".json"), encoding="utf-8") as f:
        frames = int(json.load(f).get("part_frames", 85))
    img = _middle_frame(part, width, height, frames)
    return (k, img, os.path.getmtime(part)) if img is not None else None


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
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode:  # say why (e.g. the old file is open in a player), not just "exit status 1"
            err = res.stderr.decode("utf-8", "replace").strip()[-400:]
            raise RuntimeError(f"[LC H3] ffmpeg could not write {dest}: {err}")
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


def _pic_face_refs(images: list) -> np.ndarray | None:
    """SFace features of the largest clear face in each reference picture (the Reference Assembler's face pictures)."""
    pair = _face_models()
    if pair is None:
        return None
    # your own picture is a portrait you chose: any clear face counts (a beard or grey hair scores lower than the
    # 0.8 the video frames need), the largest one wins
    det, feats = pair[0], []
    det.setScoreThreshold(0.5)
    try:
        for img in images:
            rgb = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            det.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = det.detect(bgr)
            if faces is None or not len(faces):
                continue
            f = max(faces, key=lambda t: t[2] * t[3])
            feats.append(pair[1].feature(pair[1].alignCrop(bgr, f)).reshape(-1))
    finally:
        det.setScoreThreshold(0.7)
    return np.stack(feats) if feats else None


def _face_score(frames_u8: np.ndarray, refs_path):
    """How much the main person matches segment 1: per frame the best match of any face against any reference
    (so a passer-by or a new expression doesn't count against her), then the median over the segment. Frames where
    the only clear face is somebody else (under FACE_OTHER, e.g. the barista while she has her back to us) are
    left out instead of scored as drift."""
    pair = _face_models()
    if pair is None:
        return None
    if isinstance(refs_path, np.ndarray):
        refs = refs_path
    elif refs_path and os.path.exists(refs_path):
        refs = np.load(refs_path)
    else:
        return None
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


def _save_json(path: str, data, **kw):
    """Write json to a temp file, then swap it in: the Proof Sheet never reads a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, **kw)
    os.replace(tmp, path)


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


_PERSON_CLAUSE_RE = re.compile(r"\b(man|woman|boy|girl|guy|lady)\b", re.I)
_PEOPLE_WORDS_RE = re.compile(r"\b(he|she|him|her|his|hers|they|them|their|man|woman|men|women|boy|girl|guy|lady)\b", re.I)
_STOP = {"a", "an", "the", "in", "with", "and", "of", "his", "her", "wearing", "under", "on", "black", "dark", "man", "woman"}


def _style_cast(style: str, plan: dict, action: str) -> str:
    """The STYLE block goes into every segment, and every person it describes shows up. Keep only the people this
    segment names; a segment that names nobody and mentions no person (a van pulling up) keeps no one. Clauses that
    don't name anyone ("a stocky man in a grey hoodie") are matched to the cast by gender and look words. The camera
    and look clauses always stay."""
    names = _cast_names(plan)
    if len(names) < 1 or not style:
        return style
    here = {n for n in names if re.search(rf"\b{re.escape(n)}\b", action, re.I)}
    if not here and _PEOPLE_WORDS_RE.search(re.sub(r"[\"“][^\"”]*[\"”]", "", _action_body(action))):
        return style  # someone is in it, unnamed ("she walks in"): keep everyone
    story = plan.get("story", "")
    looks = {}
    for n in names:
        m = re.search(rf"\b{re.escape(n)},\s+an?\s+(.*?)(?=\b(?:{'|'.join(map(re.escape, names))}),\s+an?\s|$)", story, re.S)
        looks[n] = {w for w in re.findall(r"[a-z]+", (m.group(1) if m else "").lower()) if w not in _STOP}

    def owner(clause: str):
        named = [n for n in names if re.search(rf"\b{re.escape(n)}\b", clause, re.I)]
        if named:
            return named[0]
        g = _PERSON_CLAUSE_RE.search(clause)
        if not g:
            return None
        words = {w for w in re.findall(r"[a-z]+", clause.lower()) if w not in _STOP}
        same = [n for n in names if _GENDER.get(names[n], names[n]) == _GENDER.get(g.group(1).lower(), g.group(1).lower())]
        return max(same or names, key=lambda n: len(looks[n] & words))

    parts = re.split(r"(?<=[.;])\s+", style)
    keep = [x for x in parts if (o := owner(x)) is None or o in here]
    return " ".join(keep).strip()


_GENDER = {"man": "m", "guy": "m", "boy": "m", "woman": "f", "lady": "f", "girl": "f"}


def _action_body(action: str) -> str:
    """The timed part of a segment, after the Place / Holding / Sound / Camera lines."""
    return "\n".join(ln for ln in str(action or "").split("\n")
                     if not re.match(r"\s*(place|holding|sound)\s*:", ln, re.I))


def _beats_only(action: str) -> str:
    """The timed beats alone, without the Place / Holding / Sound / Camera lines ("close-up of Nova" names the shot,
    not who else is in it)."""
    return "\n".join(ln for ln in str(action or "").split("\n")
                     if not re.match(r"\s*(place|holding|sound|camera)\s*:", ln, re.I))


def _read_mp4_native(path: str) -> torch.Tensor:
    """mp4 -> IMAGE tensor at the file's own size."""
    # ffmpeg -i prints the stream size (ffprobe isn't always there, imageio-ffmpeg ships ffmpeg only)
    info = subprocess.run([_ffmpeg_bin(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    size = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info)
    if size is None:  # missing or broken clip: say which one
        raise ValueError(f"[LC H3] Can't read the video {path} (missing or broken).")
    w, h = (int(v) for v in size.groups())
    return _read_mp4_frames(path, w, h)


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
    trained ("Claire (S1) squeezes his hand and says <d>[English] ... </d>"), so the right mouth moves. Everyone the
    style block names is numbered in that order, speaking or not, so a person keeps the same number in every segment
    (a CONTINUE where only the second person speaks would otherwise call her S1, and the first speaker's mouth moves).
    Speakers the style block doesn't name follow in the order they speak."""
    quotes = list(_SPOKEN.finditer(text))
    who = [_speaker(text, m.start()) for m in quotes]
    names = [w[2] for w in who if w]
    low = style.lower()
    cast = [m.group(1).lower() for m in re.finditer(r"(?:^|[;.\n]\s*)([A-Z][a-z]+):", style.strip())]  # "Anika: ...; Nova: ..."
    order = list(dict.fromkeys(cast + names))
    order = sorted(order, key=lambda k: (low.find(k) if k in low else 10 ** 6, order.index(k)))
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


def _norm_words(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", str(t).lower().replace("’", "'")).split())


def _says_line(line_text: str, quotes: list) -> bool:
    """A recorded line matches a quote in the shot: the same words, give or take punctuation and a word or two."""
    import difflib

    a = _norm_words(line_text)
    return bool(a) and any(a == q or a in q or q in a or difflib.SequenceMatcher(None, a, q).ratio() >= 0.8
                           for q in (_norm_words(x) for x in quotes) if q)


def _line_time(action: str, line_text: str) -> float:
    """When the shot says this line: the "At N seconds" before its quote (0 when the shot gives no time)."""
    want = _norm_words(line_text)
    for m in _SPOKEN.finditer(action):
        if _says_line(want, [m.group(1)]):
            stamps = list(_STAMP_RE.finditer(action[:m.start()]))
            return float(stamps[-1].group(1)) if stamps else 0.0
    return 0.0


def _lock_line(latent: dict, audio_vae, audio: dict, start_s: float) -> dict:
    """LC H3 Audio Sync for one recorded line: the line goes into the sound half at start_s and only that stretch is
    held (noise mask 0); the rest of the sound and all of the picture are still made by H3. Several lines add up."""
    import comfy.audio
    from .lc_h3_audio_sync import AUDIO_LATENT_FPS

    video, a_cur = latent["samples"].unbind()[:2]
    t = a_cur.shape[-1]
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    wave, sr = audio["waveform"][:1], int(audio["sample_rate"])
    if sr != vae_sr:
        wave = comfy.audio.resample(wave, sr, vae_sr)
    want = int(round(t / AUDIO_LATENT_FPS * vae_sr))
    start = max(0, min(want, int(round(start_s * vae_sr))))
    piece = wave[..., :want - start]
    full = torch.zeros(1, wave.shape[1], want, dtype=wave.dtype)
    full[..., start:start + piece.shape[-1]] = piece
    z = audio_vae.encode(full.movedim(1, -1))
    z = z[..., :t] if z.shape[-1] >= t else torch.cat([z, z[..., -1:].repeat_interleave(t - z.shape[-1], dim=-1)], dim=-1)
    z = z.to(device=a_cur.device, dtype=a_cur.dtype)
    a0 = int(start_s * AUDIO_LATENT_FPS)
    a1 = min(t, a0 + math.ceil(piece.shape[-1] / vae_sr * AUDIO_LATENT_FPS) + 1)
    old = latent.get("noise_mask")
    v_mask, a_mask = (old.unbind()[:2] if old is not None else (torch.ones_like(video), torch.ones_like(a_cur)))
    a_mask = a_mask.clone()
    a_mask[..., a0:a1] = 0.0
    a_new = a_cur.clone()
    a_new[..., a0:a1] = z[..., a0:a1]
    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, a_new))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((v_mask, a_mask))
    return out


def _abs_input(path: str) -> str:
    """A plan picture path: absolute (output pictures) or relative to input/; "" outside ComfyUI's folders."""
    from .lc_h3_refs import inside_comfy

    return inside_comfy(path if os.path.isabs(path) else os.path.join(folder_paths.get_input_directory(), path))


def _shot_refs(refs: dict, shot: dict, style: str, genders: dict | None = None, one_place: bool = True) -> dict | None:
    """What one shot needs from the Reference Assembler, renumbered: the people named in its action (everyone when it
    names nobody but clearly has people in it) with their face, full body, outfit and voice; the settings and props
    it names (the only setting when there is just one); look and composition always; and a recorded line only when the
    shot says those words."""
    from .lc_h3_refs import PERSON

    pics = refs.get("pictures") or []
    auds = refs.get("audios") or []
    if not pics and not auds:
        return None
    action = shot.get("action", "")
    text = action + NL + style
    cast = sorted({p["name"] for p in pics + auds if p["role"] in PERSON and p["name"]})
    who = [n for n in cast if _mentions(action, n)]
    solo = _solo(action, genders or {n: "" for n in cast})
    if solo in cast:
        # a close-up of one person: only their pictures, or H3 pulls the others into the frame. Anyone else the action
        # names comes along: "she turns toward Anika" with no picture or look made H3 invent a stranger (Coffee Date)
        who = [solo] + [n for n in cast if n != solo and _mentions(_beats_only(action), n)]
    # people the story names but who have no picture (switched off) still count as named: "Vince ... he says" sent
    # the only man with a picture (the old guard) into the robbers' van
    if shot.get("subject") in cast:
        # LC H3 Story: a close-up or medium shot is its subject alone. "Anika stares at Nova" sent Nova's face and outfit
        # too, and Anika turned into Nova halfway through her own close-up (2026-11-01)
        who = [shot["subject"]]
        solo = shot["subject"]
    named_any = any(_mentions(action, n) for n in (genders or {}))
    # everyone the story names counts for the layout rule, pictures or not (Marco had none, so the café's walk-up
    # shot counted one person and lost its layout)
    in_shot = len({n for n in (genders or {}) if _mentions(action, n)} | set(who))
    if not who and not named_any and _PEOPLE.search(action):
        who = cast
        # only "she" (or only "he"): the people of that sex, from the story's cast list ("Lonecat, a woman ...")
        she = bool(re.search(r"\b(she|her|hers)\b", action, re.I))
        he = bool(re.search(r"\b(he|him|his)\b", action, re.I))
        if genders and she != he:
            want = {"woman", "girl", "lady"} if she else {"man", "boy", "guy"}
            picked = [n for n in cast if (genders.get(n) or "") in want]
            who = picked or cast
    settings = sorted({p["name"] for p in pics if p["role"] in ("setting", "view", "layout") and p["name"]})
    place = _norm_words(shot.get("place_name") or "")
    if place:
        # the Shot List says where this shot is: that place's pictures only, or none when it has none. Matching words in
        # the action sent "outside the bank" to every "inside the bank lobby" shot and missed "the black van".
        where = [n for n in settings if _norm_words(re.sub(r"^(the|a|an)\s+", "", n, flags=re.I))
                 == _norm_words(re.sub(r"^(the|a|an)\s+", "", place, flags=re.I))]
    else:
        # the only setting goes to every shot only when the whole story stays in one place
        where = [n for n in settings if _mentions(text, n)] or (settings if len(settings) == 1 and one_place else [])
    view = str(shot.get("view_name") or "").strip().lower()
    if view:
        # LC H3 Story: the angle picked for this shot in the place's 360 panorama is its only background picture
        where = [n for n in settings if n.strip().lower() == view] or where
    props = [p["name"] for p in pics if p["role"] == "prop" and _mentions(text, p["name"])]
    keep = {n.strip().lower() for n in who + where + props}
    quotes = _SPOKEN.findall(action)
    lines = {a["id"] for a in auds if a["role"] == "line" and a["name"] in who and _says_line(a.get("text", ""), quotes)}
    sub = _refs_subset(refs, keep, lines)
    if "outfits" in shot:
        # LC H3 Story: each person wears one outfit per shot, the one the story is up to (none = their look)
        worn = set(shot.get("outfits") or [])
        sub["pictures"] = [{**q, "n": k + 1} for k, q in enumerate(q for q in sub["pictures"] if q["role"] != "outfit" or q["id"] in worn)]
    # a recorded line is locked into the sound at its second, not sent as <Audio n> as well (H3 plays a fully_copy
    # reference from 0 s, so the line came out twice)
    from .lc_h3_refs import sentences as _ref_sentences

    if solo in cast or max(in_shot, len(who)) <= 1:
        # a layout picture is about where people are: one person's shot never gets it (in the café test H3 also took
        # the layout's camera position, and every CONTINUE after it was stuck on that wide frame)
        sub["pictures"] = [{**q, "n": k + 1} for k, q in enumerate(q for q in sub["pictures"] if q["role"] != "layout")]
    # one picture per place: its layout (who stands where) when several people are in the shot, else its setting
    named = {(q["role"], (q["name"] or "").strip().lower()) for q in sub["pictures"]}
    drop = "setting" if len(who) > 1 else "layout"
    sub["pictures"] = [{**q, "n": k + 1} for k, q in enumerate(
        q for q in sub["pictures"]
        if not (q["role"] == drop and ({"setting", "layout"} - {drop}).pop() in {r for r, n in named if n == (q["name"] or "").strip().lower()}))]
    if shot.get("room") is False:
        # LC H3 Story marks tight framings (medium, close-up): a room picture pins H3 to its camera position, and a
        # close-up came out as a giant head pasted over the wide room (Coffee Date). The Place words still go in.
        sub["pictures"] = [{**q, "n": k + 1} for k, q in enumerate(q for q in sub["pictures"] if q["role"] not in ("setting", "layout"))]
    sub["lock"] = [a for a in sub["audios"] if a["role"] == "line"]
    sub["audios"] = [{**a, "n": k + 1} for k, a in enumerate(a for a in sub["audios"] if a["role"] != "line")]
    sub["text"] = _ref_sentences(sub)
    return sub if sub["pictures"] or sub["audios"] or sub["lock"] else None


_CAM_MOVE = re.compile(r"\b(push(es|ing)?|pull(s|ing)? (back|out|away)|pan(s|ning)?|dolly|dollies|zoom(s|ing)?|track(s|ing)?|"
                       r"moves?|moving|widens?|follows?|crane|tilts?|orbits?|circles?|walks? (with|behind|beside)|passes)\b", re.I)
_NO_MOVE = re.compile(r"\b(with\s+)?(no|without( any)?)\s+(camera\s+)?(move(ment)?s?|moving)\b", re.I)
_CLOTHES = re.compile(r"\b(sweater|jumper|blouse|shirt|t-shirt|tee|top|jacket|coat|parka|hoodie|dress|skirt|suit|blazer|vest|"
                      r"cardigan|jeans|trousers|pants|shorts|uniform|scarf|hat|cap|gown|robe|apron|overalls|tank)\b", re.I)


def _unsay_looks(style: str, story: str, cast: dict) -> str:
    """The Showrunner's look line describes everyone in words ("a man in his early forties with short dark hair ...").
    H3 follows the words far more than the picture, so the words have to describe the picture. When a face or full
    body picture has a note (its look: "early twenties, dark curls, clean-shaven") the note takes that person's place,
    plus the story's clothes for a face picture. No note: the story's words stay. Each description is matched to its
    person through the story ("Daniel, a man in his early forties ...")."""
    import difflib

    if not cast or ";" not in style and not re.match(r"\s*an?\s+(man|woman|boy|girl|guy|lady)\b", style, re.I):
        return style
    head, sep, rest = style.partition(". ")
    named = {m.group(1): story[m.end():m.end() + 200] for m in re.finditer(r"\b([A-Z][a-z]+),\s+(?=an?\s)", story)}
    out = []
    for part in head.split(";"):
        d = part.strip()
        who = max(named, key=lambda n: difflib.SequenceMatcher(None, d.lower(), named[n][:len(d) + 10].lower()).ratio(), default=None)
        r = cast.get(who) if who else None
        pics = (r or {}).get("face", []) + (r or {}).get("full body", [])
        look = next((p["note"].strip() for p in pics if str(p.get("note", "")).strip()), "")
        if not look:
            out.append(d)
            continue
        wear = []
        if not (r.get("full body") or r.get("outfit")):
            wear = [w.strip() for w in re.split(r",|\band\b|\bwith\b|\bwearing\b|\bin\b(?=\s+an?\b)", d) if _CLOTHES.search(w)]
        out.append(f"{who}: {look}" + (", wearing " + " and ".join(wear) if wear else ""))
    if not out:
        return rest
    return "; ".join(out) + sep + rest


def _h3_sections_refs(style: str, action: str, pics: list, memory: str | None = None, people: int = 1, auds: list | None = None, story: str = "") -> str:
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
    from .lc_h3_refs import people as _ref_people, retention as _ref_retention

    tag = lambda p: f"<Picture {p['n']}>"  # noqa: E731
    join = lambda t: t[0] if len(t) == 1 else ", ".join(t[:-1]) + " and " + t[-1]  # noqa: E731
    shift = 1 if memory == "video_audio" else 0  # segment 1's memory clip takes <Audio 1> (it comes before your clips)
    auds = [{**a, "n": a["n"] + shift} for a in (auds or [])]
    bundle = {"pictures": pics, "audios": auds}
    cast = _ref_people(bundle)
    style = _unsay_looks(style, story, cast)
    desc = (style + NL + NL + body).strip()
    people_pics = {n: r for n, r in cast.items() if r.get("face") or r.get("full body")}
    defs, kept, subjects = [], [], []
    subject_of = {}
    for k, (name, r) in enumerate(cast.items(), 1):
        subj = f"<Subject {k}>"
        subjects.append(subj)
        subject_of[name] = subj
        look = [tag(p) for p in r.get("face", []) + r.get("full body", [])]
        voice = [f"<Audio {a['n']}>" for a in r.get("voice", [])]
        defs.append(f"{subj} is {name}" + (f", the person shown in {join(look)}" if look else "")
                    + (f"; their voice is {join(voice)}" if voice else "") + ".")
    kept += _ref_retention(bundle, subject_of)
    if memory and not people_pics:
        subj = "<Subject 1> and <Subject 2>" if people > 1 else "<Subject 1>"
        subjects.append(subj)
        defs.append(f"{subj} {'are the two people' if people > 1 else 'is the person'} shown in <Video 1>.")
        kept.append(f"{subj} (<Video 1>): fully_preserved - {'their faces' if people > 1 else 'the face'}, hair and clothing are retained. "
                    "Only the people come from <Video 1>; the place, the poses and the framing are new.")
        if memory == "video_audio":
            kept.append(f"<Audio 1>: reference - the voice{'s' if people > 1 else ''} of {subj}.")
    follow = join(subjects) if subjects else "the scene"
    kinds = ["reference generation"]
    if auds or (memory == "video_audio" and not people_pics):
        kinds.append("audio reuse" if any(a["role"] == "line" for a in auds) else "audio reference")
    task = "[" + " + ".join(kinds) + "]"
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
    # "from behind Dex, tracking alongside as they sprint" is the group seen past one of them, not a close-up of Dex
    if not cam or re.search(r"\b(both|two|three|four|them|they|couple|together|everyone|group|all)\b", cam, re.I):
        return None
    hits = [n for n in names if re.search(rf"\b{n}\b", cam)]
    return hits[0] if len(hits) == 1 else None


def _cast_path(folder: str, name: str, ext: str) -> str:
    return os.path.join(work_dir(folder), f"cast_{re.sub(r'[^A-Za-z0-9]', '', name)}{ext}")


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
                "plan": (PLAN_TYPE, {"tooltip": "From LC Vision Showrunner 🎬."}),
                "clip": ("CLIP", {"tooltip": "MiniMax H3 text encoder."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 60, "tooltip": "Only used when make is redo: the segment to make again."}),
                "width": ("INT", {"default": 960, "min": 256, "max": 2048, "step": 32, "tooltip": "960 x 544 is a good low VRAM size. Keep it the same for every segment of a project."}),
                "height": ("INT", {"default": 544, "min": 256, "max": 2048, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Passed through to your sampler and saved with the segment. Change it to redo a segment."}),
                "carry_on_line": ("BOOLEAN", {"default": True, "tooltip": "CONTINUE segments: adds a line telling H3 the opening is the previous moment carrying straight on."}),
                "color_match": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Pulls the carried frames' color toward the start of this scene (the last CUT) to fight drift. 0 = off. Turn it on (0.2 to 0.4) when Make's info flags a scene drifting."}),
                "memory": (["clip", "on", "off"], {"default": "clip", "tooltip": "Keeps the cast the same. clip: about 1 s of segment 1 (picture and sound) goes in as a video reference so faces, hair, clothes and voices stay the same, on a CUT written into the story, and as a gentle nudge on any CONTINUE that starts with no clear face (or after a segment whose face drifted). on: a still portrait from segment 1 (or memory_image). off: the style text alone."}),
                "memory_gap": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5, "tooltip": "Seconds of empty time H3 sees between the memory and the start of the cut. Without a gap H3 treats the memory as the moment just before the cut and opens on its pose. 5 s worked in tests. 0 = stock H3."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "Start image for segment 1 (and for CUT segments when wired)."}),
                "memory_image": ("IMAGE", {"tooltip": "Your own cast picture (a character sheet or a still) to use as the memory instead of segment 1's frame. Used for every segment, segment 1 included."}),
                "prompt_format": (["sections (H3 trained)", "plain"], {"default": "sections (H3 trained)", "tooltip": "sections: the layout H3 was trained on (description / soundscape / non_diegetic_music: none, dialogue as <d>[English] ... </d>, and the reference block for the memory). plain: style + action text."}),
                "redo_after": (["just this segment", "this + the CONTINUE ones after it"], {"default": "this + the CONTINUE ones after it", "tooltip": "When index redoes a segment that is already saved. The old take is always kept in the project's takes folder. this + the CONTINUE ones after it: the segments that carry straight on from this one no longer join, so they are set aside too; set index back to 0 and queue to make them again. A CUT after it stays."}),
                "refs": (REFS_TYPE, {"tooltip": "From LC H3 Reference Assembler: your cast, setting, prop and style pictures. Each segment gets only the ones its shot names."}),
                "refs_on": (["cuts only", "cuts + re-anchors", "cuts + places on continues", "every segment"], {"default": "cuts only", "tooltip": "When the references are sent. cuts only: CUT segments only, so every CONTINUE carries straight on from the frames before it (smoothest joins). cuts + re-anchors: CUT segments, and a CONTINUE that lost the face (a reference on a plain CONTINUE pulled the opening away from the carried frames in tests). cuts + places on continues: CUT segments get everything; a CONTINUE gets only the place, layout and prop pictures (who stands where, which way the door opens), never the faces, which carry on from the frames before it. A CONTINUE that isn't a close-up also gets the place's room (your room picture, or the place's first wide segment) so a camera that pulls back finds the same room. every segment: always."}),
                "make": (["all remaining", "next segment", "redo"], {"default": "all remaining", "tooltip": "all remaining: one Run makes every segment not saved yet, one after another, however many there are (only one run ever waits in the queue, so Stop ends it right there; Run again to carry on). next segment: one Run, one segment. redo: makes segment number index again."}),
                "hold_scene": ("BOOLEAN", {"default": True, "tooltip": "Keeps the room the same across cuts: every later shot with more than one person gets a frame of the first wide shot as a layout reference (who sits where, the table, the window). Close-ups of one person don't. A layout picture of your own takes its place."}),
            },
        }

    # prompt last: added after the first workflows were saved (outputs only ever go on the end)
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", SEGMENT_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "seed", "segment", "info", "prompt")
    FUNCTION = "build"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = "One segment of an LC Vision Showrunner plan: conditioning + latent for your sampler. CONTINUE segments start from the previous segment's saved ending."

    @classmethod
    def IS_CHANGED(cls, plan=None, index=0, make="all remaining", **kwargs):
        if not isinstance(plan, dict):
            # plan comes in through a link, and ComfyUI doesn't hand linked inputs to IS_CHANGED: always run, or a
            # second Run with a fixed seed would replay the cached segment instead of making the next one
            return float("nan")
        i = _resolve_index(plan, index if make == "redo" else 0)
        prev = _seg_path(project_dir(plan.get("project", "")), i - 1, "_tail.mp4")
        return f"{i}:{os.path.getmtime(prev) if os.path.exists(prev) else 0}"

    def build(self, plan, clip, vae, audio_vae, index, width, height, seed, carry_on_line, color_match, memory="clip", memory_gap=5.0,
              first_frame=None, memory_image=None, prompt_format="sections (H3 trained)",
              redo_after="this + the CONTINUE ones after it", refs=None, refs_on="cuts only", make="all remaining", hold_scene=True):
        shots = plan.get("shots") or []
        if not shots:
            raise ValueError("[LC H3] The plan has no segments.")
        if make == "redo" and int(index) <= 0:  # index 0 would quietly pick the last (or next) segment
            raise ValueError("[LC H3] make is redo: set index to the segment to redo (1 and up), or use ↻ on the shot's tile in the Story board.")
        i = _resolve_index(plan, index if make == "redo" else 0)
        shot = shots[i - 1]
        folder0 = project_dir(plan["project"])
        if make != "redo" and all(_is_done(folder0, k) for k in range(1, len(shots) + 1)):
            # nothing left to make: stop quietly (the browser shows "all segments are made"), no error and no remake
            from comfy_execution.graph import ExecutionBlocker
            msg = (f"Every segment of {plan['project']} is made. To make one again, use ↻ on the shot's tile in the Story board "
                   "(or make: redo with its index).")
            print(f"[LC H3] {msg}")
            _tell_ui({"project": plan["project"], "left": 0, "made": 0, "next": 0})
            b = ExecutionBlocker(None)
            return {"ui": {"text": [msg]}, "result": (b, b, b, b, msg, b)}
        if make == "redo" and i in _locked(folder0) and _is_done(folder0, i):
            raise ValueError(f"[LC H3] Segment {i} is locked 🔒. Unlock it on the Story board to make it again.")
        if make == "redo" and _is_done(project_dir(plan["project"]), i):
            try:
                with open(_seg_path(project_dir(plan["project"]), i, ".json"), encoding="utf-8") as f:
                    old_seed = int(json.load(f).get("seed", -1))
            except Exception:
                old_seed = -1
            if int(seed) == old_seed:  # a fixed (or wired) seed would make the same take again
                seed = (old_seed * 6364136223846793005 + 1442695040888963407) % (2 ** 63)
                print(f"[LC H3] Redo segment {i}: same seed as the saved take, using {seed} so it comes out different.")
        folder = project_dir(plan["project"])
        os.makedirs(folder, exist_ok=True)
        seg_frames, ovl_frames = _frames(plan)
        seg_frames = int(shot.get("frames") or seg_frames)  # a long line gets its own longer segment (extend_segments)
        cont = shot.get("mode") == "CONTINUE" and i > 1
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
            # the memory carries faces and voices, never its place: H3 copies the room it sees (a lobby shot went
            # back into the van the memory was made in). One person: the head-only clip. Two or more: the whole
            # frame, but only at the place the memory was made; elsewhere none (their looks come from the words),
            # or every cut would get the one face the head crop centred on
            try:
                with open(_seg_path(folder, 1, "_memory.json"), encoding="utf-8") as f:
                    mem_from = int(json.load(f).get("from", 1))
            except Exception:
                mem_from = 1
            groups = _place_groups(plan)
            same = mem_from > len(groups) or groups[i - 1] is None or groups[mem_from - 1] in (None, groups[i - 1])
            if os.path.exists(head) and cast < 2:
                cfr = _read_mp4_native(head)
            elif same:
                cfr = _read_mp4_frames(_seg_path(folder, 1, "_memory_clip.mp4"), width, height)
            else:
                cfr = None
            if cfr is not None:
                n_ok = cfr.shape[0]
                while n_ok > 5 and n_ok % 17 != 5:
                    n_ok -= 1
                mem_clip = (cfr[:n_ok], _read_wav(_seg_path(folder, 1, "_memory_clip.wav")) if os.path.exists(_seg_path(folder, 1, "_memory_clip.wav")) else None)
            names = _cast_names(plan)
            solo = _solo(shot["action"], names)
            named = [n for n in names if re.search(rf"\b{re.escape(n)}\b", shot["action"], re.I)]
            if not solo and (len(names) > 2 or not named and not _PEOPLE_WORDS_RE.search(_action_body(shot["action"]))):
                mem_clip, cast = None, 1
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
        places_only = cont and refs_on == "cuts + places on continues"
        if refs and (not cont or places_only or refs_on == "every segment" or (cont and refs_on == "cuts + re-anchors" and (reanchor if memory == "clip" else _needs_reanchor(folder, i, width, height)))):
            places = {g for g in _place_groups(plan) if g is not None}
            shot_refs = _shot_refs(refs, shot, shot.get("style") or plan.get("style", ""), _cast_names(plan), len(places) <= 1)
            if shot_refs and places_only:
                # staging, not faces: the carried frames already hold the people, the place pictures hold the room
                from .lc_h3_refs import sentences as _ref_sentences
                pics = [p for p in shot_refs["pictures"] if p["role"] in ("setting", "view", "layout", "prop", "look", "composition")]
                shot_refs = {**shot_refs, "pictures": [{**p, "n": k + 1} for k, p in enumerate(pics)], "audios": []}
                shot_refs["text"] = _ref_sentences(shot_refs)
                if not pics and not shot_refs.get("lock"):
                    shot_refs = None
            if shot_refs and not places_only:
                # a person whose face picture goes in and who speaks here: their saved voice too (from their first line),
                # unless the Assembler / Story already gives them one
                from .lc_h3_refs import sentences as _ref_sentences
                pictured = {p["name"] for p in shot_refs["pictures"] if p["role"] in ("face", "full body")}
                voiced = {a["name"] for a in shot_refs["audios"] if a["role"] == "voice"}
                added = False
                for nm in dict.fromkeys(re.findall(r"\b([A-Z][\w'-]+) says\b", shot["action"])):
                    vp = _voice_path(folder, nm) if nm in pictured and nm not in voiced else None
                    if vp and len(shot_refs["audios"]) < 3:
                        shot_refs["audios"].append({"id": f"voice_{nm}", "role": "voice", "name": nm, "note": "", "file": vp,
                                                    "n": len(shot_refs["audios"]) + 1, "audio": _read_wav(vp), "text": ""})
                        added = True
                if added:
                    shot_refs["text"] = _ref_sentences(shot_refs)
        layout_from = None
        # several people in the shot: the anchor keeps who is where (layout). One person, or an outdoor shot where
        # things move: the place and its light only (setting), so nobody from the anchor walks in and the action isn't
        # frozen into the anchor's frame. Close-ups never get it.
        many = _solo(shot.get("action", ""), _cast_names(plan)) is None and len(_cast_names(plan)) > 1 \
            and not _moving_outdoors(shot)
        # a CONTINUE only sees the previous segment's last 1.6 s: after a close-up the room is not in those frames,
        # and a camera that pulls back invents a new one (café test, segment 6). So a CONTINUE that isn't a
        # close-up gets the place's room too: your own room picture for that place when there is one (above),
        # else the place's establishing frame, as a setting (the place and its light, none of its people).
        room_cont = cont and refs_on == "cuts + places on continues" and not _is_close_up(shot.get("action", ""))
        if hold_scene and (not cont or room_cont) and not _is_close_up(shot.get("action", "")) \
                and not any(p["role"] in ("layout", "setting", "view") for p in (shot_refs or {}).get("pictures", [])):
            # ✎ Room from on the Proof Sheet: none, or one segment's frame (a reverse angle needs its own room)
            af = _room_from(folder, i)
            if af == 0:
                anchor = None
            elif af:
                anchor = _anchor_of(folder, af, width, height)
            else:
                anchor = _scene_anchor(plan, folder, i, width, height, empty_first=not many or cont)
            if anchor:
                from .lc_h3_refs import sentences as _ref_sentences
                k, img, mtime = anchor
                shot_refs = shot_refs or {"pictures": [], "audios": [], "lock": [], "text": ""}
                # on a CONTINUE the people are already in the opening frames: the anchor gives the room only
                shot_refs["pictures"].append({"id": "scene", "role": "setting" if cont or not many else "layout", "name": "",
                                              "note": f"segment {k}", "file": "", "n": len(shot_refs["pictures"]) + 1, "image": img})
                shot_refs["text"] = _ref_sentences(shot_refs)
                layout_from = {"index": k, "mtime": mtime}
        if shot_refs and any(p["role"] in ("face", "full body") for p in shot_refs["pictures"]):
            mem_clip, mem = None, None  # your cast pictures take over from segment 1's memory
        latent, frame_count = _empty_av_latent(width, height, seg_frames)

        action = shot["action"]
        # (tested 2026-10-26: SceneBuilder's "everyone does only what is described below ... keeps their position" line
        # didn't help staging and once stalled her at the door with a camera move; left out)
        cam = re.search(r"^Camera:\s*(.*)$", action, re.M)
        # "with no camera move" (the Story's hold line) is not a move: read as one, a hold was told "then the camera
        # moves as described below" and at 1 MP H3 cut to a close-up (Coffee Date shot 6, 2026-10-30)
        cam_moves = bool(cam and _CAM_MOVE.search(_NO_MOVE.sub("", cam.group(1))))
        if cont and carry_on_line:
            # a shot whose own camera line moves the camera must not be told to keep the camera as well: H3 obeyed
            # "same ... camera" and the push-in / walk through the door never happened (café test)
            keep = "the same pose and position; then the camera moves as described below" if cam_moves else "the same pose, position and camera"
            action = (f"From 0.00 to {plan['overlap_s']:.2f} seconds, the shot carries straight on from the previous "
                      f"moment with {keep}." + NL) + action
        style = shot.get("style") or plan.get("style", "")  # LC H3 Story: the shot's own when someone changed outfit
        if cam:  # the shot's own camera line is the only camera instruction (the style line had a second one)
            style = re.sub(r"\s*Camera:[^;\n]*;?", "", style).strip()
        solo_now = _solo(shot["action"], _cast_names(plan))
        if solo_now:
            # the others keep their look when the action names them (else H3 invents whoever "Anika" is)
            names_all = _cast_names(plan)
            style = _style_for(style, solo_now, {n: g for n, g in names_all.items()
                                                 if n == solo_now or not _mentions(_beats_only(shot["action"]), n)})
        else:
            # a CONTINUE starts on the frames before it: whoever the shots since the last cut named is still on
            # screen, so keep their looks ("Vince raises his pistol" had dropped the other two robbers' clothes)
            cast_text = shot["action"]
            if cont:
                for s in sorted((s for s in plan.get("shots", []) if 0 < int(s.get("index", 0) or 0) < i),
                                key=lambda s: -int(s["index"])):
                    cast_text = s.get("action", "") + "\n" + cast_text
                    if str(s.get("mode", "")).upper() != "CONTINUE":
                        break
            style = _style_cast(style, plan, cast_text)
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
                prompt = _h3_sections_refs(style, action, shot_refs["pictures"], ref_kind if mem_clip is not None else None, cast,
                                           shot_refs.get("audios"), plan.get("story", ""))
            else:
                prompt = _h3_sections(style, action, ref_kind, cast, focus if cast > 1 else None)
        elif shot_refs:
            prompt = shot_refs["text"] + NL + NL + prompt
        if cont:
            # the memory lines were written for CUTs ("the place, the pose and the framing are new"): on a CONTINUE that
            # told H3 to build a new room while it carries on from the opening frames
            prompt = re.sub(r"the place, the poses? and the framing are new",
                            "the place carries straight on from the opening frames; the pose and the framing follow the text", prompt)
            prompt = prompt.replace("The place, the pose and the framing come from the text",
                                    "The place carries straight on from the opening frames; the pose and the framing come from the text")
        for a in (shot_refs or {}).get("lock", []):  # a locked line is already in the sound: quoted, not voiced again
            prompt = re.sub(r"<d>\[English\]\s*(.*?)\s*</d>",
                            lambda m: f'"{m.group(1)}"' if _says_line(a.get("text", ""), [m.group(1)]) else m.group(0), prompt)

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
        for a in (shot_refs or {}).get("audios", []):  # your voices and lines: standalone audio, after pictures and videos
            ref_items.append({"type": "audio"})
            a_lat, a_t = _encode_ref_audio(audio_vae, a["audio"])
            refs_out.append({"kind": "audio", "ref_audio_t": a_t, "audio_latent": a_lat, "lc_time_gap": memory_gap * TIME_PER_SECOND})
        locked = []
        for a in (shot_refs or {}).get("lock", []):  # a recorded line the shot says: lock it in at its second
            at = _line_time(action, a.get("text", ""))
            latent = _lock_line(latent, audio_vae, a["audio"], at)
            locked.append(f"{a['name']} at {at:g} s")
        if cont:
            tail_mp4 = _seg_path(folder, i - 1, "_tail.mp4")
            tail_wav = _seg_path(folder, i - 1, "_tail.wav")
            if not os.path.exists(tail_mp4):
                raise ValueError(f"[LC H3] Segment {i} continues segment {i - 1}, which isn't saved yet. Make segment {i - 1} first.")
            lat_path = _seg_path(folder, i - 1, "_latent.safetensors")
            kf = _latent_tail(lat_path, ovl_frames, width, height)
            if kf is None:
                tail = _read_mp4_frames(tail_mp4, width, height)[:ovl_frames]
                ref_path = _seg_path(folder, _scene_start(plan, i - 1, folder), "_tail.mp4")
                ref = _read_mp4_frames(ref_path, width, height) if color_match > 0 and os.path.exists(ref_path) else None
                tail = _color_match(tail, ref, color_match)
                kf = {"resolved_frame_index": 0, "latent": vae.encode(tail)}
                if os.path.exists(tail_wav):
                    a_lat, _ = _encode_ref_audio(audio_vae, _read_wav(tail_wav))
                    kf["audio_latent"] = a_lat
            keyframes.append(kf)
        elif shot.get("start_frame") and os.path.exists(_abs_input(shot["start_frame"])):
            # LC H3 Story: the shot's own start frame (e.g. the person in front of the shot's 360 view). H3 treats a
            # background picture as a hint only; a pinned first frame is followed exactly.
            from PIL import Image
            pic = torch.from_numpy(np.asarray(Image.open(_abs_input(shot["start_frame"])).convert("RGB"), dtype=np.float32) / 255.0)[None]
            img = _resize(pic, width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "latent": vae.encode(img)})
        elif shot.get("return_to") and _is_done(folder, int(shot["return_to"])):
            # LC H3 Story: a cut back to a setup already shown opens on that shot's first frame, so the camera, the room
            # and who sits where come back exactly
            img = _read_mp4_frames(_seg_path(folder, int(shot["return_to"]), ".mp4"), width, height)[:1]
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "latent": vae.encode(img)})
        elif first_frame is not None:
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "latent": vae.encode(img)})
        end_frame = shot.get("end_frame")  # storyboard end frame: the segment has to land on this picture
        if end_frame:
            from PIL import Image
            p = _abs_input(end_frame)
            if not p or not os.path.isfile(p):
                raise ValueError(f"[LC H3] Segment {i}'s end frame {end_frame} isn't there (or is outside ComfyUI's input / output folders).")
            pic = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)[None]
            keyframes.append({"resolved_frame_index": frame_count - 1, "latent": vae.encode(_resize(pic, width, height, "center"))})

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
                   "refs": [f"<Picture {p['n']}> {p['role']} {p['name']}".strip() for p in (shot_refs or {}).get("pictures", [])]
                           + [f"<Audio {x['n']}> {x['role']} {x['name']}".strip() for x in (shot_refs or {}).get("audios", [])],
                   "locked_lines": locked, "layout_from": layout_from,
                   "action": shot.get("action", ""), "start_frame": shot.get("start_frame") or None,  # the Story board's "changed since made"
                   "from_tail_mtime": os.path.getmtime(_seg_path(folder, i - 1, "_tail.mp4")) if cont else None}
        segment["_plan"], segment["_refs"] = plan, refs  # for Segment Save's join and the Proof Sheet, not saved
        segment["_face_pics"] = {}  # every person's face pictures, for Segment Save's face check
        for pic in (refs or {}).get("pictures", []):
            if pic.get("role") in ("face", "full body") and pic.get("name"):
                segment["_face_pics"].setdefault(pic["name"], []).append(pic["image"])
        info = (f"Segment {i} of {len(shots)} | {segment['mode']} | {frame_count} frames | seed {seed}"
                + (f" | starts from the end of segment {i - 1}" if cont else "") + (" | memory clip" if mem_clip is not None else " | memory on" if mem is not None else "")
                + (f" (face re-anchored: {reanchor})" if mem_clip is not None and reanchor else "")
                + (" | refs: " + ", ".join([f"<Picture {p['n']}> {p['name'] or p['role']}" for p in shot_refs["pictures"]]
                                          + [f"<Audio {x['n']}> {x['name']} ({x['role']})" for x in shot_refs.get("audios", [])]) if shot_refs else "")
                + (" | line locked: " + ", ".join(locked) if locked else ""))
        n_pics = len((shot_refs or {}).get("pictures", []))
        if n_pics > SHOT_PICTURES_OK:
            info += (f" | ⚠️ {n_pics} pictures in one shot: heavy on VRAM, and H3 starts mixing them up. Fewer, named "
                     f"pictures hold better (one face per person, the place).")
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
        with open(os.path.join(segment["folder"], "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
    except Exception:
        return None
    return _solo(segment.get("prompt", ""), _cast_names(plan))


_SAID_AT = re.compile(r"At (\d+(?:\.\d+)?) seconds, ([A-Z][\w'-]+)(?: \(S\d+\))? says:?\s*(?:<d>\[\w+\]\s*(.+?)\s*</d>|[\"“](.+?)[\"”])", re.S)


def _voice_path(folder: str, name: str) -> str | None:
    """A person's saved voice: from their first line, else from their first solo close-up."""
    for p in (_cast_path(folder, name, "_voice.wav"), _cast_path(folder, name, ".wav")):
        if os.path.exists(p):
            return p
    return None


def _save_voice_memory(segment: dict, wave, sr: int):
    """The first line each person speaks ("At 1.20 seconds, Nova says: ..."): about 3 s of that stretch of sound, kept
    as their voice. A later shot that sends their face picture sends this as their voice too; without it H3 made up a
    new voice on every cut (Coffee Date: Nova's voice drifted in her close-up)."""
    if wave is None or not sr:
        return
    folder = segment["folder"]
    for m in _SAID_AT.finditer(segment.get("prompt", "")):
        name, words = m.group(2), (m.group(3) or m.group(4) or "")
        if os.path.exists(_cast_path(folder, name, "_voice.wav")):
            continue
        t0 = float(m.group(1))
        t1 = t0 + max(2.0, len(words.split()) / 2.5 + 0.6)
        a, b = int(t0 * sr), min(wave.shape[-1], int(t1 * sr))
        if b - a < int(1.5 * sr):  # too little of the line made it into the segment
            continue
        _write_wav(_cast_path(folder, name, "_voice.wav"), wave[:, a:b], sr)
        print(f"[LC H3] Saved {name}'s voice from their line in segment {segment['index']}.")


def _save_cast_memory(segment: dict, frames_u8: np.ndarray, wave, sr: int):
    """A shot of one named person ("Camera: close-up of Daniel"): keep about 3 s of their head and their voice, the
    loudest stretch (their line), as that person's memory. Later shots of them get it, so the face and voice match
    and nobody else is pulled into their close-up. Only the first solo shot of each person is kept."""
    folder = segment["folder"]
    try:
        with open(os.path.join(folder, "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
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


def _locked(folder: str) -> set:
    """Segments locked 🔒 on the Story board (read from plan.json, so a lock set a moment ago counts)."""
    try:
        with open(os.path.join(folder, "plan.json"), encoding="utf-8") as f:
            return {k for k, sh in enumerate(json.load(f).get("shots", []), start=1) if sh.get("locked")}
    except Exception:
        return set()


def _history(folder: str, entry: dict) -> None:
    """One step for ↶ Undo on the Proof Sheet (takes/history.json, newest last)."""
    path = os.path.join(folder, "takes", "history.json")
    try:
        with open(path, encoding="utf-8") as f:
            hist = json.load(f)
    except Exception:
        hist = []
    hist.append({**entry, "time": time.time()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _save_json(path, hist[-50:], indent=1)


SHOT_PICTURES_OK = 4  # more than this in one shot: the Proof Sheet tags it


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
    take = _keep_take(folder, i)
    note = f" | redo: old take kept as takes/{take}"
    aside = []
    if os.path.exists(_seg_path(folder, i, "_up.mp4")):  # the old upscale is of the old take: make it again
        try:
            os.remove(_seg_path(folder, i, "_up.mp4"))
        except OSError as e:  # open in a player: keep going, the new take's json has no "up", so it counts as not upscaled
            print(f"[LC H3] Could not remove segment {i}'s old upscale: {e}")
        note += " (its upscale too: run LC H3 Upscale for it again)"
    if segment.get("redo_after"):
        nxt = []
        k = i + 1
        locked = _locked(folder)
        while k <= segment["count"] and _is_done(folder, k):
            with open(_seg_path(folder, k, ".json"), encoding="utf-8") as f:
                if json.load(f).get("mode") != "CONTINUE":
                    break
            if k in locked:  # a locked segment stays (it shows yellow: made from the old ending)
                note += f" | segment {k} is locked, so it was kept"
                break
            aside.append([k, _keep_take(folder, k, move=True)])
            nxt.append(k)
            k += 1
        if nxt:
            note += (f" | segment{'s' if len(nxt) > 1 else ''} {', '.join(map(str, nxt))} set aside (they continue from this one):"
                     f" set index to 0 and queue {len(nxt)} more")
    _history(folder, {"kind": "redo", "index": i, "take": take, "aside": aside})
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
    # the cast memory (kept under segment 1's name) comes from the first segment that shows a face: a story can open
    # on a van pulling up, and a memory of the van would be sent as "the people" to every later cut
    mem_files = ("_memory.png", "_face_ref.png", "_face_refs.npy", "_memory_clip.mp4", "_memory_head.mp4", "_memory_clip.wav",
                 "_memory.json")
    bi = _best_face_frame(frames_u8) if i == 1 or not os.path.exists(_seg_path(folder, 1, "_memory_clip.mp4")) else None
    if i == 1 and bi is None:
        for ext in mem_files:  # an older segment 1 left its memory behind
            if os.path.exists(_seg_path(folder, 1, ext)):
                os.remove(_seg_path(folder, 1, ext))
    if bi is not None:
        from PIL import Image
        best = frames_u8[bi]
        Image.fromarray(np.ascontiguousarray(_portrait_crop(best))).save(_seg_path(folder, 1, "_memory.png"))
        Image.fromarray(best).save(_seg_path(folder, 1, "_face_ref.png"))
        refs = _face_refs(frames_u8)
        if refs is not None:
            np.save(_seg_path(folder, 1, "_face_refs.npy"), refs)
        c0 = min(max(0, bi - 11), max(0, n - 22))  # the memory clip is centred on the best face too
        _encode(frames_u8[c0:c0 + 22], _seg_path(folder, 1, "_memory_clip.mp4"), 8, "yuv444p")
        head = _head_crop(frames_u8[c0:c0 + 22])
        if head is not None:
            _encode(head, _seg_path(folder, 1, "_memory_head.mp4"), 8, "yuv444p")
        if wave is not None:
            _write_wav(_seg_path(folder, 1, "_memory_clip.wav"), wave[:, int(c0 * per_frame):int((c0 + 22) * per_frame)], sr)
        with open(_seg_path(folder, 1, "_memory.json"), "w", encoding="utf-8") as f:
            json.dump({"from": i}, f)
    _save_cast_memory(segment, frames_u8, wave, sr)
    _save_voice_memory(segment, wave, sr)
    solo = _solo_name(segment) if i > 1 else None
    # your face pictures (Reference Assembler) are the truth: a close-up is checked against that person's picture,
    # any other shot against the whole cast. Without pictures: their first close-up, else segment 1.
    pics = segment.get("_face_pics") or {}
    mine = pics.get(solo) if solo else [im for ims in pics.values() for im in ims]
    pic_refs = _pic_face_refs(mine) if mine else None
    own = None if pic_refs is not None else (_cast_refs(folder, solo) if solo else None)
    face = _face_score(frames_u8, pic_refs if pic_refs is not None else (own or _seg_path(folder, 1, "_face_refs.npy")))
    if face is not None and (own or (pic_refs is not None and solo)):
        face["who"] = solo  # matched against this person's own face, not segment 1's
    if face is not None and pic_refs is not None:
        face["against"] = "picture"
    meta = {**{k: v for k, v in segment.items() if not k.startswith("_")}, "decoded_frames": n, "part_frames": n - start, "look": _look(frames_u8[n - ovl:]),
            "face": face, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_json(_seg_path(folder, i, ".json"), meta, indent=1, ensure_ascii=False)

    info = f"Saved segment {i} of {segment['count']} ({n - start} new frames) to {folder}"
    if face and i > 1 and face.get("match") is None:
        info += (". No clear face of the main person to check (the clear face was someone else's)" if face.get("others", 0) >= MIN_FACE_FRAMES
                 else ". No clear face to check (seen from behind or too small)")
    elif face and i > 1:
        verdict = "same person" if face["match"] >= FACE_SAME else "close" if face["match"] >= FACE_CLOSE else "⚠️ looks like someone else, try another seed"
        info += f". Face match{' to ' + face['who'] if face.get('who') else ''} {face['match']:.2f} ({verdict})"
    left = [k for k in range(1, int(segment["count"]) + 1) if not _is_done(folder, k)]
    more = [k for k in segment.get("_more") or [] if k != i and _is_done(folder, k)]  # LC H3 Make: edited shots still to remake
    if not left and more:
        _tell_ui({"project": segment["project"], "made": i, "left": len(more), "next": more[0], "changed": True})
    else:
        _tell_ui({"project": segment["project"], "made": i, "left": len(left), "next": left[0] if left else 0})
    if left:  # a redo can leave earlier segments (or none after it) still to make
        info += f". Next: segment {left[0]}."
    else:
        info += ". All segments done: the full video is joined."
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
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "info", "video")
    OUTPUT_TOOLTIPS = ("This segment's sound, for LC H3 Dialogue Check.",
                       "What was saved, the face match and how smooth every join is.", "The full video so far: every saved segment joined.")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("Decodes and saves one segment (a preview mp4, the part for the final video and the ending the next segment "
                   "starts from), then joins every saved segment into <project>_full.mp4 and reports how smooth each join is.")

    def save(self, segment, samples, vae, audio_vae, crf, prompt=None, extra_pnginfo=None):
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
        video, seams, failed = "", "", ""
        plan = segment.get("_plan")
        if plan:
            try:
                joined = LCH3Assemble().assemble(plan, 8, 15, crf, prompt=prompt, extra_pnginfo=extra_pnginfo)["result"]
                video, seams = joined[0], joined[1]
            except ValueError as exc:  # segment 1 is redone and not saved yet, etc.
                seams = str(exc)
            except comfy.model_management.InterruptProcessingException:
                raise
            except Exception as exc:  # ffmpeg failed: the segment is saved, so the run (and the chain) carries on, loudly
                failed = f"NOT JOINED: the full movie was not updated ({exc}). The segment itself is saved."
                seams = "⚠️ " + failed
                print(f"[LC H3] WARNING {failed}")
        if video and os.path.exists(video) and plan and all(_is_done(segment["folder"], k) for k in range(1, len(plan.get("shots", [])) + 1)):
            final = os.path.join(segment["folder"], f"{os.path.basename(segment['folder'])}.mp4")  # top of the project
            try:
                shutil.copy2(video, final)
                seams = f"🎬 Finished movie: {final}" + (NL + NL + seams if seams else "")
            except OSError as exc:  # e.g. the old movie is open in a player
                failed = f"NOT COPIED: the finished movie {final} was not updated (is it open in a player?): {exc}"
                seams = "⚠️ " + failed + (NL + NL + seams if seams else "")
                print(f"[LC H3] WARNING {failed}")
        if failed:
            _warn_ui(segment.get("project", ""), failed)
            text = "⚠️ " + failed + NL + NL + info + (NL + NL + seams if seams else "")  # on top, so it is seen
        else:
            text = info + (NL + NL + seams if seams else "")
        return {"ui": {"text": [text]}, "result": (audio, text, video)}


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
                "plan": (PLAN_TYPE, {"tooltip": "From LC Vision Showrunner 🎬 (a saved plan loads instantly)."}),
                "blend_frames": ("INT", {"default": 8, "min": 0, "max": 30, "tooltip": "CONTINUE joins: crossfades this many frames of the old ending into the remade start (picture and sound), so the join doesn't pop. 0 = hard join."}),
                "seam_fade_ms": ("INT", {"default": 15, "min": 0, "max": 200, "tooltip": "CUT joins: a tiny audio fade so the cut doesn't click."}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 40}),
            },
            "optional": {
                "after": ("STRING", {"forceInput": True, "tooltip": "Wire LC H3 Segment Save's info here, so the join runs after the new segment is saved (in the same workflow)."}),
                "version": (["normal", "upscaled"], {"default": "normal", "tooltip": "upscaled: joins the LC H3 Upscale copies instead (every segment needs one) into <project>_full_upscaled.mp4."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
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

    def assemble(self, plan, blend_frames, seam_fade_ms, crf, after=None, version="normal", prompt=None, extra_pnginfo=None):
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
        dest = os.path.join(work_dir(folder), f"{os.path.basename(folder)}_full{'_upscaled' if up else ''}.mp4")
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
                comfy.model_management.throw_exception_if_processing_interrupted()
                i = seg["index"]
                full = _seg_path(folder, i, "_up.mp4" if up else ".mp4")
                frames = (_read_mp4_frames(full, w, h) * 255.0).round().to(torch.uint8).numpy()
                aud = _read_audio(full, sr)
                nxt = segs[k + 1] if k + 1 < len(segs) else None
                n, ovl = len(frames), int(seg.get("overlap_frames", 0))
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
                    start = int(seg.get("trim_start", 0))  # it carried on from a shot that was cut since
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
            # the prompt + workflow go on the joined movie as tags: CivitAI reads those from an uploaded video
            meta_in, meta_map = metadata_args(comfy_tags(prompt, extra_pnginfo), tmp, 2)
            subprocess.run([ff, "-y", "-v", "error", "-i", video, "-i", wav] + meta_in + ["-map", "0:v", "-map", "1:a"] + meta_map +
                           ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart+use_metadata_tags", dest],
                           check=True, capture_output=True)
        finally:
            if enc.poll() is None:
                enc.kill()
                enc.wait()  # let go of v.mp4, or the temp folder can't be removed on Windows
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
        _save_json(os.path.join(work_dir(folder), "seams.json"), {"joins": report, "drift": drift, "faces": faces}, indent=1)
        seam_text = ("Join smoothness (1.00x = as smooth as the rest of the video, over 1.60x is flagged):\n" + "\n".join(lines)) if lines else "No joins yet."
        if stale:
            seam_text += NL + NL + "⚠️ " + ", ".join(f"segment {k}" for k in stale) + (
                " continued from an older version of the segment before it (that one was redone). Redo "
                + ("it" if len(stale) == 1 else "them") + " so the join lines up.")
        if drift_lines:
            seam_text += ("\n\nColor drift inside each scene (under 3 = invisible, over 6 is flagged; a scene that walks "
                          "into new light on purpose will read high):\n" + "\n".join(drift_lines))
        if face_lines:
            seam_text += (NL + NL + f"Face match to your cast card pictures, or without pictures to segment 1 / that person's first close-up (0 to 1; {FACE_SAME:.2f} and up = same person, "
                          f"{FACE_CLOSE:.2f} to {FACE_SAME:.2f} = close, under {FACE_CLOSE:.2f} = someone else):" + NL + NL.join(face_lines))
        written = state["written"]
        info = f"Joined {len(segs)} segment(s) into {dest} ({written} frames, {written / FPS:.2f} s)"
        print(f"[LC H3] {info}\n{seam_text}")
        return {"ui": {"text": [info + "\n\n" + seam_text]}, "result": (dest, seam_text)}


# Segment, Segment Save and Assemble are no longer nodes: LC H3 Make runs them (2026-11-01).
NODE_CLASS_MAPPINGS = {}

NODE_DISPLAY_NAME_MAPPINGS = {}
