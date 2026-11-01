"""
LC Load Video / LC Preview Video
--------------------------------
LC Load Video: a video file in, its frames, sound and frame rate out, exactly as the file has them (no resizing, frame
skipping or trimming). Frames stream in through ffmpeg with a progress bar, and Cancel stops it.
LC Preview Video: plays a video file, or a batch of frames (with sound), on the node.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid

import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths

from .lc_save_video import _encode_ffmpeg, _ffmpeg_bin, _frame_order


def _resolve(path):
    p = str(path or "").strip().strip('"')
    if not p:
        return ""
    if os.path.isfile(p):
        return p
    for base in (folder_paths.get_output_directory(), folder_paths.get_input_directory(), folder_paths.get_temp_directory()):
        q = os.path.join(base, p)
        if os.path.isfile(q):
            return q
    return p


def probe(path):
    """{width, height, fps, duration, frames, sample_rate, channels} from ffmpeg's own read of the file."""
    ff = _ffmpeg_bin()
    if not ff:
        raise RuntimeError("[LC Load Video] ffmpeg not found (install it or the imageio-ffmpeg package).")
    err = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace").stderr
    info = {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0, "frames": 0, "sample_rate": 0, "channels": 0}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    v = re.search(r"Stream #\S+.*?: Video: .*", err)
    if v:
        line = v.group(0)
        s = re.search(r", (\d{2,5})x(\d{2,5})", line)
        if s:
            info["width"], info["height"] = int(s.group(1)), int(s.group(2))
        f = re.search(r"([\d.]+) fps", line) or re.search(r"([\d.]+) tbr", line)
        if f:
            info["fps"] = float(f.group(1))
    rot = re.search(r"rotation of (-?[\d.]+) degrees", err, re.IGNORECASE)
    if rot and round(abs(float(rot.group(1)))) % 180 == 90:
        # phone video: ffmpeg turns the frames upright, so they come out with width and height swapped
        info["width"], info["height"] = info["height"], info["width"]
    a = re.search(r"Stream #\S+.*?: Audio: .*", err)
    if a:
        line = a.group(0)
        r = re.search(r"(\d+) Hz", line)
        info["sample_rate"] = int(r.group(1)) if r else 48000
        info["channels"] = 1 if "mono" in line else 2
    if not info["width"]:
        raise ValueError(f"[LC Load Video] No video in {path}.")
    if info["fps"] <= 0:
        info["fps"] = 24.0
    info["frames"] = max(1, int(round(info["duration"] * info["fps"])))
    return info


def read_frames(path, info):
    """Frames as IMAGE [N, H, W, 3] float 0..1, streamed one at a time (progress bar, Cancel between frames)."""
    w, h = info["width"], info["height"]
    size = w * h * 3
    proc = subprocess.Popen([_ffmpeg_bin(), "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size)
    pbar = comfy.utils.ProgressBar(info["frames"])
    frames = []
    try:
        while True:
            comfy.model_management.throw_exception_if_processing_interrupted()
            buf = proc.stdout.read(size)
            if not buf or len(buf) < size:
                break
            frames.append(torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(h, w, 3))
            pbar.update(1)
    except comfy.model_management.InterruptProcessingException:
        proc.kill()
        proc.wait()
        raise
    finally:
        proc.stdout.close()  # always: after a normal finish too, or the pipe handle stays open
        if proc.poll() is None:
            proc.wait()
    if not frames:
        raise ValueError(f"[LC Load Video] Could not read any frames from {path}.")
    out = torch.empty((len(frames), h, w, 3), dtype=torch.float32)
    for i, f in enumerate(frames):  # uint8 -> float one frame at a time (no second full-size copy)
        out[i] = f.float().div_(255.0)
        frames[i] = None
    return out


def read_audio(path, info):
    """AUDIO {waveform [1, C, S], sample_rate} or None when the file has no sound."""
    if not info["sample_rate"]:
        return None
    sr, ch = info["sample_rate"], info["channels"] or 2
    raw = subprocess.run([_ffmpeg_bin(), "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "pipe:1"],
                         capture_output=True).stdout
    if not raw:
        return None
    wave = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).view(-1, ch).t().unsqueeze(0)
    return {"waveform": wave, "sample_rate": sr}


def _thumb(frame):
    from PIL import Image

    tmp = folder_paths.get_temp_directory()
    os.makedirs(tmp, exist_ok=True)
    name = f"lc_load_video_{uuid.uuid4().hex[:8]}.png"
    Image.fromarray((frame.clamp(0, 1) * 255).byte().numpy()).save(os.path.join(tmp, name))
    return {"filename": name, "subfolder": "", "type": "temp"}


class LCLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "tooltip": "A video file: a full path, or a path inside ComfyUI's output or input folder. "
                                                              "Wire LC H3 Make's video here."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count")
    OUTPUT_TOOLTIPS = ("Every frame, at the file's own size.", "The sound (silence as long as the video when the file has none).",
                       "The file's frame rate.", "How many frames were read.")
    FUNCTION = "load"
    CATEGORY = "LC AV/io"
    DESCRIPTION = ("Loads a video file exactly as it is: every frame at its own size, its sound and its frame rate. No resizing, "
                   "frame skipping or trimming. Streams through ffmpeg with a progress bar, and Cancel works.")

    @classmethod
    def IS_CHANGED(cls, path):
        p = _resolve(path)
        try:
            st = os.stat(p)
            return f"{p}|{st.st_size}|{st.st_mtime}"
        except OSError:
            return p

    def load(self, path):
        if not (path or "").strip():
            raise ValueError("[LC Load Video] No video path came in. If it is wired to LC H3 Make, turn Make on: with "
                             "every shot made it just hands over the finished movie, nothing renders again.")
        p = _resolve(path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[LC Load Video] File not found: {path}")
        info = probe(p)
        images = read_frames(p, info)
        audio = read_audio(p, info)
        sound = audio is not None
        if not sound:  # no sound: silence as long as the video, so audio nodes after it get a real AUDIO, not None
            n = max(1, int(round(images.shape[0] / float(info["fps"]) * 48000)))
            audio = {"waveform": torch.zeros(1, 2, n), "sample_rate": 48000}
        print(f"[LC Load Video] {os.path.basename(p)}: {images.shape[0]} frames, {info['width']} x {info['height']}, "
              f"{info['fps']:g} fps, {'sound' if sound else 'no sound (silent track)'}")
        return {"ui": {"images": [_thumb(images[0])]}, "result": (images, audio, float(info["fps"]), int(images.shape[0]))}


def _view_info(path):
    """/view address of a file in output / input / temp; anything else is copied into temp first."""
    full = os.path.abspath(path)
    for kind, base in (("output", folder_paths.get_output_directory()), ("input", folder_paths.get_input_directory()),
                       ("temp", folder_paths.get_temp_directory())):
        base = os.path.abspath(base)
        if full.lower().startswith(base.lower() + os.sep):
            rel = os.path.relpath(full, base)
            return {"filename": os.path.basename(rel), "subfolder": os.path.dirname(rel), "type": kind}
    tmp = folder_paths.get_temp_directory()
    os.makedirs(tmp, exist_ok=True)
    name = f"lc_preview_{uuid.uuid4().hex[:8]}{os.path.splitext(full)[1]}"
    shutil.copyfile(full, os.path.join(tmp, name))
    return {"filename": name, "subfolder": "", "type": "temp"}


class LCPreviewVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                                  "tooltip": "Frame rate for a batch of frames. A file plays at its own rate."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "A batch of frames to play."}),
                "audio": ("AUDIO", {"tooltip": "Sound to play with the frames."}),
                "path": ("STRING", {"forceInput": True, "tooltip": "Or a video file to play (LC H3 Make's video, for example)."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "LC AV/io"
    DESCRIPTION = ("Plays a video on the node: a file, or a batch of frames with its sound. Nothing is saved to your output "
                   "folder (frames are encoded to a temp file).")

    def preview(self, fps, images=None, audio=None, path=None):
        if path and str(path).strip():
            p = _resolve(path)
            if not os.path.isfile(p):
                print(f"[LC Preview Video] Nothing to play yet ({path} isn't there).")
                return {"ui": {}}
            info = _view_info(p)
        elif images is not None and len(images):
            tmp = folder_paths.get_temp_directory()
            os.makedirs(tmp, exist_ok=True)
            name = f"lc_preview_{uuid.uuid4().hex[:8]}.mp4"
            src = images if images.dim() == 4 else images[None]
            _encode_ffmpeg(src, _frame_order(len(src), False, 0), os.path.join(tmp, name), float(fps), "video/h264-mp4", 20,
                           "yuv420p", audio=audio)
            info = {"filename": name, "subfolder": "", "type": "temp"}
        else:
            return {"ui": {}}  # nothing came in (an empty path, a bypassed branch): stay quiet, don't stop the run
        info["format"] = "video/h264-mp4"
        return {"ui": {"video": [info], "gifs": [info]}}


NODE_CLASS_MAPPINGS = {"LCLoadVideo": LCLoadVideo, "LCPreviewVideo": LCPreviewVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"LCLoadVideo": "LC Load Video 🎞️", "LCPreviewVideo": "LC Preview Video 📺"}
