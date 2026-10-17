"""
LC H3 Proof Sheet: the storyboard of an H3 long-video project. One frame from every segment, with the shot number,
CUT / CONTINUE, where it starts in the final video and its Camera line underneath, as a batch for LC Image Grid.
Segments that are not made yet show as an empty card with their Camera line, so the sheet fills in as you render.
With refs wired (LC H3 Reference Assembler) every card also shows its line of dialogue and the reference pictures that
segment sends to H3, picked exactly the way LC H3 Segment picks them, so the board can be checked before any render.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .lc_h3_refs import REFS_TYPE
from .lc_h3_segment import FPS, PLAN_TYPE, _SPOKEN, _cast_names, _is_done, _seg_path, _shot_refs, project_dir
from .lc_save_video import _ffmpeg_bin

AMBER = (255, 200, 80)
GRAY = (210, 210, 210)
BG = (24, 24, 24)
EMPTY = (38, 38, 38)


def _font(size: int, bold: bool = False):
    for name in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _frame(path: str, t: float, w: int, h: int) -> Image.Image | None:
    """The frame of an mp4 at t seconds, scaled to w x h."""
    if not os.path.exists(path):
        return None
    ff = _ffmpeg_bin()
    for seek in ([f"{max(0.0, t):.3f}"], []):  # a seek past the end falls back to the first frame
        raw = subprocess.run([ff, "-v", "error", *(["-ss", seek[0]] if seek else []), "-i", path, "-frames:v", "1",
                              "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        if len(raw) >= w * h * 3:
            return Image.fromarray(np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3))
    return None


def _fit(draw, text: str, font, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text.rstrip(" ,") + "…"


def _camera(shot: dict) -> str:
    for ln in str(shot.get("action", "")).split("\n"):
        if ln.lower().startswith("camera:"):
            return ln.split(":", 1)[1].strip()
    return ""


class LCH3ProofSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {"tooltip": "From LC Vision Director 🎬."}),
                "frame": (["middle", "start", "end"], {"default": "middle", "tooltip": "Which moment of each segment goes on the sheet. middle shows what the shot is about; start and end are handy for checking the joins."}),
                "cell_width": ("INT", {"default": 480, "min": 160, "max": 1280, "step": 16, "tooltip": "Width of each picture on the sheet. The height follows the video's shape."}),
                "captions": ("BOOLEAN", {"default": True, "tooltip": "Shot number, CUT / CONTINUE, start time and the Camera line under each picture."}),
            },
            "optional": {
                "after": ("STRING", {"forceInput": True, "tooltip": "Wire LC H3 Segment Save's info here, so the sheet updates right after each new segment is saved."}),
                "refs": (REFS_TYPE, {"tooltip": "From LC H3 Reference Assembler: each card shows the reference pictures its segment sends, and its line of dialogue. Works before anything is rendered."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "made")
    OUTPUT_TOOLTIPS = ("One card per segment, for LC Image Grid (4 columns looks like a storyboard).", "How many segments are made so far.")
    FUNCTION = "sheet"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("The project's storyboard: a frame from every segment with its shot number, CUT / CONTINUE, start time and "
                   "Camera line. Segments not made yet show as empty cards. With refs wired, each card also shows its dialogue "
                   "and the reference pictures it sends. Wire it into LC Image Grid.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def sheet(self, plan, frame, cell_width, captions, after=None, refs=None):
        folder = project_dir(plan["project"])
        shots = plan.get("shots") or []
        if not shots:
            raise ValueError("[LC H3] Proof Sheet: the plan has no segments.")
        w = int(cell_width) // 2 * 2
        h = None
        for i in range(1, len(shots) + 1):  # the video's shape, from the first saved segment
            if _is_done(folder, i):
                with open(_seg_path(folder, i, ".json"), encoding="utf-8") as f:
                    m = json.load(f)
                h = int(round(w * m["height"] / m["width"] / 2)) * 2
                break
        h = h or int(round(w * 9 / 16 / 2)) * 2
        cap = int(round(w * 0.1)) if captions else 0
        thumb = int(round(h * 0.3)) if refs and captions else 0  # the reference strip under the Camera line
        if thumb:
            cap += thumb + int(round(w * 0.075))
        genders = _cast_names(plan)
        f1, f2 = _font(max(10, int(w * 0.037)), True), _font(max(9, int(w * 0.029)))
        at = {"start": 0.0, "middle": 0.5, "end": 1.0}[frame]
        cards, made = [], 0
        for i, s in enumerate(shots, start=1):
            card = Image.new("RGB", (w, h + cap), BG)
            pic = None
            if _is_done(folder, i):
                with open(_seg_path(folder, i, ".json"), encoding="utf-8") as f:
                    n = int(json.load(f).get("part_frames", 85))
                pic = _frame(_seg_path(folder, i, "_part.mp4"), (n - 1) * at / FPS, w, h)
            d = ImageDraw.Draw(card)
            if pic is not None:
                card.paste(pic, (0, 0))
                made += 1
            else:
                d.rectangle([0, 0, w - 1, h - 1], fill=EMPTY)
                d.text((w // 2, h // 2), "not made yet", fill=(120, 120, 120), font=f2, anchor="mm")
            if captions:
                pad = max(4, w // 80)
                d.text((pad, h + pad // 2), f"{i} · {s.get('mode', '')} · {float(s.get('start_s', 0)):.1f} s", fill=AMBER, font=f1)
                y = h + pad // 2 + int(f1.size * 1.25)
                d.text((pad, y), _fit(d, _camera(s), f2, w - 2 * pad), fill=GRAY, font=f2)
                if thumb:
                    y += int(f2.size * 1.35)
                    said = _SPOKEN.search(str(s.get("action", "")))
                    d.text((pad, y), _fit(d, f'"{said.group(1).strip()}"' if said else "(no dialogue)", f2, w - 2 * pad),
                           fill=(170, 200, 255) if said else (120, 120, 120), font=f2)
                    y += int(f2.size * 1.45)
                    if s.get("mode") == "CUT" or i == 1:
                        sub = _shot_refs(refs, s, plan.get("style", ""), genders)
                        x = pad
                        for p in (sub or {}).get("pictures", []):
                            a = (p["image"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                            im = Image.fromarray(a)
                            im.thumbnail((int(thumb * 1.4), thumb))
                            if x + im.width > w - pad:
                                break
                            card.paste(im, (x, y))
                            d.text((x + 3, y + 2), f"P{p['n']}", fill=AMBER, font=f2, stroke_width=2, stroke_fill=(0, 0, 0))
                            x += im.width + pad
                        if not (sub or {}).get("pictures"):
                            d.text((pad, y + thumb // 3), "no references for this shot", fill=(120, 120, 120), font=f2)
                    else:
                        d.text((pad, y + thumb // 3), "carries on from the shot before (references only if the face is lost)",
                               fill=(120, 120, 120), font=f2)
            cards.append(np.asarray(card, dtype=np.float32) / 255.0)
        print(f"[LC H3] Proof sheet: {made} of {len(shots)} segments made.")
        return (torch.from_numpy(np.stack(cards)), made)


NODE_CLASS_MAPPINGS = {"LCH3ProofSheet": LCH3ProofSheet}
NODE_DISPLAY_NAME_MAPPINGS = {"LCH3ProofSheet": "LC H3 Proof Sheet (BETA) 🗂️"}
