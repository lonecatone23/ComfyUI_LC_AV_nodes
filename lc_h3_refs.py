"""
LC H3 Reference Assembler (BETA): the reference images for a MiniMax H3 movie, each with a role and a name. Cast (a
face close-up and / or a full body), settings, props and a style picture. It numbers them the way H3 reads them
(<Picture 1> ... <Picture 9>) and writes the sentences that tell H3 what to take from each one: the person's face,
the place but not its people, the prop exactly as shown, only the look.

The bundle (LC_H3_REFS) carries the images with their roles, so a later node can hand H3 only the references a shot
needs, renumbered, with the sentences rewritten to match.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from comfy_api.latest import io

MAX_PICTURES = 9  # MiniMax H3 reads up to 9 reference pictures
SLOTS = 9
ROLES = ["cast face", "cast full body", "setting", "prop", "style", "off"]
REFS_TYPE = "LC_H3_REFS"
NAMES = [f"image_{i}" for i in range(1, SLOTS + 1)]

EXAMPLE = {"items": [
    {"slot": 1, "role": "cast face", "name": "Lonecat", "note": ""},
    {"slot": 2, "role": "setting", "name": "the fishing pier", "note": ""},
]}


def _join(tags):
    tags = list(tags)
    return tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]


def _note(t):
    t = str(t or "").strip().rstrip(".")
    return f" {t[0].upper() + t[1:]}." if t else ""


def sentences(pictures, include=None):
    """The role sentences for these pictures. pictures: [{n, role, name, note}] (n = the <Picture n> number).
    include: names to keep (a shot's cast, its setting and props); None keeps every picture."""
    pics = [p for p in pictures if include is None or (p["name"] or "").strip().lower() in include or p["role"] == "style"]
    out = []
    cast = {}
    for p in pics:
        if p["role"].startswith("cast"):
            cast.setdefault(p["name"].strip() or "The person", []).append(p)
    for name, ps in cast.items():
        faces = [f"<Picture {p['n']}>" for p in ps if p["role"] == "cast face"]
        bodies = [f"<Picture {p['n']}>" for p in ps if p["role"] == "cast full body"]
        tags = faces + bodies
        if faces and bodies:
            what = "the same face, hair and clothes"
        elif faces:
            what = "the same face and hair"
        else:
            what = "the same look and clothes"
        s = f"{name} is the person shown in {_join(tags)}: {what}, exactly as shown, but not the background."
        out.append(s + _note("; ".join(p["note"].strip() for p in ps if p.get("note", "").strip())))
    for p in pics:
        tag = f"<Picture {p['n']}>"
        name = p["name"].strip()
        note = _note(p.get("note"))
        if p["role"] == "setting":
            place = (name[0].upper() + name[1:]) if name else "The place"
            out.append(f"{place} is the place shown in {tag}: take only the place and its light from {tag}, none of its people.{note}")
        elif p["role"] == "prop":
            thing = name or "object"
            out.append(f"The {thing} is exactly as shown in {tag}, the same shape, colors and any printed text; take only the {thing} from {tag}.{note}")
        elif p["role"] == "style":
            out.append(f"The look follows {tag}: its colors, light and grain. Take nothing else from {tag}.{note}")
    return " ".join(out)


def subset(refs, include):
    """The references one shot needs (its cast, setting, props; style always), renumbered from <Picture 1>, with
    the sentences rewritten. include: a set of lower-case names."""
    keep = [p for p in refs["pictures"] if (p["name"] or "").strip().lower() in include or p["role"] == "style"]
    pics = [{**p, "n": i + 1} for i, p in enumerate(keep)]
    return {"pictures": pics, "text": sentences(pics)}


def _font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _sheet(pictures, height=300):
    """Every picture with its number, role and name under it, in one strip (wraps after 5)."""
    if not pictures:
        return torch.zeros(1, 64, 64, 3)
    tiles = []
    f_big, f_small = _font(18), _font(14)
    for p in pictures:
        a = (p["image"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(a)
        im.thumbnail((int(height * 1.6), height))
        tile = Image.new("RGB", (max(im.width, 200), im.height + 48), (24, 24, 24))
        tile.paste(im, ((tile.width - im.width) // 2, 0))
        d = ImageDraw.Draw(tile)
        d.text((8, im.height + 4), f"<Picture {p['n']}>", fill=(238, 187, 68), font=f_big)
        d.text((8, im.height + 26), f"{p['role']} · {p['name'] or '(no name)'}"[:40], fill=(220, 220, 220), font=f_small)
        tiles.append(tile)
    per_row = 5
    rows = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
    gap = 8
    w = max(sum(t.width for t in r) + gap * (len(r) - 1) for r in rows)
    h = sum(max(t.height for t in r) for r in rows) + gap * (len(rows) - 1)
    out = Image.new("RGB", (w, h), (16, 16, 16))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            out.paste(t, (x, y))
            x += t.width + gap
        y += max(t.height for t in r) + gap
    return torch.from_numpy(np.asarray(out, dtype=np.float32) / 255.0)[None]


class LCH3ReferenceAssembler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LCH3ReferenceAssembler",
            display_name="LC H3 Reference Assembler (BETA) 🖼️",
            category="LC AV/H3 Long Video",
            description=("Reference pictures for MiniMax H3, each with a role: a person's face or full body, a setting, a "
                         "prop or a style. Numbers them <Picture 1> to <Picture 9> and writes what H3 should take from each "
                         "one (the face exactly, the place but not its people, the prop as shown, only the look)."),
            inputs=[
                io.String.Input("references", default=json.dumps(EXAMPLE), multiline=False,
                                tooltip="The roles and names as JSON. Fill them in on the node."),
                io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(io.Image.Input("image"), names=NAMES, min=0),
                                  tooltip="Reference pictures. A new socket appears as you fill the last one (9 at most)."),
            ],
            outputs=[
                io.Custom(REFS_TYPE).Output(display_name="refs", tooltip="The references with their roles, for the H3 nodes."),
                io.String.Output(display_name="ref_text", tooltip="The sentences that tell H3 what to take from each picture. Put them at the start of a prompt."),
                io.Image.Output(display_name="sheet", tooltip="Every picture with its <Picture n> number, role and name."),
                io.String.Output(display_name="warnings", tooltip="What to fix. Empty when it all looks good."),
            ],
        )

    @classmethod
    def execute(cls, references, images=None) -> io.NodeOutput:
        return io.NodeOutput(*cls.build(references, **(images or {})))

    @staticmethod
    def build(references, **images):
        try:
            data = json.loads(references or "{}")
        except Exception:
            raise ValueError("[LC H3] Reference Assembler: the roles are not valid. Undo the last change, or delete the node and add a new one.")
        rows = {int(r.get("slot", 0)): r for r in data.get("items", []) if isinstance(r, dict)}
        pictures, warn = [], []
        for i in range(1, SLOTS + 1):
            img = images.get(f"image_{i}")
            if not torch.is_tensor(img) or img.ndim != 4 or img.shape[0] == 0:
                img = None
            r = rows.get(i, {})
            role = r.get("role") or "cast face"
            if img is None:
                continue
            if role == "off":
                continue
            if role not in ROLES:
                role = "cast face"
            name = str(r.get("name", "")).strip()
            if role.startswith("cast") and not name:
                warn.append(f"Picture in slot {i}: give the person a name. The shots find people by their names.")
            if role in ("setting", "prop") and not name:
                warn.append(f"Picture in slot {i}: name the {role} (e.g. 'the kitchen', 'red cup') so shots can ask for it.")
            pictures.append({"n": len(pictures) + 1, "slot": i, "role": role, "name": name, "note": str(r.get("note", "")), "image": img[:1]})
        if len(pictures) > MAX_PICTURES:
            warn.append(f"{len(pictures)} pictures: H3 reads 9 at most. Turn some off.")
            pictures = pictures[:MAX_PICTURES]
        faces = {}
        for p in pictures:
            if p["role"] == "cast face":
                faces.setdefault(p["name"].lower(), 0)
                faces[p["name"].lower()] += 1
        for p in pictures:
            if p["role"] == "cast full body" and p["name"].lower() not in faces:
                warn.append(f"{p['name'] or 'Slot ' + str(p['slot'])}: only a full body picture. A face close-up holds the face far better.")
        if not pictures:
            warn.append("No pictures connected.")
        text = sentences(pictures)
        refs = {"pictures": pictures, "text": text}
        warnings = "\n".join(dict.fromkeys(warn))
        if warnings:
            print("[LC H3] Reference Assembler: " + " | ".join(dict.fromkeys(warn)))
        return (refs, text, _sheet(pictures), warnings)


NODE_CLASS_MAPPINGS = {"LCH3ReferenceAssembler": LCH3ReferenceAssembler}
NODE_DISPLAY_NAME_MAPPINGS = {"LCH3ReferenceAssembler": "LC H3 Reference Assembler (BETA) 🖼️"}
