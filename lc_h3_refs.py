"""
LC H3 Reference Assembler (BETA), v2: the cast, places, props, look and voices of a MiniMax H3 movie, added on the
node itself (no loader nodes). Every item has a role and a name:

  pictures  face / full body / outfit (worn by a named person) / setting / prop / look / composition / layout
  audio     voice (how a named person sounds) / line (an exact recorded line: Segment locks it into the clip)

Pictures are cropped on the node and capped on their long edge (2048 by default, never enlarged), so a wide sheet
can't blow up H3's reference tokens. They are numbered the way H3 reads them (<Picture 1> ... <Picture 9>, then
<Audio 1> ...), with empty and switched-off items closing the gaps.

The bundle (LC_H3_REFS) carries the media with their roles, so LC H3 Segment hands each shot only what it names,
renumbered. Ideas from AdudeGuyman's ComfyUI-Fantastic-MiniMaxH3-PromptBuilder (MIT: role sentences, retention markers)
and Hearmeman24's ComfyUI-MiniMaxRefPack (MIT: compacted numbering, fraction crops, the long-edge cap); written for LC_AV.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

import folder_paths

MAX_PICTURES = 9  # MiniMax H3 reads up to 9 reference pictures
MAX_AUDIOS = 3  # and 3 standalone audio clips
REFS_TYPE = "LC_H3_REFS"
PICTURE_ROLES = ["face", "full body", "outfit", "setting", "view", "prop", "look", "composition", "layout", "off"]
AUDIO_ROLES = ["voice", "line", "off"]
PERSON = {"face", "full body", "outfit", "voice", "line"}  # roles whose name is a person
ALWAYS = {"look", "composition", "layout"}  # roles every shot gets (layout: every shot with more than one person)

EXAMPLE = {"items": []}


# ------------------------------------------------------------------ media
def inside_comfy(path: str) -> str:
    """path as a real path when it sits in ComfyUI's input, output or temp folder, else "". A story, plan or
    Reference Assembler is plain JSON anyone can edit: no file outside those folders is ever read through one."""
    if not path:
        return ""
    real = os.path.realpath(path)
    for base in (folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory()):
        b = os.path.realpath(base)
        try:
            if os.path.commonpath([real, b]) == b:
                return real
        except ValueError:  # another drive
            continue
    return ""


def _path(file):
    f = str(file or "").replace("\\", "/").strip()
    if not f:
        return ""
    p = inside_comfy(f if os.path.isabs(f) else os.path.join(folder_paths.get_input_directory(), f))
    return p if p and os.path.isfile(p) else ""


def load_picture(file, crop=None, max_edge=2048):
    """IMAGE [1, H, W, 3]: EXIF-rotated, cropped (fractions x, y, w, h), long edge capped (never enlarged)."""
    with Image.open(_path(file)) as src:  # closed right away: Windows keeps an open file locked
        im = ImageOps.exif_transpose(src).convert("RGB")
    try:
        x, y, w, h = (float(v) for v in crop) if crop and len(crop) == 4 else (0, 0, 1, 1)
    except (TypeError, ValueError):
        x, y, w, h = 0, 0, 1, 1  # a broken crop: use the whole picture
    if w > 0 and h > 0 and (x, y, w, h) != (0, 0, 1, 1):
        W, H = im.size
        box = (max(0, int(round(x * W))), max(0, int(round(y * H))), min(W, int(round((x + w) * W))), min(H, int(round((y + h) * H))))
        if box[2] - box[0] >= 16 and box[3] - box[1] >= 16:
            im = im.crop(box)
    edge = int(max_edge or 0)
    if edge > 0 and max(im.size) > edge:
        s = edge / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]


def face_frame(img):
    """A face picture shot from far away (a full-length photo) gives H3 a face of a few dozen pixels once it is shrunk
    to the video's size, and it makes up a new person. Uncropped face pictures are framed head and shoulders around
    the biggest face (needs the YuNet model in models/facerestore). Returns (image, framed)."""
    try:
        from .lc_h3_segment import _face_models
        pair = _face_models()
    except Exception:
        pair = None
    if pair is None:
        return img, False
    rgb = (img[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    H, W = rgb.shape[:2]
    det = pair[0]
    det.setInputSize((W, H))
    _, faces = det.detect(np.ascontiguousarray(rgb[..., ::-1]))
    if faces is None or not len(faces):
        return img, False
    fx, fy, fw, fh = (float(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])
    if fw * fh > 0.12 * W * H:
        return img, False  # already a close-up
    bw = min(W, fw * 2.6)
    bh = min(H, bw * 1.25)
    cx, cy = fx + fw / 2, fy + fh * 0.55
    if int(bw) < 16 or int(bh) < 16:
        return img, False  # a speck of a face: framing it would leave a few pixels
    x0 = int(max(0, min(W - bw, cx - bw / 2)))
    y0 = int(max(0, min(H - bh, cy - bh * 0.42)))
    return img[:, y0:y0 + int(bh), x0:x0 + int(bw)], True


def load_audio(file, trim=None):
    """AUDIO {waveform [1, C, S], sample_rate} through ffmpeg, cut to trim [start, end] seconds (end 0 = to the end)."""
    from .lc_save_video import _ffmpeg_bin

    ff = _ffmpeg_bin()
    if not ff:
        raise RuntimeError("[LC H3] Reference Assembler: ffmpeg not found (install it or the imageio-ffmpeg package).")
    cmd = [ff, "-v", "error"]
    s, e = (float(trim[0] or 0), float(trim[1] or 0)) if trim and len(trim) == 2 else (0.0, 0.0)
    if s > 0:
        cmd += ["-ss", f"{s:.3f}"]
    cmd += ["-i", _path(file)]
    if e > s:
        cmd += ["-t", f"{e - s:.3f}"]
    sr = 48000
    raw = subprocess.run(cmd + ["-vn", "-f", "f32le", "-ac", "2", "-ar", str(sr), "pipe:1"], capture_output=True).stdout
    if not raw:
        raise ValueError(f"[LC H3] Reference Assembler: no sound in {file}.")
    wave = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).view(-1, 2).t().unsqueeze(0)
    return {"waveform": wave, "sample_rate": sr}


# ------------------------------------------------------------------ the words
def _join(tags):
    tags = list(tags)
    return tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]


def _note(t):
    t = str(t or "").strip().rstrip(".")
    return f" {t[0].upper() + t[1:]}." if t else ""


_BELOW_FRAME = re.compile(r"\b(skirts?|jeans|trousers|pants|shorts|leggings|stockings|tights|socks|shoes?|boots?|heels?|"
                          r"sneakers?|trainers|sandals?|loafers?|pumps|stilettos?|flats|knee[- ]high)\b", re.I)


def frame_look(text, tight=True):
    """A look for a close-up or medium shot: the clothes that can't be in frame (skirt, jeans, boots, heels ...) left
    out. Named in a tight shot, they pulled the camera back until they fit, or a start frame drew them on extra people
    in the background (Coffee Date: Anika's "black pleated skirt with high heel boots")."""
    text = str(text or "")
    if not tight or not _BELOW_FRAME.search(text):
        return text
    parts = [p.strip() for p in re.split(r",|\band\b|\bwith\b|\.\s", text)]
    keep = [p for p in parts if p and not _BELOW_FRAME.search(p)]
    out = ", ".join(keep).strip(" ,.")
    return re.sub(r"\bwearing\s*$", "", out).strip(" ,")


def _cap(s):
    s = str(s or "").strip()
    return s[0].upper() + s[1:] if s else s


def people(refs):
    """{person name: {"face": [...], "full body": [...], "outfit": [...], "voice": [...], "line": [...]}}."""
    out = {}
    for p in refs.get("pictures", []) + refs.get("audios", []):
        if p["role"] in PERSON:
            out.setdefault(p["name"] or "the person", {}).setdefault(p["role"], []).append(p)
    return out


def sentences(refs):
    """Plain sentences: what H3 takes from each picture and clip."""
    pt = lambda p: f"<Picture {p['n']}>"  # noqa: E731
    at = lambda a: f"<Audio {a['n']}>"  # noqa: E731
    out = []
    for name, r in people(refs).items():
        look = [pt(p) for p in r.get("face", []) + r.get("full body", [])]
        if look:
            what = "the same face, hair and clothes" if r.get("full body") else "the same face and hair"
            rest = "not the clothes or the background" if r.get("outfit") and not r.get("full body") else "not the background"
            out.append(f"{_cap(name)} is the person shown in {_join(look)}: {what}, exactly as shown, but {rest}.")
        for p in r.get("outfit", []):
            out.append(f"{_cap(name)} wears {_outfit_words(name, p.get('note'))} shown in {pt(p)}; nothing else from {pt(p)} is used.")
        for a in r.get("voice", []):
            out.append(f"{_cap(name)}'s voice sounds like {at(a)}: the same timbre and delivery, not its words.")
        for a in r.get("line", []):
            words = a.get("text", "").strip()
            out.append(f"{_cap(name)} says the line heard in {at(a)}" + (f": \"{words}\"" if words else ".") )
    for p in refs.get("pictures", []):
        name, note, tag = p["name"].strip(), _note(p.get("note")), pt(p)
        if p["role"] == "setting":
            out.append(f"{_cap(name) or 'The place'} is the place shown in {tag}, seen from other angles as the camera moves: take only the place and its light, none of its people.{note}")
        elif p["role"] == "view":  # a flat view cut from a 360 panorama: the background from this very camera angle
            out.append(f"{tag} is the exact background behind the people in this shot, seen from this camera angle: the same "
                       f"windows, furniture, light and angle; none of its people.{note}")
        elif p["role"] == "prop":
            out.append(f"The {name or 'object'} is exactly as shown in {tag}: the same shape, colors and any printed text; take only the {name or 'object'}.{note}")
        elif p["role"] == "look":
            out.append(f"{tag} is a look reference: its palette, contrast and light are used; its content is not.{note}")
        elif p["role"] == "composition":
            out.append(f"{tag} is a composition reference: its framing, subject placement and camera height are echoed; its own content is not reproduced.{note}")
        elif p["role"] == "layout":
            clothes = any(q["role"] == "outfit" for q in refs.get("pictures", []))
            out.append(f"{tag} is a scene-and-blocking reference, not a first frame: keep who is where in the room, where the "
                       f"furniture, doors and counter are and which way the doors open; not the poses, faces"
                       f"{', clothes' if clothes else ''} or framing.{note}")
    return " ".join(out)


_CLOTHING = re.compile(r"\b(sweaters?|jumpers?|blouses?|shirts?|t-shirts?|tees?|tops?|jackets?|coats?|parkas?|hoodies?|dress(es)?|"
                       r"skirts?|suits?|jumpsuits?|bodysuits?|catsuits?|spacesuits?|flightsuits?|armou?r|"
                       r"blazers?|vests?|cardigans?|jeans|trousers|pants|shorts|uniforms?|scarf|scarves|hats?|caps?|"
                       r"gowns?|robes?|aprons?|overalls|tank|leggings|stockings|tights|socks|shoes?|boots?|heels?|sneakers?|"
                       r"trainers|sandals?|loafers?|pumps|stilettos?|wearing|dressed|outfit|clothes|clothing)\b", re.I)


def outfit_name(notes, fallback="outfit 1"):
    """A short name for a cast card's own 👗 outfit from its notes, up to the first garment: "dark blue jumpsuit",
    "red and grey jumpsuit", "white sweater". The label is all H3 reads about an outfit picture, and a meaningless
    "outfit 1" over a bare-shouldered headshot gave two naked women ("Uber run", 2026-11-07); a label that names the
    garment held it 8 of 8 (2026-11-02). No garment in the notes: the fallback."""
    for part in re.split(r"[,;.]", str(notes or "")):
        m = next((m for m in _CLOTHING.finditer(part)
                  if m.group(1).lower() not in ("wearing", "dressed", "outfit", "clothes", "clothing")), None)
        if m:
            head = re.sub(r"^\s*((wearing|dressed in|in)\s+)?(a|an|the)?\s*", "", part[:m.end()].strip(), flags=re.I)
            words = head.split()
            return " ".join(words[-6:]).lower() or fallback
    return fallback


def strip_clothes(text):
    """A look with every clothing clause taken out: the outfit picture shows the clothes, and words beat pictures
    (2026-11-02: "red leather jacket" in the words won over a white-sweater picture)."""
    text = str(text or "")
    if not _CLOTHING.search(text):
        return text
    parts = [p.strip() for p in re.split(r",|\band\b|\bwith\b|\.\s", text)]
    return ", ".join(p for p in parts if p and not _CLOTHING.search(p)).strip(" ,.")


def _outfit_words(name, note):
    """ "Anika's outfit 1" / "Anika's red dress": the person's own, never "the the red dress"."""
    what = re.sub(r"^(the|a|an)\s+", "", str(note or "").strip(), flags=re.I).rstrip(".") or "outfit"
    return f"{_cap(name)}'s {what}" if name else f"the {what}"


def _is_are(words):
    """ "gym clothes are", "jeans and a tee are", "red dress is", "outfit 2 is": two things joined by "and", or a last
    word ending in one s (not ss), is plural."""
    ws = re.findall(r"[a-z]+", str(words).lower())
    last = ws[-1] if ws else ""
    return "are" if "and" in ws or (last.endswith("s") and not last.endswith("ss") and len(last) > 2) else "is"


def retention(refs, subject_of, tight=False):
    """H3's trained retention_analysis lines. subject_of: {person name: "<Subject k>"}. tight: a close-up or medium
    shot, so the notes leave out the clothes that can't be in frame."""
    pt = lambda p: f"<Picture {p['n']}>"  # noqa: E731
    at = lambda a: f"<Audio {a['n']}>"  # noqa: E731
    lines = []
    for name, r in people(refs).items():
        subj = subject_of.get(name, name)
        look = r.get("face", []) + r.get("full body", [])
        if look:
            what = "face, hair and clothing" if r.get("full body") else "face and hair"
            note = "; ".join(frame_look(p["note"].strip(), tight) for p in look if p.get("note", "").strip())
            # an outfit picture rides along: the headshot gives the face and hair only. "Only the person" took its clothes
            # too, and its white sweater beat outfit 2's red dress (2026-11-02: a white top cut like the dress)
            only = "the face and hair come" if r.get("outfit") and not r.get("full body") else "the person comes"
            rest = "not the clothes or the background" if r.get("outfit") and not r.get("full body") else "not the background"
            lines.append(f"{subj} ({', '.join(pt(p) for p in look)}): fully_preserved - the {what} of {subj} are retained"
                         + (f" ({note})" if note else "") + f". Only {only} from {_join([pt(p) for p in look])}; {rest}.")
        for p in r.get("outfit", []):
            ow = _outfit_words(name, p.get("note"))
            lines.append(f"{pt(p)}: attribute_transfer - {ow} {_is_are(ow)} worn by {subj}; nothing else from this picture is used.")
        for a in r.get("voice", []):
            lines.append(f"{at(a)}: reference - the voice of {subj}, guiding timbre and delivery without copying its words.")
        for a in r.get("line", []):
            lines.append(f"{at(a)}: fully_copy - the line {subj} says, used exactly as recorded.")
    for p in refs.get("pictures", []):
        name = p["name"].strip()
        if p["role"] == "setting":
            lines.append(f"{pt(p)} ({name or 'the place'}): reference - only the place and its light are used, none of its people.")
        elif p["role"] == "view":
            lines.append(f"{pt(p)} ({name or 'the background'}): fully_preserved - the background behind the subject, from this "
                         "exact camera angle: its windows, furniture and light stay where they are. Only the background is used.")
        elif p["role"] == "prop":
            lines.append(f"{pt(p)} ({name or 'the object'}): fully_preserved - the shape, colors and any printed text of the {name or 'object'} are retained.")
        elif p["role"] == "look":
            lines.append(f"{pt(p)}: weak_reference - only its palette, contrast and light are used.")
        elif p["role"] == "composition":
            lines.append(f"{pt(p)}: weak_reference - only its framing, subject placement and camera height are echoed.")
        elif p["role"] == "layout":
            # outfit pictures in this shot: the clothes come from them, not from the earlier shot this frame is from
            # (2026-11-02: Anika kept the layout frame's sweater instead of her outfit 2)
            clothes = any(q["role"] == "outfit" for q in refs.get("pictures", []))
            lines.append(f"{pt(p)}: weak_reference - a scene-and-blocking reference, not a first frame: only the layout of the "
                         "scene is kept: who is where in the room, which side of the table each person is on, the furniture, the "
                         "window, the doors and which way they open. Not the poses, the faces"
                         + (", the clothes" if clothes else "") + " or the framing.")
    return lines


def subset(refs, include, lines=None):
    """What one shot needs, renumbered: people in `include` (their face, full body, outfit, voice), settings and props
    in `include`, look and composition always. Recorded lines only those in `lines` (ids), when given."""
    inc = {n.strip().lower() for n in include}

    def keep(p):
        nm = (p["name"] or "").strip().lower()
        if p["role"] == "layout" and nm:  # a layout with a place name (an anchor) goes to shots at that place
            return nm in inc
        return p["role"] in ALWAYS or nm in inc
    pics = [{**p, "n": i + 1} for i, p in enumerate([p for p in refs.get("pictures", []) if keep(p)])]
    auds = [a for a in refs.get("audios", []) if keep(a) and (a["role"] != "line" or lines is None or a["id"] in lines)]
    auds = [{**a, "n": i + 1} for i, a in enumerate(auds)]
    out = {"pictures": pics, "audios": auds}
    out["text"] = sentences(out)
    return out


# ------------------------------------------------------------------ the sheet
def _font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wave_tile(audio, w=260, h=150):
    tile = Image.new("RGB", (w, h), (30, 34, 40))
    d = ImageDraw.Draw(tile)
    x = audio["waveform"][0].mean(0).numpy()
    if x.size:
        step = max(1, x.size // w)
        peaks = np.abs(x[: step * w].reshape(-1, step)).max(1) if x.size >= w else np.abs(x)
        peaks = peaks / (peaks.max() or 1)
        for i, v in enumerate(peaks[:w]):
            d.line([(i, h / 2 - v * h * 0.42), (i, h / 2 + v * h * 0.42)], fill=(80, 170, 200))
    return tile


def sheet(refs, height=260):
    """Every picture and clip with its <Picture n> / <Audio n> tag, role and name under it (wraps after 5)."""
    tiles = []
    f_big, f_small = _font(18), _font(14)
    items = [("Picture", p) for p in refs.get("pictures", [])] + [("Audio", a) for a in refs.get("audios", [])]
    if not items:  # a card that says so, instead of a black square
        card = Image.new("RGB", (640, 360), (24, 24, 24))
        d = ImageDraw.Draw(card)
        for y, txt, f, col in ((140, "No reference pictures", f_big, (238, 187, 68)),
                               (172, "The story's words do all the work. Add pictures on the Assembler.", f_small, (200, 200, 200))):
            d.text(((640 - d.textlength(txt, font=f)) // 2, y), txt, fill=col, font=f)
        return torch.from_numpy(np.asarray(card, dtype=np.float32) / 255.0)[None]
    for kind, p in items:
        if kind == "Picture":
            im = Image.fromarray((p["image"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            im.thumbnail((int(height * 1.6), height))
        else:
            im = _wave_tile(p["audio"], h=height)
        tile = Image.new("RGB", (max(im.width, 210), im.height + 48), (24, 24, 24))
        tile.paste(im, ((tile.width - im.width) // 2, 0))
        d = ImageDraw.Draw(tile)
        d.text((8, im.height + 4), f"<{kind} {p['n']}>", fill=(238, 187, 68), font=f_big)
        d.text((8, im.height + 26), f"{p['role']} · {p['name'] or '(no name)'}"[:42], fill=(220, 220, 220), font=f_small)
        tiles.append(tile)
    per_row, gap = 5, 8
    rows = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
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


# ------------------------------------------------------------------ the node
def build(references, max_edge=2048):
    try:
        data = json.loads(references or "{}")
    except Exception:
        raise ValueError("[LC H3] Reference Assembler: the list is not valid. Undo the last change, or delete the node and add a new one.")
    items = data.get("items", []) if isinstance(data, dict) else None
    if not isinstance(items, list):  # valid JSON but not {"items": [...]}
        raise ValueError("[LC H3] Reference Assembler: the list is not valid. Undo the last change, or delete the node and add a new one.")
    pictures, audios, warn = [], [], []
    seen = set()
    for k, it in enumerate([x for x in items if isinstance(x, dict)]):
        kind = it.get("kind", "image")
        role = str(it.get("role") or "")
        file = it.get("file", "")
        label = f"{kind} {k + 1} ({os.path.basename(str(file)) or 'no file'})"
        if not it.get("on", True) or role == "off":
            continue
        if not role:  # a card brought in with nothing filled in: none until you pick one
            warn.append(f"{label}: pick a role (and a name). It isn't sent until it has one.")
            continue
        if not _path(file):
            warn.append(f"{label}: the file is missing. Add it again.")
            continue
        if (file, json.dumps(it.get("crop")), json.dumps(it.get("trim"))) in seen:
            warn.append(f"{label}: the same file is in the list twice.")
        seen.add((file, json.dumps(it.get("crop")), json.dumps(it.get("trim"))))
        name = str(it.get("name", "")).strip()
        base = {"id": str(it.get("id") or k), "role": role, "name": name, "note": str(it.get("note", "")), "file": file}
        if kind == "image":
            if role not in PICTURE_ROLES:
                role = base["role"] = "face"
            if role in PERSON and not name:
                warn.append(f"{label}: name the person (shots find people by their names).")
            if role in ("setting", "view", "prop") and not name:
                warn.append(f"{label}: name the {role} (e.g. 'the kitchen', 'red cup') so shots can ask for it.")
            if role in ("face", "full body") and name and not base["note"].strip():
                warn.append(f"{label}: describe {name}'s look in the note. H3 follows the words more than the picture.")
            if role == "outfit" and not base["note"].strip():
                warn.append(f"{label}: say what the outfit is in the note (e.g. 'red raincoat').")
            try:
                img = load_picture(file, it.get("crop"), max_edge)
                crop = it.get("crop")
                if role == "face" and (not crop or [float(v) for v in crop] == [0, 0, 1, 1]):
                    img = face_frame(img)[0]
            except Exception as e:
                warn.append(f"{label}: could not be read ({e}).")
                continue
            pictures.append({**base, "n": len(pictures) + 1, "image": img})
        else:
            if role not in AUDIO_ROLES:
                role = base["role"] = "voice"
            if not name:
                warn.append(f"{label}: name the person whose {role} this is.")
            text = str(it.get("text", "")).strip()
            if role == "line" and not text:
                warn.append(f"{label}: type the words of the line, so Segment can find the shot that says it.")
            try:
                aud = load_audio(file, it.get("trim"))
            except Exception as e:
                warn.append(f"{label}: could not be read ({e}).")
                continue
            secs = aud["waveform"].shape[-1] / aud["sample_rate"]
            if role == "voice" and not (2.0 <= secs <= 15.5):
                warn.append(f"{label}: a voice clip works best at 2 to 15 seconds (this one is {secs:.1f} s). Trim it.")
            audios.append({**base, "n": len(audios) + 1, "audio": aud, "text": text, "seconds": secs})
    if len(pictures) > MAX_PICTURES:
        warn.append(f"{len(pictures)} pictures: H3 reads 9 at most per shot. Each shot only sends the ones it names, so keep any one shot to 9 or fewer.")
    if len([a for a in audios if a["role"] == "voice"]) > MAX_AUDIOS:
        warn.append(f"More than {MAX_AUDIOS} voices: H3 reads 3 audio clips per shot at most.")
    refs = {"version": 2, "pictures": pictures, "audios": audios}
    for name, r in people(refs).items():
        if not r.get("face") and (r.get("full body") or r.get("outfit")):
            warn.append(f"{_cap(name)}: no face picture. A face close-up holds a face far better than a full body.")
        if (r.get("voice") or r.get("line")) and not (r.get("face") or r.get("full body")):
            warn.append(f"{_cap(name)}: a voice but no picture. Add a face so H3 knows who speaks.")
    if not pictures and not audios:
        warn.append("Nothing added yet: use + Pictures / + Audio on the node.")
    refs["text"] = sentences(refs)
    warnings = "\n".join(dict.fromkeys(warn))  # the caller (Story) prints them
    return refs, warnings
