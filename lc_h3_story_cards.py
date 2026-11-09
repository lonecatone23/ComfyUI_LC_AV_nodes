"""
Pictures for LC H3 Story, made with any image model (the image model lives in the workflow, never in these nodes).

LC H3 Story Card Prompt: a cast or place card -> an image prompt and an empty latent at the size that picture wants
  (portrait, room, 2:1 360 panorama). Its project_name comes from the Story node, so Story saves before it reads.
LC H3 Save to Story: the picture -> input/lc_h3_refs/<project>/ and onto the card (cast picture, room picture, 360,
  shot start / end frame). A name with no card yet makes one. story.json on disk is updated, and an open Story node takes the
  picture at once (lc_story_card in the ui).
LC H3 Shot Card: one shot's start or end frame as a storyboard job for any image model: the prompt (style, place, camera,
  the people in it with their looks stamped word for word, the frozen moment), the 📐 blocking boxes (elements for LC MOSS
  Layout Prompt / LC Flux 3 Layout Prompt, plus the same as words), the background (the shot's 360 view or room picture)
  and the cast pictures. Blank shot = the next empty slot, so one module run fills the board slot by slot.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image

import folder_paths

from .lc_h3_refs import frame_look, strip_clothes
from .lc_h3_story import (TIGHT, wardrobe, _framing, _names, _one_clause, _pic_abs, _pic_file, _place, _shot_view, _story_path,
                           _view_path, _with_auto_views, _write_json)

KINDS = ["cast card", "cast card + outfit", "cast outfit", "outfit", "place", "360 panorama", "prop"]
SAVE_KINDS = ["from card", "cast card", "cast outfit", "outfit", "place", "360 panorama", "prop", "shot start", "shot end"]
SIZES = {"cast card": (1024, 1024), "cast outfit": (896, 1152), "outfit": (896, 1152), "place": (1344, 768), "360 panorama": (2048, 1024),
         "prop": (1024, 1024)}
GREY_KINDS = {"cast card", "cast outfit", "outfit", "prop"}  # cut out onto flat grey: a card's own background leaks into every shot it is in


def _kind(kind: str) -> str:
    return "cast card" if kind == "character" else kind  # workflows saved before the rename
SUFFIX = "Natural light, sharp focus, photographic."


def _safe(name) -> str:
    safe = re.sub(r"[^A-Za-z0-9 _.-]", "_", str(name or "").strip())
    return safe if safe.strip(" .") else "story"  # "." / ".." would step out of lc_h3_refs/


def _read(project: str) -> dict:
    p = _story_path(project)
    if not os.path.exists(p):
        raise ValueError(f"[LC H3] No story saved for '{project}' yet. Run (or 💾 Save) the LC H3 Story node once.")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _sentences(text: str) -> str:
    text = re.sub(r"\.\s*\.", ".", re.sub(r"\s+", " ", text)).strip()
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def _find(items: list, name: str) -> dict | None:
    key = str(name or "").strip().lower()
    if not key:
        return items[0] if items else None
    return next((x for x in items if str(x.get("name") or "").strip().lower() == key), None)


SLOT_OF = {"cast card": "picture", "cast outfit": "outfit_picture", "outfit": "picture", "prop": "picture", "place": "picture",
           "360 panorama": "panorama"}
WHAT_OF = {"cast card": "cast", "cast outfit": "cast", "outfit": "outfit", "prop": "prop", "place": "place", "360 panorama": "place"}


def _resolve(items: list, name: str, slot: str, what: str) -> dict | None:
    """The card a picture job is for. A name picks that card (the job makes a new take of it). A blank name picks the first card
    whose slot is still empty and refuses when every card has a picture: a blank name never writes over a picture."""
    if str(name or "").strip():
        return _find(items, name)
    for x in items:
        if not _pic_file(x.get(slot)):
            return x
    names = ", ".join(str(x.get("name") or "?") for x in items) or "none yet"
    raise ValueError(f"[LC H3] Name the {what} card: each one already has a picture ({names}). A blank name never writes over a picture. "
                     "Type the card's name to make a new take of it.")


def _keep_old(card: dict, slot: str):
    """The picture a slot had goes on the card's list of earlier ones (newest first, 8 kept), so ↶ on the slot can bring it back."""
    old = card.get(slot)
    if not isinstance(old, dict) or not old.get("filename"):
        return
    prev = card.setdefault("prev", {}).setdefault(slot, [])
    prev[:] = [p for p in prev if p.get("filename") != old.get("filename")]
    prev.insert(0, {k: v for k, v in old.items() if k in ("type", "subfolder", "filename", "crop")})
    del prev[8:]


def card_prompt(story: dict, kind: str, name: str, style: str = "") -> tuple:
    """(prompt, card name) for one picture, in plain sentences from the card's own words."""
    extra = f" {style.strip().rstrip('.')}." if style.strip() else ""
    kind = _kind(kind)
    if kind == "outfit":
        o = _resolve(story.get("outfits", []), name, "picture", "outfit")
        if not o:
            raise ValueError(f"[LC H3] No outfit card named '{name}'. Add it on the Story node first.")
        # the description paints the picture in full detail; H3 only ever reads the card's simple name ("gym clothes")
        desc = str(o.get("description") or o.get("name") or "").strip().rstrip(".")
        wearer = _find(story.get("cast", []), o.get("who")) if str(o.get("who") or "").strip() else None
        if wearer:
            # on the person, head to toe: a chest-up picture left the rest to H3, and it put back the old skirt and boots
            # (2026-11-02, Anika's outfit 2)
            who = str(wearer.get("who") or "person").strip()
            look = strip_clothes(str(wearer.get("look") or "")).strip().rstrip(".")
            return _sentences(f"A full-length photo of a {who}{', ' + look if look else ''}, wearing {desc}. Standing, head to toe, "
                              f"the whole outfit in view, facing the camera, soft even light, plain light grey "
                              f"background.{extra} {SUFFIX}"), o.get("name", "")
        return _sentences(f"A fashion catalog photo of {desc} on an invisible mannequin, front view, the whole outfit in view, "
                          f"soft even light, plain light grey background.{extra} {SUFFIX}"), o.get("name", "")
    if kind == "prop":
        r = _resolve(story.get("props", []), name, "picture", "prop")
        if not r:
            raise ValueError(f"[LC H3] No prop card named '{name}'. Add it on the Story node first.")
        desc = str(r.get("description") or r.get("name") or "").strip().rstrip(".")
        return _sentences(f"A product photo of {desc}, alone in the middle of the frame, the whole object in view, soft even light, "
                          f"plain light grey background.{extra} {SUFFIX}"), r.get("name", "")
    if kind == "cast outfit":
        c = _resolve(story.get("cast", []), name, "outfit_picture", "cast")
        if not c:
            raise ValueError(f"[LC H3] No cast card named '{name}'. Add it on the Story node first.")
        who = str(c.get("who") or "person").strip()
        look = str(c.get("look") or "").strip().rstrip(".")
        wear = str(c.get("outfit_notes") or "").strip().rstrip(".")
        # the card's 👗 outfit picture: the person head to toe in the outfit. Its words are only for this picture; H3
        # never reads them (the picture carries the clothes)
        return _sentences(f"A full-length photo of a {who}{', ' + look if look else ''}{', wearing ' + wear if wear else ''}. Standing, "
                          f"head to toe, the whole outfit in view, facing the camera, soft even light, plain light grey background.{extra} {SUFFIX}"), c.get("name", "")
    if kind == "cast card":
        c = _resolve(story.get("cast", []), name, "picture", "cast")
        if not c:
            raise ValueError(f"[LC H3] No cast card named '{name}'. Add it on the Story node first.")
        who = str(c.get("who") or "person").strip()
        look = strip_clothes(str(c.get("look") or "")).strip().rstrip(".")
        # a headshot, never clothes: the headshot's sweater beat every outfit picture (2026-11-02). Not "head and
        # shoulders" or a torso: shoulders bring a top with them. Clothes come from outfit cards only.
        return _sentences(f"A headshot of a {who}{' with ' + look if look else ''}. Looking straight at the camera with a calm, "
                          f"neutral expression, soft even light, plain light grey background.{extra} {SUFFIX}"), c.get("name", "")
    p = _resolve(story.get("places", []), name, SLOT_OF[kind], "place")
    if not p:
        raise ValueError(f"[LC H3] No place card named '{name}'. Add it on the Story node first.")
    desc = str(p.get("description") or p.get("name") or "").strip().rstrip(".")
    if kind == "360 panorama":
        read = " ".join(str(p.get("pano_prompt") or "").split())
        if read:  # 🌐 360 from the room picture: LC Vision's reading of it, the whole place all the way round
            return (f"A 360 degree equirectangular panorama photo, taken from the middle of the place at seated eye level. "
                    f"{read} No people.{extra} Seamless 360 panorama, horizon level in the middle."), p.get("name", "")
        return _sentences(f"A 360 degree equirectangular panorama photo, taken from the middle of {desc} at seated eye level. "
                          f"No people.{extra} Seamless 360 panorama, horizon level in the middle."), p.get("name", "")
    return _sentences(f"A wide establishing photo of {desc}, with nobody in it. Eye level, 35mm lens, the whole space in "
                      f"view.{extra} {SUFFIX}"), p.get("name", "")


class LCH3StoryCardPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"forceInput": True, "tooltip": "From LC H3 Story 🎬 (so the story is saved before it is read)."}),
                "vae": ("VAE", {"tooltip": "The image model's VAE: the empty latent is made to fit it."}),
                "kind": (KINDS, {"default": "cast card", "tooltip": "cast card: a front-facing headshot from the look, on plain grey. cast card + outfit: the headshot and the person head to toe in their outfit (from the card's outfit notes), one run: both slots of the cast card get filled. cast outfit: only the 👗 outfit picture. outfit: a change of clothes on an invisible mannequin, on plain grey. place: the room, empty. 360 panorama: the room all the way round (2:1), for the 🌐 slot. prop: the object alone, on plain grey."}),
                "name": ("STRING", {"default": "", "tooltip": "The cast, outfit, place or prop card. Blank = the first one."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "0 = the size this kind wants (cast card and prop 1024 x 1024, outfit 896 x 1152, place 1344 x 768, 360 2048 x 1024)."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "0 = the size this kind wants."}),
                "style": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional. Words added to the prompt (a film look, a period, an art style)."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "prompt", "card")
    OUTPUT_IS_LIST = (True, True, True)  # cast card + outfit: two pictures, everything after runs once for each
    OUTPUT_TOOLTIPS = ("An empty latent at the picture's size.", "The image prompt. Edit it on the way if you like.",
                       "Which card the picture is for: plug into LC H3 Save to Story.")
    FUNCTION = "build"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("Turns a Story cast, outfit, place or prop card into an image prompt and an empty latent for any image model. "
                   "Save the picture with LC H3 Save to Story and it lands on the card.")

    @classmethod
    def VALIDATE_INPUTS(cls, kind):
        return True if _kind(kind) in KINDS else f"kind must be one of: {', '.join(KINDS)}"

    def build(self, project_name, vae, kind, name, width=0, height=0, style=""):
        kind = _kind(kind)
        story = _read(project_name)
        ch = getattr(vae, "latent_channels", 16)
        down = vae.spacial_compression_encode() if hasattr(vae, "spacial_compression_encode") else 8
        latents, prompts, cards = [], [], []
        if kind == "cast card + outfit" and not str(name or "").strip():  # one person for both pictures: the first with a slot still empty
            cast = story.get("cast", [])
            pick = next((c for c in cast if not _pic_file(c.get("picture")) or not _pic_file(c.get("outfit_picture"))), None)
            if pick is None:
                _resolve(cast, "", "picture", "cast")  # raises: says which cards are full
            name = str(pick.get("name") or "")
        # cast card + outfit: the headshot (who she is) and the 👗 outfit picture (what she wears), one run
        for k in (["cast card", "cast outfit"] if kind == "cast card + outfit" else [kind]):
            prompt, card_name = card_prompt(story, k, name, style)
            dw, dh = SIZES[k]
            w, h = max(16, (width or dw) // 16 * 16), max(16, (height or dh) // 16 * 16)  # a typed 8 would make an empty latent
            latents.append({"samples": torch.zeros([1, ch, h // down, w // down])})
            prompts.append(prompt)
            cards.append(json.dumps({"project": project_name, "kind": k, "name": card_name}, ensure_ascii=False))
        return {"ui": {"text": ["\n\n".join(prompts)]}, "result": (latents, prompts, cards)}


def save_to_story(img: Image.Image, project: str, kind: str, name: str) -> dict:
    """Saves the picture and puts it on the card in story.json. Returns the patch the Story node applies."""
    kind = _kind(kind)
    story = _read(project)
    if kind in GREY_KINDS:
        from .lc_h3_cutout import on_grey

        img = on_grey(img)
    sub = f"lc_h3_refs/{_safe(project)}"
    folder = os.path.join(folder_paths.get_input_directory(), sub)
    os.makedirs(folder, exist_ok=True)
    tag = {"cast card": "cast", "cast outfit": "dress", "outfit": "outfit", "place": "place", "360 panorama": "pano", "prop": "prop",
           "shot start": "start", "shot end": "end"}[kind]
    base = f"{tag}_{_safe(name or 'card').replace(' ', '_')}"
    fname, k = f"{base}.png", 2
    while os.path.exists(os.path.join(folder, fname)):  # never over an older picture: takes made from it keep it
        fname, k = f"{base}_{k}.png", k + 1
    img.save(os.path.join(folder, fname))
    pic = {"type": "input", "subfolder": sub, "filename": fname}
    made = False
    slot = SLOT_OF.get(kind)
    if kind == "outfit":
        items = story.setdefault("outfits", [])
        card = _resolve(items, name, slot, "outfit")
        if not card:
            card = {"id": f"o{int(time.time() * 1000) % 10**8}", "name": name or "the outfit", "who": "", "description": "", "picture": None}
            items.append(card)
            made = True
    elif kind == "prop":
        items = story.setdefault("props", [])
        card = _resolve(items, name, slot, "prop")
        if not card:
            card = {"id": f"r{int(time.time() * 1000) % 10**8}", "name": name or "the prop", "description": "", "picture": None}
            items.append(card)
            made = True
    elif kind in ("cast card", "cast outfit"):
        items = story.setdefault("cast", [])
        card = _resolve(items, name, slot, "cast")
        if not card:
            card = {"id": f"c{int(time.time() * 1000) % 10**8}", "name": name or "New", "who": "person", "look": "", "picture": None, "voice": None}
            items.append(card)
            made = True
    elif kind in ("place", "360 panorama"):
        items = story.setdefault("places", [])
        card = _place(story, name) if name.strip() else _resolve(items, name, slot, "place")
        if not card:
            card = {"id": f"p{int(time.time() * 1000) % 10**8}", "name": name or "the place", "description": "", "sound": "", "picture": None, "anchors": []}
            items.append(card)
            made = True
    else:
        shots = story.get("shots", [])
        key = str(name or "").strip()
        card = next((s for s in shots if str(s.get("id")) == key), None)
        if card is None and key.isdigit() and 0 < int(key) <= len(shots):
            card = shots[int(key) - 1]
        if card is None:
            raise ValueError(f"[LC H3] Save to Story: no shot '{key}'. Use the shot number (1 = the first shot).")
        slot = "end" if kind == "shot end" else "start"
    _keep_old(card, slot)  # the picture that was there is not lost: ↶ on the slot brings it back
    card[slot] = pic
    story["saved"] = time.time()
    _write_json(_story_path(project), story)
    return {"project": project, "kind": kind, "id": card.get("id"), "card": card, "new": made, "saved": story["saved"], "pic": pic}


class LCH3SaveToStory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The picture (the first one of a batch)."}),
                "kind": (SAVE_KINDS, {"default": "from card", "tooltip": "Which slot it goes in. from card: what the Card Prompt was making. cast card: the headshot. cast outfit: the cast card's 👗 outfit. Cast, outfit and prop pictures are cut out onto plain grey. shot start / end: the shot's 🖼 start or 🏁 end frame."}),
                "name": ("STRING", {"default": "", "tooltip": "Blank = the card from the Card Prompt. A cast, outfit, place or prop name with no card yet makes a new card. For shot start / end: the shot number."}),
                "project": ("STRING", {"default": "", "tooltip": "Blank = the project from the Card Prompt."}),
            },
            "optional": {
                "card": ("STRING", {"forceInput": True, "tooltip": "From LC H3 Story Card Prompt or LC H3 Shot Card."}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("Puts a picture straight onto a Story card: cast picture, outfit, room picture, 360 panorama, prop or a shot's start / end frame. "
                   "Cast, outfit and prop pictures are cut out onto plain grey. The Story node shows it at once. Older pictures are kept; "
                   "a new one never writes over them.")

    @classmethod
    def VALIDATE_INPUTS(cls, kind):
        return True if _kind(kind) in SAVE_KINDS else f"kind must be one of: {', '.join(SAVE_KINDS)}"

    def save(self, images, kind, name, project, card=None):
        c = {}
        if card:
            try:
                c = json.loads(card)
            except ValueError:
                c = {}
        kind = _kind(c.get("kind") if kind == "from card" else kind)
        if kind not in SAVE_KINDS[1:]:
            raise ValueError("[LC H3] Save to Story: pick a kind, or plug in LC H3 Story Card Prompt.")
        name = name.strip() or str(c.get("name") or "")
        project = project.strip() or str(c.get("project") or "")
        if not project:
            raise ValueError("[LC H3] Save to Story: no project. Plug in LC H3 Story Card Prompt, or type the project name.")
        arr = (images[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        patch = save_to_story(Image.fromarray(arr), project, kind, name)
        what = {"cast card": "picture", "cast outfit": "👗 outfit picture", "outfit": "outfit picture", "place": "room picture", "360 panorama": "🌐 360", "prop": "prop picture", "shot start": "🖼 start frame",
                "shot end": "🏁 end frame"}[kind]
        who = patch["card"].get("name") or f"shot {name}"
        msg = f"Saved as {who}'s {what}" + (" (new card)" if patch["new"] else "") + f": {patch['pic']['filename']}"
        print(f"[LC H3] {msg}")
        return {"ui": {"lc_story_card": [patch], "text": [msg],
                       "images": [{"filename": patch["pic"]["filename"], "subfolder": patch["pic"]["subfolder"], "type": "input"}]}}


# ------------------------------------------------------------------ LC H3 Shot Card
SLOTS = ["start", "end"]
COLORS_NOTE = "boxes are drawn on the Story card (📐)"


def _shot_list(story: dict) -> list:
    """[(number, shot, place)] with each shot's place worked out the way the plan does (blank = the shot before's)."""
    out, where_now = [], ""
    for n, shot in enumerate([x for x in story.get("shots", []) if isinstance(x, dict)], start=1):
        where_now = str(shot.get("where") or "").strip() or where_now or (story.get("places") or [{}])[0].get("name", "")
        out.append((n, shot, _place(story, where_now)))
    return out


def _wants(shot: dict, slot: str) -> bool:
    """A slot the board still needs: a cut's start frame (a keep-rolling shot starts from the frames before it), or
    an end frame (any shot)."""
    if slot == "start" and shot.get("join") == "roll":
        return False
    return not _pic_file(shot.get(slot))


def _pick_shot(story: dict, shot: str, slot: str):
    shots = _shot_list(story)
    key = str(shot or "").strip()
    if not key:
        hit = next((x for x in shots if _wants(x[1], slot)), None)
        if not hit:
            raise ValueError(f"[LC H3] Shot Card: every shot already has its {slot} frame. Press ↻ on a board picture to redo it, "
                             f"or type a shot number.")
        return hit
    hit = next((x for x in shots if str(x[1].get("id")) == key), None)
    if hit is None and key.isdigit() and 0 < int(key) <= len(shots):
        hit = shots[int(key) - 1]
    if hit is None:
        raise ValueError(f"[LC H3] Shot Card: no shot '{key}'. Use the shot number (1 = the first shot), or leave it blank.")
    return hit


def _present(story: dict, shot: dict) -> list:
    """Cast cards in this frame: named in the shot, in a box, or its subject. Nobody named and one cast = that one.
    Only these get their look (Kinburg: the whole cast in every frame grows strangers)."""
    if shot.get("subject") == "nobody":  # an empty shot: the place only
        return []
    names = _names(story)
    text = " ".join(str(shot.get(k) or "") for k in ("action", "speaker", "line", "after"))
    here = {n for n in names if re.search(rf"\b{re.escape(n)}\b", text, re.I)}
    here |= {str(b.get("name") or "").strip() for b in shot.get("boxes") or [] if str(b.get("name") or "").strip() in names}
    if shot.get("subject") in names:
        here.add(shot["subject"])
    if not here and len(names) == 1:
        here = set(names)
    return [c for c in story.get("cast", []) if str(c.get("name") or "").strip() in here]


def _where_words(b: dict) -> str:
    cx, cy, area = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["w"] * b["h"]
    side = "on the left" if cx < 0.36 else "on the right" if cx > 0.64 else "in the middle"
    depth = "in the foreground" if area > 0.22 or cy > 0.7 else "in the background" if area < 0.05 else ""
    return f"{side}{', ' + depth if depth else ''} of the frame"


def _moment(shot: dict, slot: str) -> str:
    """The frozen moment the still shows: the start of the action, or how the shot ends."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", str(shot.get("action") or "").strip()) if x.strip()]
    if slot == "end":
        t = str(shot.get("after") or "").strip() or (sents[-1] if sents else "")
    else:
        # the first beat of a chain of actions: cut where the next verb starts (", picks up", " and walks"); names and
        # descriptions between commas ("Nova, still seated at the table, ...") don't start with a verb, so they stay
        t = re.split(r"(?:,\s+|\s+and\s+|\s+then\s+)(?=[a-z]+s\b)", sents[0], maxsplit=1)[0] if sents else ""
    return t.rstrip(".,")


def _earlier_wide(shots: list, n: int, place: dict | None) -> tuple | None:
    """(number, shot) of the first earlier wide shot in the same place that has its start frame: the frame a later
    wide shot there is built from, so the room, the camera side and everything on the table stay put."""
    for m, sh, pl in shots[:n - 1]:
        if (pl or {}).get("name") == (place or {}).get("name") and (sh.get("framing") or "") not in TIGHT \
                and _pic_file(sh.get("start")) and os.path.exists(_pic_abs(sh.get("start"))):
            return m, sh
    return None


def _screen_side(shots: list, n: int, place: dict | None, name: str) -> str:
    """Where a person sat in the first earlier wide shot in this place that boxes them: "left" / "right" / ""."""
    for m, sh, pl in shots[:n - 1]:
        if (pl or {}).get("name") != (place or {}).get("name") or (sh.get("framing") or "") in TIGHT:
            continue
        for b in sh.get("boxes") or []:
            if str(b.get("name") or "").strip() == name and b.get("w"):
                cx = b["x"] + b["w"] / 2
                return "left" if cx < 0.45 else "right" if cx > 0.55 else ""
    return ""


# "holds one coffee cup" read the count as the object: counts and quantities are skipped like "the" (2026-11-01)
_HANDLED = re.compile(r"\b(?:picks? up|grabs?|takes?|lifts?|holds?|carries|puts? down|drops?)\s+"
                      r"(?:(?:her|his|their|the|a|an|its|one|two|three|each|some|another|both)\s+)*([a-z]+)", re.I)


def _unset_props(shots: list, n: int, shot: dict) -> list:
    """Things this shot's action handles ("picks up her bag") that no earlier shot's boxes show yet."""
    seen = " ".join(str(b.get("name") or "") for _, sh, _ in shots[:n - 1] for b in sh.get("boxes") or []).lower()
    return [w for w in dict.fromkeys(m.group(1).lower() for m in _HANDLED.finditer(str(shot.get("action") or "")))
            if w not in seen]


def _load_rgb(path: str, w: int | None = None, h: int | None = None, crop=None):
    img = Image.open(path).convert("RGB")
    if isinstance(crop, (list, tuple)) and len(crop) == 4:  # the card's ✂ crop (fractions x, y, w, h)
        try:
            x, y, cw, ch = (float(v) for v in crop)
            box = (round(x * img.width), round(y * img.height), round((x + cw) * img.width), round((y + ch) * img.height))
            if box[2] > box[0] and box[3] > box[1]:
                img = img.crop(box)
        except (TypeError, ValueError):
            pass  # a broken crop: the whole picture
    if w and h:  # centre crop to the frame's shape, then fit
        sw, sh = img.size
        r = max(w / sw, h / sh)
        img = img.resize((max(w, round(sw * r)), max(h, round(sh * r))), Image.LANCZOS)
        L, T = (img.width - w) // 2, (img.height - h) // 2
        img = img.crop((L, T, L + w, T + h))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]


def _load_fit(path: str, w: int, h: int, crop=None):
    """The whole picture fitted inside w x h on plain grey (an outfit keeps its head and feet)."""
    img = _load_rgb(path, crop=crop)[0].numpy()
    pic = Image.fromarray((img * 255).astype(np.uint8))
    pic.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (128, 128, 128))
    canvas.paste(pic, ((w - pic.width) // 2, (h - pic.height) // 2))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)[None]


def _worn_at(story: dict, n: int) -> dict:
    """{person: outfit} at shot n, as the plan has it: the cast card's own outfit from the start, then a shot that picks
    one (👗 dropdown) or names it puts it on from there."""
    robe = wardrobe(story)
    worn = {nm: items[0] for nm, items in robe.items() if items and items[0]["card"] is None}
    for k, shot in enumerate([x for x in story.get("shots", []) if isinstance(x, dict)], start=1):
        if k > n:
            break
        said = " ".join(str(shot.get(f) or "") for f in ("action", "after"))
        for nm, items in robe.items():
            for o in items:
                named = o["card"] and re.search(rf"\b{re.escape(o['card']['name'].strip())}\b", said, re.I)
                labelled = re.search(rf"\b{re.escape(o['label'])}\b", said, re.I) and re.search(rf"\b{re.escape(nm)}\b", said, re.I)
                if named or labelled:
                    worn[nm] = o
        for nm, oid in (shot.get("outfit") or {}).items():
            o = next((o for o in robe.get(nm, []) if o["id"] == oid), None)
            if o:
                worn[nm] = o
    return worn


def shot_card(story: dict, shot_key: str, slot: str, width: int, height: int, style: str = "") -> dict:
    story = _with_auto_views(story)
    n, shot, place = _pick_shot(story, shot_key, slot)
    shots = _shot_list(story)
    names = _names(story)
    cast = _present(story, shot)
    if (shot.get("framing") or "") in TIGHT and shot.get("subject") in names:
        # a close-up is one person: "Nova looks at Anika" put Anika on the cast sheet too, and the edit drew them side by side
        cast = [c for c in cast if c["name"].strip() == shot["subject"]] or cast
    worn = _worn_at(story, n)

    tight = (shot.get("framing") or "") in TIGHT

    def look(c):
        # a tight frame leaves out the clothes that can't be in it: named anyway, they were drawn on extra people in
        # the background (Coffee Date shot 3: two more women in Anika's skirt and boots)
        o = worn.get(c["name"].strip())
        if o and o["picture"]:  # the outfit picture is on the cast sheet: no clothing words (they beat the picture)
            own = strip_clothes(_one_clause(c.get("look")))
            bits = ([own] if own else []) + ["wearing the outfit shown in the outfit picture"]
        else:
            own = frame_look(_one_clause(c.get("look")), tight)
            bits = [own] if own else []
            if o:  # the outfit the story is up to: the start frame is copied exactly, so it has to wear it already
                worn_look = frame_look(o["notes"], tight)
                if worn_look:
                    bits.append("wearing " + worn_look)
        return f"{c['name'].strip()}, a {c.get('who') or 'person'}" + (", " + ", ".join(bits) if bits else "")
    cam, _focus = _framing(shot, names)
    desc = str((place or {}).get("description") or (place or {}).get("name") or "").strip().rstrip(".")
    boxes = [b for b in shot.get("boxes") or [] if isinstance(b, dict) and b.get("w") and b.get("h")]
    moment = _moment(shot, slot)
    held = _earlier_wide(shots, n, place) if slot == "start" and (shot.get("framing") or "") not in TIGHT else None
    if held:
        # props of the earlier frame that sit under a box that changes get redrawn too: carry them into the edit, in
        # place, or the box edit paints over them (Nova standing up took the handbag beside her with her)
        def hits(p, q):
            return p["x"] < q["x"] + q["w"] and q["x"] < p["x"] + p["w"] and p["y"] < q["y"] + q["h"] and q["y"] < p["y"] + p["h"]
        mine = {str(b.get("name") or "").strip() for b in boxes}
        carry = [dict(b) for b in held[1].get("boxes") or [] if isinstance(b, dict) and b.get("w") and b.get("h")
                 and str(b.get("name") or "").strip() not in names and str(b.get("name") or "").strip() not in mine
                 and any(hits(b, q) for q in boxes)]
        boxes = boxes + carry
    # a close-up is the reverse shot down the table: face to camera, eyes just past the lens on the wide shot's side
    # (the one sitting on the left looks just right of the lens). A turned head reads as a Dutch angle or a new subject.
    facing = ""
    if (shot.get("framing") or "") in TIGHT and len(cast) == 1:
        side = _screen_side(shots, n, place, cast[0]["name"].strip())
        if side:
            facing = (f"{cast[0]['name'].strip()} faces the camera, eyes just past the lens to the "
                      f"{'right' if side == 'left' else 'left'}, toward the person across the table")
    bible = " ".join(x for x in (str(story.get("look") or "").strip().rstrip("."), style.strip().rstrip(".")) if x)
    scene = _sentences(" ".join(x for x in (
        f"A still frame from a film: {cam}." if cam else "A still frame from a film.",
        f"In {desc}." if desc else "",
        f"{moment}." if moment else "",
        f"{facing}." if facing else "",
        f"{bible}." if bible else "", SUFFIX) if x))
    people = " ".join(f"{look(c)}." for c in cast)
    where = " ".join(f"{(b.get('name') or 'A prop').strip()} is {_where_words(b)}." for b in boxes)
    prompt = _sentences(" ".join(x for x in (scene.replace(SUFFIX, "").strip(), people, where, SUFFIX) if x))
    # elements in LC Bounding Boxes' format (0-1000 [y0, x0, y1, x1]) for LC MOSS Layout Prompt / LC Flux 3 Layout Prompt
    by_name = {c["name"].strip(): c for c in cast}
    els = []
    for k, b in enumerate(boxes, start=1):
        nm = str(b.get("name") or "").strip()
        c = by_name.get(nm) or next((x for x in story.get("cast", []) if str(x.get("name") or "").strip() == nm), None)
        y0, x0 = round(b["y"] * 1000), round(b["x"] * 1000)
        bb = [y0, x0, min(1000, round((b["y"] + b["h"]) * 1000)), min(1000, round((b["x"] + b["w"]) * 1000))]
        desc = look(c) if c else (nm or "a prop")
        if held and c and moment and re.search(rf"\b{re.escape(nm)}\b", moment, re.I):
            desc = f"{desc}, {moment[0].lower()}{moment[1:]}" if not moment.startswith(nm) else f"{desc}. {moment}"
        el = {"id": k, "type": "obj", "desc": desc, "bbox": bb, "kind": "change" if held and c else "new", "from": "target",
              "src_bbox": None, "tgt_bbox": bb, "canvas": [width, height]}
        if c:
            el["label"] = "character"
        els.append(el)
    # background: the start frame for an end frame (same room, same camera), else the shot's 360 view, else the room
    bg, bg_what = None, "none"
    if slot == "end" and _pic_file(shot.get("start")) and os.path.exists(_pic_abs(shot.get("start"))):
        bg, bg_what = _pic_abs(shot.get("start")), "start frame"
    elif held:
        bg, bg_what = _pic_abs(held[1].get("start")), f"shot {held[0]}'s frame (held: only the boxes change)"
    else:
        v = _shot_view(shot, place)
        if v:
            try:
                bg, bg_what = _view_path(story, shot, place, v), "360 view"
            except Exception as exc:
                print(f"[LC H3] Shot Card: no 360 view ({exc}).")
        if not bg and place and _pic_file(place.get("picture")) and os.path.exists(_pic_abs(place.get("picture"))):
            bg, bg_what = _pic_abs(place.get("picture")), "room picture"
    background = _load_rgb(bg, width, height) if bg else torch.zeros([1, height, width, 3])
    # left to right as the boxes stand, so a side-by-side sheet reads in the frame's order
    left = {str(b.get("name") or "").strip(): b["x"] + b["w"] / 2 for b in boxes}
    order = sorted(cast, key=lambda c: left.get(c["name"].strip(), 2.0))
    faces = []
    for c in order:  # each person's headshot, then the outfit they wear in this shot (whole, on grey, not cut to a square)
        if _pic_file(c.get("picture")) and os.path.exists(_pic_abs(c.get("picture"))):
            faces.append(_load_rgb(_pic_abs(c.get("picture")), 1024, 1024, (c.get("picture") or {}).get("crop")))
        o = worn.get(str(c.get("name") or "").strip())
        if o and o["picture"] and os.path.exists(_pic_abs(o["picture"])):
            faces.append(_load_fit(_pic_abs(o["picture"]), 1024, 1024, o["picture"].get("crop")))
    cast_img = torch.cat(faces) if faces else torch.zeros([1, 1024, 1024, 3])
    # one picture with everyone side by side; None (= not plugged) when nobody in the frame has a picture
    cast_sheet = torch.cat(faces, dim=2) if faces else None
    framing = shot.get("framing") or "wide shot"
    info = (f"Shot {n} {slot} frame ({framing}{', tight' if framing in TIGHT else ''}): {len(cast)} in frame "
            f"({', '.join(c['name'] for c in cast) or 'nobody'}), {len(boxes)} boxes, background = {bg_what}, "
            f"{len(faces)} cast pictures.")
    if cast and len(faces) < len(cast):
        info += " ⚠️ Missing a cast picture: render or import it on the Story node for a steady face."
    if not boxes:
        info += f" 💡 No blocking yet: {COLORS_NOTE}."
    if held and not boxes:
        info += " ⚠️ Held shot with no boxes: nothing would change. Box what changes (a person standing up, a new prop)."
    for w in _unset_props(shots, n, shot):
        info += f" ⚠️ The action handles '{w}' but no earlier shot shows it: box it in the first wide shot so it doesn't appear out of nowhere."
    if facing:
        info += f" Screen direction from the wide shot: {facing}."
    return {"n": n, "shot": shot, "prompt": prompt, "scene": scene, "where": where, "elements": els, "background": background,
            "cast": cast_img, "cast_sheet": cast_sheet, "info": info, "hold": bool(held)}


class LCH3ShotCard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"forceInput": True, "tooltip": "From LC H3 Story 🎬 (so the story is saved before it is read)."}),
                "vae": ("VAE", {"tooltip": "The image model's VAE: the empty latent is made to fit it."}),
                "shot": ("STRING", {"default": "", "tooltip": "Blank = the next shot still missing this frame (run again for the next one). "
                                                             "A number = that shot."}),
                "slot": (SLOTS, {"default": "start", "tooltip": "start: the shot's first picture (cuts). end: the picture the shot lands on."}),
                "width": ("INT", {"default": 1344, "min": 256, "max": 4096, "step": 16, "tooltip": "16:9 like the video. The boxes scale to it."}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 16}),
                # not named "seed": that name gets a control_after_generate widget, and with it ComfyUI's loader drops the
                # first saved value (it takes it for project_name's old widget slot) and every widget below shifts by one
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Goes out plus the shot number, so each shot gets its own."}),
            },
            "optional": {
                "style": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional. Added to the Story's look on every frame, word for word."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # the board decides which slot is next

    RETURN_TYPES = ("LATENT", "STRING", "STRING", "ARRAY", "STRING", "IMAGE", "IMAGE", "INT", "INT", "INT", "STRING", "BOOLEAN", "IMAGE",
                    "BOOLEAN", "STRING")
    RETURN_NAMES = ("latent", "prompt", "scene", "elements", "where", "background", "cast", "width", "height", "seed", "card", "hold",
                    "cast_sheet", "tight", "edit_prompt")
    OUTPUT_TOOLTIPS = ("An empty latent at the frame's size.",
                       "The whole prompt: scene, the people in it with their looks, where they stand. For models without boxes.",
                       "The scene only (no people): the base prompt for LC MOSS Layout Prompt, whose boxes carry the people.",
                       "The 📐 boxes for LC MOSS Layout Prompt / LC Flux 3 Layout Prompt.",
                       "The boxes as words (\"Nova is on the left of the frame.\").",
                       "The shot's 360 view or room picture (for an end frame: its start frame), sized to the frame. Black if none.",
                       "The pictures of the people in this frame (1024 x 1024), for reference / face nodes.",
                       "Width.", "Height.", "seed + shot number.",
                       "Which slot it is: plug into LC H3 Save to Story.",
                       "True when this shot is built on an earlier wide frame of the same place (background): send it through "
                       "a box edit (LC MOSS Box Edit) so only the boxes change. False: render it fresh from the prompt.",
                       "The cast pictures side by side, left to right as they stand in the frame: the face reference for an edit "
                       "pass (Krea 2 Identity Edit's 2nd reference). Empty when nobody in the frame has a picture, so the edit "
                       "keeps the frame's own faces.",
                       "True for a close-up or medium shot: its edit goes onto the empty background (beside), not the layout "
                       "frame, or the person in the layout comes back as a second copy.",
                       "The prompt for the Identity Edit pass: the whole prompt, plus faces from the cast sheet and clothes as written.")
    FUNCTION = "build"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("A storyboard job for any image model: one shot's start or end frame as a prompt, the blocking boxes, the background "
                   "and the cast pictures. Save the picture with LC H3 Save to Story and it lands in the shot's slot; leave shot blank "
                   "and each run takes the next empty slot.")

    def build(self, project_name, vae, shot, slot, width, height, base_seed, style=""):
        story = _read(project_name)
        w, h = width // 16 * 16, height // 16 * 16
        r = shot_card(story, shot, slot, w, h, style)
        ch = getattr(vae, "latent_channels", 16)
        down = vae.spacial_compression_encode() if hasattr(vae, "spacial_compression_encode") else 8
        latent = {"samples": torch.zeros([1, ch, h // down, w // down])}
        card = json.dumps({"project": project_name, "kind": f"shot {slot}", "name": str(r["n"])}, ensure_ascii=False)
        print(f"[LC H3] {r['info']}")
        return {"ui": {"text": [r["info"] + "\n\n" + r["prompt"]]},
                "result": (latent, r["prompt"], r["scene"], r["elements"], r["where"], r["background"], r["cast"], w, h,
                           (base_seed + r["n"]) % 0xFFFFFFFFFFFFFFFF, card, r["hold"], r["cast_sheet"],
                           (r["shot"].get("framing") or "") in TIGHT,
                           r["prompt"] + " The faces and hair are the ones in the second picture. Everyone's clothes are exactly as "
                           "written here.")}


NODE_CLASS_MAPPINGS = {"LCH3StoryCardPrompt": LCH3StoryCardPrompt, "LCH3SaveToStory": LCH3SaveToStory, "LCH3ShotCard": LCH3ShotCard}
NODE_DISPLAY_NAME_MAPPINGS = {"LCH3StoryCardPrompt": "LC H3 Story Card Prompt (BETA) 📝",
                              "LCH3SaveToStory": "LC H3 Save to Story (BETA) 🃏",
                              "LCH3ShotCard": "LC H3 Shot Card (BETA) 🎞️"}
