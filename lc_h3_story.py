"""
LC H3 Story (BETA, working name): the whole movie on one node. Cast (name, picture, look, voice), places (name,
picture, description, anchors) and one card per shot. One shot = one segment (a shot longer than max_shot_seconds is
made in parts). No AI rewrites anything: the text a shot builds (Place / Camera lines and timed beats) is exactly what
LC H3 Segment gets, and the node shows it on the card ("H3 gets").

Outputs the same plan and refs LC H3 Segment already reads, and writes <project>/story.json (this node's own data) and
<project>/plan.json (what Segment, the board and the chain read).

  GET  /lc_h3/story/load?project=     -> story.json, or an old Shot List + Reference Assembler project imported
  POST /lc_h3/story/save {story}      -> writes story.json + plan.json
  POST /lc_h3/story/preview {story}   -> {"shots": [{"segments": [...], "seconds", "warnings"}], "warnings"}
  GET  /lc_h3/story/images?where=input|output -> pictures to pick from (newest first)
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import time

import folder_paths

from .lc_h3_refs import REFS_TYPE, build as _build_refs, frame_look, inside_comfy, outfit_name, strip_clothes
from .lc_h3_segment import FPS, PLAN_TYPE, _is_done, _seg_path, project_dir, project_size, work_dir

NODE_ID = "LCH3Story"
NODE_DISPLAY_NAME = "LC H3 Story (BETA) 🎬"

WORDS_PER_SECOND = 2.5
OVERLAP_FRAMES = 39  # 1.62 s carried over from the shot before on a keep-rolling shot (the tested default)
MIN_SECONDS = 3.0
LINE_LEAD = 0.6  # seconds before a line starts
LINE_TAIL = 0.7  # seconds after it ends
# talking while doing something with the mouth makes H3 lose the lips (Anika "drinking coffee" while she talks: no mouth movement)
_MOUTH_RE = re.compile(r"\b(drink(s|ing)?|drank|sip(s|ping)?|eat(s|ing)?|ate|bit(e|es|ing)|chew(s|ing)?|kiss(es|ing)?|smok(e|es|ing)|"
                       r"yawn(s|ing)?|lick(s|ing)?|blow(s|ing)?)\b", re.I)
DEFAULT_SECONDS = 5.0

FRAMINGS = ["wide shot", "side view", "two-shot", "medium shot", "close-up", "extreme close-up", "over the shoulder",
            "from behind", "custom"]
GROUP_FRAMINGS = {"wide shot", "side view", "two-shot"}
MOVES = {
    "still": "",
    "slow push-in": "Slow push-in on {who}.",
    "slow pull back": "Slow pull back from {who}.",
    "pan": "Slow pan with {who}.",
    "tracking": "Slow tracking shot alongside {who}.",
    "orbit": "Slow orbit around {who}.",
    "handheld": "Handheld camera, slight natural movement.",
}
WHO = ["woman", "man", "girl", "boy", "person", "cat", "dog"]
# how tight each framing is: a keep-rolling shot pushes in or pulls back by this
SCALE = {"wide shot": 0, "side view": 1, "two-shot": 1, "from behind": 1, "over the shoulder": 2, "medium shot": 2,
         "close-up": 3, "extreme close-up": 4, "custom": 1}
# framings where a room picture pins H3 to the wide camera position: these get the Place words only
TIGHT = {"medium shot", "close-up", "extreme close-up", "over the shoulder"}

_AT_RE = re.compile(r"\bAt (\d+(?:\.\d+)?) seconds\b", re.I)
_SPEECH_IN_SOUND = re.compile(r"\b(?:speak\w*|talk\w*|voices?|says?|saying|spoken|dialogue|nobody|no one|quoted)\b", re.I)
_QUOTE_RE = re.compile(r"[\"“](.+?)[\"”]", re.DOTALL)

EXAMPLE = {
    "version": 1,
    "project": "parka_test",
    "look": "Cinematic, soft natural light, shallow depth of field.",
    "cast": [{"id": "c1", "name": "Lonecat", "who": "woman",
              "look": "in her twenties with long blonde hair and a parka with a big fur-lined hood", "picture": None, "voice": None}],
    "places": [{"id": "p1", "name": "the cafe", "description": "a small cafe with a big rainy window, warm lamps and wooden tables",
                "sound": "rain on the window, soft cafe chatter", "picture": None, "anchors": []}],
    "shots": [{"id": "s1", "framing": "medium shot", "subject": "Lonecat", "move": "still", "join": "cut", "where": "the cafe",
               "action": "Lonecat sits by the window and watches the raindrops run down the glass.",
               "speaker": "Lonecat", "line": "It never stops raining here.", "after": "She smiles and sips her coffee.", "seconds": ""}],
}


# ------------------------------------------------------------------ small helpers
def snap_frames(seconds: float, minimum: int = 22) -> int:
    """H3 frame counts sit on a 17k + 5 grid."""
    k = max(0, round((seconds * FPS - 5) / 17))
    return max(minimum, 17 * k + 5)


def _words(text) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", str(text or "")))


def _sentence(text) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return ""
    t = t[:1].upper() + t[1:]
    return t if t[-1] in ".!?\"”" else t + "."


def _beat(text: str) -> str:
    """A sentence placed after "At 1.20 seconds, ": lower-case its first word unless it is a name."""
    t = _sentence(text)
    if t.split(" ", 1)[0] in ("He", "She", "They", "The", "A", "An", "His", "Her", "Their", "It", "Both"):
        t = t[:1].lower() + t[1:]
    return t


def _one_clause(text) -> str:
    """A look as one clause: sentence breaks and semicolons inside it become commas."""
    t = re.sub(r"\s*[.;!]+\s+(?=\S)", ", ", str(text or "").strip())
    return re.sub(r"\s+", " ", t).strip(" .;,")


def _clean_line(line) -> str:
    return re.sub(r"\s+", " ", str(line or "")).strip().strip('"“”').strip()


def _line_seconds(line) -> float:
    n = _words(line)
    return n / WORDS_PER_SECOND if n else 0.0


VO = " (V.O.)"  # the speaker dropdown's voiceover pick: "Parka (V.O.)"
# the subject dropdown's empty pick: a shot with no people in it (an establishing shot of the place, a ship in space).
# No cast line, no faces, camera words say nobody is there; only a voiceover can speak (2026-11-07, "Uber run" shot 1)
NOBODY = "nobody"
# body words in a cast look (the look should stop at face and hair when an outfit picture shows the rest)
_BODY_RE = re.compile(r"\b(breasts?|boobs?|chest|cleavage|curv(?:y|es)|figure|body|hips|waist|butt|ass|legs|thighs|abs|"
                      r"busty|petite|athletic|muscular|slim|thin|skinny|chubby|toned|nipples?|naked|nude|topless)\b", re.I)


def _who_says(speaker) -> tuple:
    """(cast name, voiceover?) from the speaker field."""
    s = str(speaker or "").strip()
    return (s[:-len(VO)].strip(), True) if s.endswith(VO) else (s, False)


def _names(story: dict) -> list:
    return [str(c.get("name") or "").strip() for c in story.get("cast", []) if str(c.get("name") or "").strip()]


def _place(story: dict, name) -> dict | None:
    key = str(name or "").strip().lower()
    return next((p for p in story.get("places", []) if str(p.get("name") or "").strip().lower() == key), None) if key else None


def _pic_file(pic) -> str:
    """A picture slot as the path the Reference Assembler reads: input-relative, or absolute for output pictures."""
    if not isinstance(pic, dict):
        return ""
    sub, name = str(pic.get("subfolder") or "").strip("/\\"), str(pic.get("filename") or "")
    if not name:
        return ""
    rel = f"{sub}/{name}" if sub else name
    base = folder_paths.get_output_directory() if pic.get("type") == "output" else folder_paths.get_input_directory()
    if os.path.isabs(name) or not inside_comfy(os.path.join(base, rel)):  # never a file outside ComfyUI's folders
        return ""
    return os.path.join(base, rel) if pic.get("type") == "output" else rel


# ------------------------------------------------------------------ 360 views
VIEW_W, VIEW_H = 1280, 720  # the cut view (16:9, like the Segment's 960 x 544)


def _pic_abs(pic) -> str:
    f = _pic_file(pic)
    return f if not f or os.path.isabs(f) else os.path.join(folder_paths.get_input_directory(), f)


def _shot_view(shot: dict, place: dict | None) -> dict | None:
    """The shot's own camera angle in its place's 360 panorama: {yaw, pitch, fov, behind}, or None. The card's yaw
    counts from the place's `front` (where the wide shot faces in the panorama): 90 = the right side of the room."""
    v = shot.get("view")
    if not (isinstance(v, dict) and v.get("on") and place and _pic_file(place.get("panorama"))):
        return None
    def num(k, d, lo, hi):
        try:
            x = float(v.get(k) if v.get(k) not in (None, "") else d)
        except (TypeError, ValueError):  # text typed in a number box: the default
            x = float(d)
        return max(lo, min(hi, x))

    try:
        front = float((place.get("front") or {}).get("yaw") or 0)
    except (TypeError, ValueError):
        front = 0.0
    yaw = (num("yaw", 0, -360, 360) + front + 180) % 360 - 180
    return {"yaw": round(yaw, 1), "pitch": round(num("pitch", -5, -85, 85), 1),
            "fov": round(num("fov", 60, 10, 150), 1), "behind": str(v.get("behind") or "").strip()}


def cut_view(pano_path: str, yaw: float, pitch: float, fov: float, w: int = VIEW_W, h: int = VIEW_H):
    """A flat view out of an equirectangular panorama (same projection as LC 360 View in MaskMaker)."""
    import numpy as np
    from PIL import Image

    src = np.asarray(Image.open(pano_path).convert("RGB"), dtype=np.float32)
    H, W = src.shape[:2]
    f = (w / 2) / math.tan(math.radians(fov) / 2)
    x, y = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    d = np.stack([x, -y, np.full_like(x, f)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p, t = math.radians(pitch), math.radians(yaw)
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])  # + pitch looks up
    ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    d = d @ (ry @ rx).T
    u = (np.arctan2(d[..., 0], d[..., 2]) / (2 * np.pi) + 0.5) * W - 0.5
    v = (0.5 - np.arcsin(np.clip(d[..., 1], -1, 1)) / np.pi) * H - 0.5
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    v0c, v1c = np.clip(v0, 0, H - 1), np.clip(v0 + 1, 0, H - 1)
    u0w, u1w = u0 % W, (u0 + 1) % W  # wraps around the seam
    out = (src[v0c, u0w] * (1 - fu) * (1 - fv) + src[v0c, u1w] * fu * (1 - fv)
           + src[v1c, u0w] * (1 - fu) * fv + src[v1c, u1w] * fu * fv)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def _view_path(story: dict, shot: dict, place: dict, v: dict) -> str:
    """The shot's view as a file in <project>/views/, cut once per angle."""
    pano = _pic_abs(place.get("panorama"))
    folder = os.path.join(project_dir(str(story.get("project") or "untitled").strip() or "untitled"), "views")
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(shot.get("id") or "shot"))
    out = os.path.join(folder, f"{sid}_{v['yaw']:g}_{v['pitch']:g}_{v['fov']:g}.png")
    if not os.path.exists(out) or (os.path.exists(pano) and os.path.getmtime(pano) > os.path.getmtime(out)):
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):  # the shot's earlier angles
            if old.startswith(sid + "_"):
                try:
                    os.remove(os.path.join(folder, old))
                except OSError:
                    pass
        cut_view(pano, v["yaw"], v["pitch"], v["fov"]).save(out)
    return out


def _with_auto_views(story: dict) -> dict:
    """Every cut gets a background: a close cut with no 🌐 angle of its own, in a place with a 360 panorama, gets one.
    Across from a person who already has an angle (two people at a table) it is the reverse angle (+180), else the way
    the wide shot faces. Coffee Date 2026-10-29: Nova's close-up had a background and held the café; Anika's had none
    and H3 put her in a room it made up."""
    shots = [s for s in story.get("shots", []) if isinstance(s, dict)]
    if not shots:
        return story
    story = {**story, "shots": [dict(s) for s in story.get("shots", [])]}
    shots = [s for s in story["shots"] if isinstance(s, dict)]
    where_now = ""
    wheres = []
    for s in shots:
        where_now = str(s.get("where") or "").strip() or where_now or (story.get("places") or [{}])[0].get("name", "")
        wheres.append(where_now)
    for i, s in enumerate(shots):
        place = _place(story, wheres[i])
        if (s.get("framing") or "wide shot") not in TIGHT or _shot_view(s, place) or not (place and _pic_file(place.get("panorama"))):
            continue
        if i > 0 and s.get("join") == "roll" and _same_camera(shots[i - 1], s):
            continue  # a roll carries on from the frames before it
        other = next((o for k, o in enumerate(shots) if wheres[k] == wheres[i] and (o.get("subject") or "everyone") != (s.get("subject") or "everyone")
                      and isinstance(o.get("view"), dict) and o["view"].get("on")), None)
        if other:
            v = other["view"]
            try:
                yaw0 = float(v.get("yaw") or 0)
            except (TypeError, ValueError):
                yaw0 = 0.0
            s["view"] = {"yaw": (yaw0 + 360) % 360 - 180, "pitch": v.get("pitch", -1), "fov": v.get("fov", 55),
                         "on": True, "behind": "", "auto": True}
        else:
            s["view"] = {"yaw": 0, "pitch": -1, "fov": 55, "on": True, "behind": "", "auto": True}
    return story


ROLL_LOCKED = ("framing", "where", "view", "start", "boxes", "outfit", "loras")  # greyed out on a keep-rolling card (who it is on stays open: people can walk in)


def inherit_rolls(story: dict) -> dict:
    """Keep rolling carries the camera straight on from the shot before: the framing, the place, the 🌐 angle and the
    clothes come from the cut that started the run (who it is on stays the card's own: someone can walk in), and there is no start frame or blocking of its own
    (the opening frames are the shot before's last 1.6 s). Only the action, the line, the camera move, the length and
    the end frame are the card's own. Before this, a roll into a new framing was quietly made as a cut, and the story
    writer kept writing them ("Uber run" shot 2: wide -> medium + tracking, 2026-11-07)."""
    out, src = [], None
    for k, s in enumerate(story.get("shots") or []):
        if isinstance(s, dict) and k and s.get("join") == "roll" and src is not None:
            s = {**s, "framing": src.get("framing"), "view": src.get("view"),
                 "where": "", "start": None, "boxes": [], "outfit": {},  # where "" = the place of the shot before
                 "loras": src.get("loras") or []}  # the cut's LoRAs: a new LoRA mid-roll would jump at the join
        elif isinstance(s, dict):
            src = s
        out.append(s)
    return {**story, "shots": out}


def shot_loras(shot: dict) -> list:
    """The shot's own LoRAs ([{lora, strength}]), on top of the base LoRAs the pipe's model already has. Only for that
    shot: the next cut starts from the base model again. Any LoRA (style, motion, character, adult)."""
    out = []
    for x in (shot.get("loras") if isinstance(shot, dict) else None) or []:
        name = str((x or {}).get("lora") or "").strip()
        try:
            st = float((x or {}).get("strength", 1.0))
        except (TypeError, ValueError):
            st = 1.0
        if name and st:
            out.append({"lora": name, "strength": round(st, 3)})
    return out


def cut_views(story: dict) -> list:
    """[(shot, place, view, file)] for every shot with a 🌐 angle; the files land in <project>/views/ (ready for any
    image workflow that makes the shot's start frame)."""
    story = _with_auto_views(inherit_rolls(story))
    out, where_now = [], ""
    for shot in story.get("shots", []):
        if not isinstance(shot, dict):
            continue
        where_now = str(shot.get("where") or "").strip() or where_now or (story.get("places") or [{}])[0].get("name", "")
        place = _place(story, where_now)
        v = _shot_view(shot, place)
        if not v:
            continue
        try:
            out.append((shot, place, v, _view_path(story, shot, place, v)))
        except Exception as exc:
            print(f"[LC H3] Story: could not cut the 360 view for shot {shot.get('id')} ({exc}).")
    return out


def _view_name(place: dict, shot: dict) -> str:
    return f"{str(place.get('name') or 'place').strip()} view {shot.get('id')}"


# ------------------------------------------------------------------ one shot -> its text
def _framing(shot: dict, names: list) -> tuple:
    """(camera words, who the shot is on): 'a close-up of Nova's face and shoulders, ...'."""
    kind = shot.get("framing") or "wide shot"
    if shot.get("subject") == NOBODY:
        if kind == "custom":
            return "", "the scene"
        size = {"two-shot": "wide shot", "over the shoulder": "wide shot", "from behind": "wide shot"}.get(kind, kind)
        return f"{'an' if size[:1] in 'aeiou' else 'a'} {size} of the scene, with no people in it", "the scene"
    who = shot.get("subject") if shot.get("subject") in names else "everyone"
    if who == "everyone" and len(names) == 1:
        who = names[0]
    text = " ".join(str(shot.get(k) or "") for k in ("action", "speaker", "after"))
    here = [n for n in names if re.search(rf"\b{re.escape(n)}\b", text, re.I)] or names
    group = "both of them" if len(here) == 2 else "all of them" if len(here) > 2 else (here[0] if here else "the scene")
    if who == "everyone":
        subj, focus = group, group
    else:
        subj, focus = who, who
    other = next((n for n in names if n != who), "the other person")
    words = {
        "wide shot": f"a wide shot of {subj}",
        "side view": f"a side view of {subj}",
        "two-shot": f"a two-shot of {subj}" + ("" if who == "everyone" else f" and {other}") + ", framed from the waist up",
        "medium shot": f"a medium shot of {subj}, framed from the waist up",
        "close-up": f"a close-up of {subj}'s face and shoulders, the face in the middle of the frame",  # "from the chest up" framed her chest (Coffee Date)
        "extreme close-up": f"an extreme close-up of {subj}'s face",
        "over the shoulder": f"an over-the-shoulder shot from behind {other}, looking at {subj}",
        "from behind": f"a shot from behind {subj}",
        "custom": "",
    }.get(kind, f"a wide shot of {subj}")
    return words, focus


def _auto_beats(shot: dict, span: float, names: list) -> list:
    """(time, text) beats from the plain fields: the action spread over the time before the line, the line, then
    what happens after it."""
    action = []
    for s in re.split(r"(?<=[.!?])\s+", str(shot.get("action") or "").strip()):
        if not s.strip():
            continue
        if action and s[:1].islower():  # "Anika, still seated. leans forward": one beat, or "leans forward" had no one doing it
            action[-1] = action[-1].rstrip(".!? ") + ", " + s
        else:
            action.append(s)
    line = _clean_line(shot.get("line"))
    after = str(shot.get("after") or "").strip()
    speaker, vo = _who_says(shot.get("speaker"))
    speaker = speaker or (names[0] if len(names) == 1 else "")
    say_s = _line_seconds(line)
    beats = []
    if line and not vo:  # a voiceover comes from off screen: the lips stay closed, so a drink can go on
        # mouth actions (drinking, eating ...) and a line in one shot: H3 had her drink right through the line even with
        # the drinking moved after it (Coffee Date take 4), so the drinks stay down for the whole shot
        mouth = [a for a in action if _MOUTH_RE.search(a)]
        if mouth:
            # an action H3 can do, not a "don't": "nobody drinks" still had her drinking (the place anchor shows a raised cup)
            who = [n for n in names if any(re.search(rf"\b{re.escape(n)}\b", a, re.I) for a in mouth)] or ([speaker] if speaker else [])
            who_s = " and ".join(who) if who else "Everyone"
            food = re.search(r"\b(eat|ate|bite|chew)", " ".join(mouth), re.I)
            action = [f"{who_s} {'put' if len(who) != 1 else 'puts'} {'the food' if food else ('their cups' if len(who) != 1 else 'the cup')} "
                      "down on the table and rest their hands."] + [a for a in action if a not in mouth]
    t_line = min(LINE_LEAD, span * 0.25) if action else 0.3
    if not line:
        t_line = span  # no line: the action fills the shot
    if action:
        step = max(0.6, (t_line if line else span - 0.6) / len(action))
        for k, a in enumerate(action):
            beats.append((round(k * step, 2), _beat(a)))
    if line:
        t_line = max(t_line, beats[-1][0] + 0.6) if beats else t_line
        # MiniMax's guide: "says in an off-screen voiceover", then the lips stay closed (Segment adds that sentence)
        beats.append((round(t_line, 2), f"{speaker or 'Someone'} says{' in an off-screen voiceover' if vo else ''}: \"{line}\""))
    if after:
        t_after = (t_line + say_s + 0.3) if line else (beats[-1][0] + 0.8 if beats else 0.0)
        beats.append((round(min(t_after, max(0.0, span - 0.4)), 2), _beat(after) + (" Nobody says anything more." if line else "")))
    elif line and t_line + say_s + 0.3 < span - 0.3:
        # time left after the line: H3 filled it by saying the line again (Coffee Date 2026-10-29, Nova)
        beats.append((round(t_line + say_s + 0.3, 2), f"{'The voiceover ends' if vo else (speaker or 'They') + ' stops talking'}. Nobody says anything more; the moment holds in silence."))
    if not beats:
        beats.append((0.0, "The moment holds: small natural movement only."))
    return beats


def _typed_beats(text: str) -> list:
    """(time, text) beats from an action written as "At 0.00 seconds, ... At 2.10 seconds, ..."."""
    marks = list(_AT_RE.finditer(text))
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        body = text[m.end():end].strip().lstrip(",").strip()
        if body:
            out.append((float(m.group(1)), body))
    return out


def _shot_seconds(shot: dict, max_s: float) -> tuple:
    """(seconds, warnings) for one shot: the number you typed wins; auto fits the line and the action."""
    warn = []
    line_need = _line_seconds(_clean_line(shot.get("line"))) + 1.2 if _clean_line(shot.get("line")) else 0.0
    typed = _typed_beats(str(shot.get("action") or ""))
    raw = str(shot.get("seconds") or "").strip().lower().rstrip("s").strip()
    if raw and raw != "auto":
        try:
            sec = float(raw)
        except ValueError:
            warn.append(f"⚠️ \"{shot.get('seconds')}\" is not a length, using auto.")
            sec = None
    else:
        sec = None
    if sec is None:
        acts = _words(shot.get("action")) + _words(shot.get("after"))
        if line_need:
            # a talking shot is as long as the line plus a breath either side: the old padding (1.2 s before, 1.2 s
            # after and more per word of action) left long silences between lines (Coffee Date: 34 s for a 25 s scene)
            say = _line_seconds(_clean_line(shot.get("line")))
            sec = max(MIN_SECONDS, LINE_LEAD + say + LINE_TAIL + (1.2 if str(shot.get("after") or "").strip() else 0.0))
        else:
            sec = max(DEFAULT_SECONDS, 0.08 * acts + 1.2)
        if typed:
            sec = max(sec, max(t for t, _ in typed) + 1.0)
        sec = min(sec, max_s)
    else:
        if sec < MIN_SECONDS:
            warn.append(f"⚠️ {sec:g} s is too short for H3, made at {MIN_SECONDS:g} s.")
            sec = MIN_SECONDS
        if line_need and sec < line_need:
            warn.append(f"⚠️ The line needs about {line_need:.1f} s, so the shot is stretched from {sec:g} s to fit it.")
            sec = line_need
    return round(sec, 2), warn


def _transition(prev: dict, shot: dict, names: list) -> tuple:
    """A keep-rolling shot that changes framing, with the card's move on still: a slow move spelled out from the old
    framing to the new one (a Camera line that only names the new framing left it on the wide shot: Coffee Date; "large
    amplitude" made it lunge, with exposure flashes). Returns (sentence, warning)."""
    a, fa = _framing(prev, names)
    b, fb = _framing(shot, names)
    if not a or not b or a == b or _same_camera(prev, shot):
        return "", ""
    sa, sb = SCALE.get(prev.get("framing") or "wide shot", 0), SCALE.get(shot.get("framing") or "wide shot", 0)
    if sb > sa or (sb == sa and fa not in names and fb in names):  # both of them -> one of them is tighter too
        verb = "Slow push-in"
    elif sb < sa:
        verb = "Slow pull back"
    else:
        verb = "Slow camera move"
    return f"{verb} from {a} to {b}.", KEEP_ROLLING_WARN


# Coffee Date 2026-10-28: every keep-rolling shot that changed framing broke (H3 cut by itself, into a room it made up)
KEEP_ROLLING_WARN = ("⚠️ Keeps rolling into a different framing: H3 usually makes its own cut here, into a room it invents. "
                     "A keep-rolling shot gets no pictures: no faces, no room picture and no anchor. Make this a ✂ cut "
                     "(it gets the faces and the room, plus the anchor when two or more people are in it), or keep the framing of the "
                     "shot before.")


_SIT_RE = re.compile(r"\b(sit|sits|sitting|sat|seated)\b", re.I)
_UP_RE = re.compile(r"\b(stand|stands|standing|stood|gets? up|got up|walks?|walking|rises?|leaves?)\b", re.I)


def _seated(shots: list, shot: dict, where: str, story: dict) -> bool:
    """A close cut on one person who sits somewhere in this place's shots, and doesn't stand up in this one: a close-up
    gets no room picture, so without the word H3 stood her up (Coffee Date: Anika, take 5)."""
    who = shot.get("subject")
    if (shot.get("framing") or "wide shot") not in TIGHT or who not in _names(story):  # NOBODY is never a cast name
        return False
    if _UP_RE.search(str(shot.get("action") or "")):
        return False
    here = [s for s in shots if not str(s.get("where") or "").strip() or str(s.get("where")).strip() == where]
    text = " ".join(str(s.get(k) or "") for s in here for k in ("action", "after"))
    return any(re.search(rf"\b{re.escape(who)}\b", sent, re.I) and _SIT_RE.search(sent) for sent in re.split(r"(?<=[.!?])\s+", text))


def _same_camera(prev: dict, shot: dict) -> bool:
    """A keep-rolling shot only rolls when the camera really carries on: the same framing, and on a tight framing the
    same person too. Anything else is made as a cut (Coffee Date 2026-10-28: every roll into a new framing made H3 cut
    by itself into a room it invented, right after the join)."""
    if (prev.get("framing") or "wide shot") != (shot.get("framing") or "wide shot"):
        return False
    if str(prev.get("where") or "").strip() and str(shot.get("where") or "").strip() and prev.get("where") != shot.get("where"):
        return False
    # who it is on may change (someone walks in): a keep-rolling card locks the framing and place, the people are yours
    return True


def _shot_text(story: dict, shot: dict, place: dict | None, span: float, offset: float, names: list,
               frame_shot: dict | None = None, prev: dict | None = None) -> tuple:
    """(action text for Segment, warnings) for one segment of a shot. span = the new seconds this segment adds,
    offset = where its own clock starts (the carried-over overlap on a keep-rolling segment), prev = the shot before
    when this one keeps rolling into it."""
    warn = []
    lines = []
    if place:
        desc = str(place.get("description") or "").strip()
        nm = str(place.get("name") or "").strip()
        lines.append(f"Place: {_sentence(desc or nm)[:-1] if (desc or nm) else ''}".rstrip() + ".")
    cam, focus = _framing(frame_shot or shot, names)  # a part of a long shot keeps the whole shot's framing
    if cam and (frame_shot or shot).get("framing") == "over the shoulder":
        cam += _behind_outfit(story, frame_shot or shot, names)
    if cam and shot.get("_seated"):
        cam += f", {focus} seated at the table"
    move = MOVES.get(shot.get("move") or "still", "").format(who=focus)
    # one camera move per shot: the card's own move wins; on still, a keep-rolling framing change is spelled out
    trans, tw = _transition(prev, frame_shot or shot, names) if prev else ("", "")
    if move:
        trans = ""
    if tw:
        warn.append(tw)
    if prev and not move and _same_camera(prev, frame_shot or shot):
        # a roll carries the camera on: naming a person here read as a push-in onto them
        lines.append(f"Camera: The camera holds the same {(frame_shot or shot).get('framing') or 'wide shot'} as the moment before, "
                     "with no camera move.")
    elif trans:
        lines.append("Camera: " + trans)
    elif cam or move:
        view = _shot_view(frame_shot or shot, place)
        if cam and view and view["behind"]:
            cam += f", with {_one_clause(view['behind'])} behind {focus}"
        elif cam and place and place.get("close_view") and str(place.get("description") or "").strip():
            # a close view's background said in words too: H3 follows words over pictures
            cam += f", with {_one_clause(place['description'])} behind {focus}"
        lines.append(("Camera: " + (cam[:1].upper() + cam[1:] + "." if cam else "") + (" " + move if move else "")).strip())
    typed = _typed_beats(str(shot.get("action") or ""))
    beats = typed or _auto_beats(shot, span, names)
    if typed:
        last = max(t for t, _ in typed)
        if last > span - 0.3 and last > 0:
            warn.append(f"⚠️ The beats run to {last:.2f} s but the shot is {span:.2f} s: squeezed to fit.")
            k = (span - 0.3) / last
            beats = [(round(t * k, 2), b) for t, b in typed]
    body = " ".join(f"At {t + offset:.2f} seconds, {b}" for t, b in beats)
    has_line = bool(_QUOTE_RE.search(body))
    # Segment moves the Sound line into H3's overall_soundscape section
    ambience = str((place or {}).get("sound") or "").strip()
    # 2026-11-03 A/B, same seed: "nobody speaks except the quoted lines" in a place's sound made H3 talk in a silent shot (French,
    # then English); without it, silence. The H3 prompt box on the card shows what is sent, and can be edited.
    kept = [c for c in re.split(r"\s*,\s*", ambience) if c and not _SPEECH_IN_SOUND.search(c)]
    ambience = _sentence(", ".join(kept) or "quiet natural ambience")
    sound = f"Sound: {ambience} " + ("The only voice is the quoted line." if has_line else "Nobody speaks.")
    head = "\n".join(lines)
    return (f"{head}\n\n{body}\n{sound}" if head else f"{body}\n{sound}"), warn


def _split_beats(beats: list, total: float, part_s: float, ovl_s: float) -> list:
    """A shot longer than max_shot_seconds: [(start, end, beats)] parts; a quoted line is never cut in two (its beat
    moves to the part it starts in, and that part ends after it)."""
    parts, start = [], 0.0
    while start < total - 0.05:
        end = min(total, start + (part_s if not parts else part_s - ovl_s))
        parts.append([start, end, []])
        start = end
    for t, b in beats:
        part = next((p for p in parts if p[0] <= t < p[1]), parts[-1])
        part[2].append((round(t - part[0], 2), b))
    return parts


def _sig(story: dict, seg: dict, shot: dict, place: dict | None) -> str:
    """What else goes into a take besides its prompt: the pictures and voices of the people it names, its 360 angle
    and its start / end frames. A take made from another signature is "changed since made"."""
    text = seg.get("action", "")
    bits = []
    for c in story.get("cast", []):
        nm = str(c.get("name") or "").strip()
        if nm and re.search(rf"\b{re.escape(nm)}\b", text, re.I):
            crop = (c.get("picture") or {}).get("crop")
            bits.append(f"{nm}:{_pic_file(c.get('picture'))}:{_pic_file(c.get('voice'))}" + (f":crop{crop}" if crop else ""))
    v = _shot_view(shot, place)
    if v:
        bits.append(f"view:{_pic_file(place.get('panorama'))}:{v['yaw']}:{v['pitch']}:{v['fov']}")
    if seg.get("part", 1) == 1 and _pic_file(shot.get("start")):
        bits.append(f"start:{_pic_file(shot.get('start'))}")
    if seg.get("part", 1) == seg.get("parts", 1) and _pic_file(shot.get("end")):
        bits.append(f"end:{_pic_file(shot.get('end'))}")
    return "|".join(bits)


def changed_since_made(folder: str, disk: dict, now: dict | None = None) -> set:
    """Segment numbers (of the saved plan) whose take was made from other text or pictures than the cards have now
    (`now`: a fresh plan of the cards on screen; else the saved plan itself). A take made after the plan was saved
    was made from that plan."""
    cur = {(s.get("shot_id"), s.get("part")): s for s in (now or disk).get("shots", [])}
    try:
        plan_t = os.path.getmtime(os.path.join(folder, "plan.json"))
    except OSError:
        plan_t = 0
    out = set()
    size = project_size(folder)
    for k, s in enumerate(disk.get("shots", []), start=1):
        if not _is_done(folder, k):
            continue
        # made at another size than the project is set to now (LC H3 Make writes the pipe's size before it looks):
        # a mixed-size movie distorts the joins and breaks the anchors, so it is made again
        meta = _take_meta(folder, k)
        if size and meta.get("width") and (int(meta["width"]), int(meta.get("height") or 0)) != size:
            out.add(k)
            continue
        made = s.get("made")
        if os.path.getmtime(_seg_path(folder, k, ".json")) >= plan_t:
            made = {"prompt": s.get("prompt"), "sig": s.get("sig")}  # made again since: the snapshot is old
        if not made:  # made before snapshots existed: the take's own record of the text it got
            meta = _take_meta(folder, k)
            if "action" in meta:
                made = {"action": meta["action"], "start_frame": meta.get("start_frame") or None}
        c = cur.get((s.get("shot_id"), s.get("part")))
        # a new length is a change too: the same text in a shorter take cuts the line off (Coffee Date shot 1: typed
        # 7 s, back on auto = 9.4 s, the 7.3 s take was never flagged)
        took = _take_meta(folder, k).get("frames")
        if c and took and c.get("frames") and int(took) != int(c["frames"]):
            out.add(k)
            continue
        if not (made and c):
            continue
        if "action" in made:
            differs = made["action"] != c.get("action") or made["start_frame"] != (c.get("start_frame") or None)
        else:
            differs = made.get("prompt") != c.get("prompt") or (made.get("sig") is not None and made["sig"] != c.get("sig", ""))
        if differs:
            out.add(k)
    # a cut back to an earlier setup opens on that shot's first frame: when that shot was made again since, so is this
    for k, s in enumerate((now or disk).get("shots", []), start=1):
        rt = int(s.get("return_to") or 0)
        if rt and _is_done(folder, k) and _is_done(folder, rt) and \
                os.path.getmtime(_seg_path(folder, rt, ".mp4")) > os.path.getmtime(_seg_path(folder, k, ".mp4")):
            out.add(k)
    return out


def _take_meta(folder: str, k: int) -> dict:
    try:
        with open(_seg_path(folder, k, ".json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ------------------------------------------------------------------ the whole story -> plan + refs
def _one_person(shot: dict, names: list) -> dict:
    """A tight shot left on "everyone" whose text names one person is on that person: the seated words, her own
    pictures and the background all hang on the subject (Coffee Date shot 3: Anika's close-up on "everyone" lost
    "seated at the table" and stood up, the Nova cut set to Nova never did)."""
    if (shot.get("framing") or "wide shot") not in TIGHT or shot.get("subject") in names or shot.get("subject") == NOBODY:
        return shot
    text = " ".join(str(shot.get(k) or "") for k in ("action", "speaker", "after"))
    here = [n for n in names if re.search(rf"\b{re.escape(n)}\b", text, re.I)]
    return {**shot, "subject": here[0]} if len(here) == 1 else shot


_UNDRESS = re.compile(r"\b(naked|nude|topless|bottomless|bare[- ]?(chested|breasted|skinned)?|undressed|unclothed|"
                      r"wearing nothing|nothing on|in (her|his|their) (underwear|bra and panties|lingerie))\b", re.I)


def worn_words(notes: str) -> str:
    """An outfit's words for the cast line: a state of dress is said as it is ("naked", "topless"), only real clothes get
    "wearing" ("topless, wearing black jeans"). Before, a "Naked" outfit card read "wearing naked, wearing nothing" and
    one with a picture "wearing the Naked"."""
    state, clothes = [], []
    for part in re.split(r"[,;]", str(notes or "")):
        p = part.strip().rstrip(".")
        if not p:
            continue
        if re.search(r"\b(wearing nothing|nothing on)\b", p, re.I):
            state.append("naked")
        elif _UNDRESS.search(p) or re.match(r"(wrapped|draped|covered|dressed|clad|in)\b", p, re.I):
            # "wrapped in a white towel", "dressed in black": said as it is, never "wearing wrapped in ..."
            state.append(re.sub(r"^\s*wearing\s+", "", p, flags=re.I))
        else:
            clothes.append(re.sub(r"^\s*wearing\s+", "", p, flags=re.I))
    state = list(dict.fromkeys(state))
    return ", ".join(state + (["wearing " + ", ".join(clothes)] if clothes else []))


def wardrobe(story: dict) -> dict:
    """{person: [outfit]}: the cast card's own 👗 outfit picture is outfit 1, then the person's outfit cards in order.
    An outfit is {"id" (its reference), "label", "picture", "card", "notes"}. With a picture, the label is all H3 reads
    about it. A name that contradicts the picture wins ("the red dress" over a white-sweater picture); one that agrees
    held it 8 of 8, where "outfit 2" held 4 of 8 against the headshot's own clothes (2026-11-02)."""
    out = {}
    for c in story.get("cast", []) or []:
        nm = str(c.get("name") or "").strip()
        if not nm:
            continue
        items = out.setdefault(nm, [])
        if _pic_file(c.get("outfit_picture")):
            items.append({"id": f"{c.get('id')}_dress", "label": outfit_name(c.get("outfit_notes")), "picture": c["outfit_picture"],
                          "card": None, "notes": ""})
    for o in story.get("outfits", []) or []:
        if not isinstance(o, dict) or not str(o.get("name") or "").strip():
            continue
        who = str(o.get("who") or "").strip()
        if who not in out:
            continue
        items = out[who]
        name = re.sub(r"^(the|a|an)\s+", "", str(o.get("name")).strip(), flags=re.I).rstrip(".")
        items.append({"id": f"{o.get('id')}_outfit", "label": f"the {name}" if name else f"outfit {len(items) + 1}",
                      "picture": o.get("picture") if _pic_file(o.get("picture")) else None, "card": o,
                      "notes": _one_clause(o.get("description") or o.get("name"))})
    return out


def _behind_outfit(story: dict, shot: dict, names: list) -> str:
    """", Marco wearing a dark grey shirt ...": the person an over-the-shoulder shot shows from behind. Only the other
    person's pictures go to H3 there, so without words Marco wore her parka (Parka night cafe v5)."""
    who = shot.get("subject") if shot.get("subject") in names else ""
    other = next((n for n in names if n != who), "")
    if not who or not other:
        return ""
    notes = ""
    for c in story.get("cast", []) or []:
        if str(c.get("name") or "").strip() == other:
            notes = str(c.get("outfit_notes") or "").strip()
    if not notes:
        notes = next((str(o.get("description") or o.get("name") or "").strip() for o in story.get("outfits", []) or []
                      if isinstance(o, dict) and str(o.get("who") or "").strip() == other), "")
    notes = frame_look(_one_clause(notes), True) if notes else ""
    return f", {other} wearing {notes}" if notes else ""


def build_plan(story: dict, max_s: float = 20.0) -> tuple:
    """story -> (plan for LC H3 Segment, per-shot preview list, warnings)."""
    story = inherit_rolls(story)  # a keep-rolling card takes the framing and place of the cut before it
    story = {**story, "shots": [_one_person(s, _names(story)) if isinstance(s, dict) else s for s in story.get("shots") or []]}
    story = _with_auto_views(story)
    max_s = max(5.0, float(max_s))  # the node's minimum: a part shorter than the overlap never moves on (endless loop)
    names = _names(story)
    cast = [c for c in story.get("cast", []) if str(c.get("name") or "").strip()]
    warnings = []
    for c in cast:
        if not re.match(r"[A-Z]", c["name"].strip()):
            warnings.append(f"🔠 Start the name \"{c['name']}\" with a capital letter: H3 reads it as a name (and who speaks) by the capital. In the shots you can write it any way.")
    # a look is one clause: "...curvy body. wearing a white sweater..." split at the period, and when the Segment
    # dropped Anika's clause from Nova's close-up, "wearing a white sweater and black pleated skirt" stayed and Nova
    # wore it (Coffee Date)
    look_of = {c["name"]: _one_clause(c.get("look")) for c in cast}

    def style_for(worn: dict, tight: bool = False, nobody: bool = False) -> str:
        """The cast line; a person in an outfit gets "wearing <outfit>" after their look. tight: a close-up or medium
        shot, so the clothes that can't be in frame are left out. nobody: an empty shot, the visual style only (the
        cast's looks would put them in the frame)."""
        # never empty for an empty shot: Segment falls back to the story's whole cast line when a shot's style is blank
        bits = ["No people are in this shot"] if nobody else []
        for c in ([] if nobody else cast):
            nm = c["name"].strip()
            o = worn.get(nm)
            if o and o["picture"]:
                # the outfit picture shows the clothes: no clothing words (they beat the picture and pull close-ups wide)
                look = strip_clothes(look_of[c["name"]])
                state = _UNDRESS.search(o["label"] + " " + (o.get("notes") or ""))
                # a state of dress ("Topless") is said as it is, never "wearing the Topless"
                extra = ([look] if look else []) + [worn_words(o.get("notes") or o["label"]) if state else "wearing " + o["label"]]
            else:
                look = frame_look(look_of[c["name"]], tight)
                extra = [look] if look else []
                if o:  # an outfit card with no picture: its words are all there is
                    worn_look = frame_look(o["notes"], tight) or (o["notes"] if _UNDRESS.search(o["notes"] or "") else "")
                    if worn_look:
                        extra.append(worn_words(worn_look))
            bits.append(f"{nm}: a {c.get('who') or 'person'}" + (", " + ", ".join(extra) if extra else ""))
        if str(story.get("look") or "").strip():
            bits.append("Visual style: " + str(story["look"]).strip().rstrip("."))
        return "; ".join(bits) + ("." if bits else "")

    outfits = [o for o in story.get("outfits", []) or [] if isinstance(o, dict) and str(o.get("name") or "").strip()]
    for o in outfits:
        if str(o.get("who") or "").strip() not in names:
            warnings.append(f"👗 Pick who wears \"{o['name']}\".")
    robe = wardrobe(story)
    pictured = {nm for nm, items in robe.items() if any(o["picture"] for o in items)}
    for c in cast:
        nm = c["name"].strip()
        clothes_said = strip_clothes(str(c.get("look") or "")) != str(c.get("look") or "")
        if nm in pictured and clothes_said:
            warnings.append(f"👗 {nm}'s look names clothes: they are left out, the outfit picture shows them. Keep the look to face and hair.")
        body = _BODY_RE.findall(str(c.get("look") or ""))
        if nm in pictured and body:
            # "medium breasts" in the look + a bare-shouldered headshot: H3 undressed both women ("Uber run", 2026-11-07)
            warnings.append(f"👗 {nm}'s look names the body ({', '.join(dict.fromkeys(b.lower() for b in body))}): next to a "
                            "bare-shouldered headshot H3 can leave the clothes off. Keep the look to face and hair; the outfit "
                            "picture shows the body.")
        elif nm not in pictured and clothes_said:
            warnings.append(f"💡 {nm}: put the clothes in a 👗 outfit picture on the cast card. Clothing words beat the picture "
                            "and pull close-ups wide.")
        elif nm not in pictured and not robe.get(nm):
            # headshot cards carry no clothes: with no outfit either, H3 and Krea dressed her differently every shot (test)
            warnings.append(f"👗 {nm} has no outfit picture: the clothes will change from shot to shot. Add a 👗 outfit picture "
                            "on the cast card.")
    for o in story.get("outfits", []) or []:
        # a simple name ("gym clothes"): a list of pieces is read as words, and shoes and pants pull close-ups wide
        on = str((o or {}).get("name") or "").strip()
        if on and len(re.split(r",|\band\b|\bwith\b", on)) > 2:
            warnings.append(f"👗 Outfit \"{on}\": give it a simple name (gym clothes, red cocktail dress). The pieces go in "
                            "its description, which only makes the picture.")
    # person -> the outfit they are in now: the cast card's own outfit from the start, then the shot that picks or names one
    worn = {nm: items[0] for nm, items in robe.items() if items and items[0]["card"] is None}
    style = style_for(dict(worn))
    # Segment finds the cast and their sex in the story text ("Nova, a woman ...")
    cast_story = " ".join(f"{c['name'].strip()}, a {c.get('who') or 'person'} {look_of[c['name']]}".strip() + "." for c in cast)

    ovl_s = OVERLAP_FRAMES / FPS
    plan_shots, previews = [], []
    end = 0.0
    where_now = ""
    shots = [s for s in story.get("shots", []) if isinstance(s, dict)]
    if not shots:
        warnings.append("🛑 Add at least one shot.")
    first_seg = {}  # setup (place, framing, subject) -> the first segment that opened on it with a cut
    seen = {}  # segment -> what its frames show: {"shot", "place", "wide", "people", "props", "worn"}
    prop_names = [str(r.get("name") or "").strip() for r in story.get("props", []) or [] if str(r.get("name") or "").strip()]

    def _mismatch(old, now, people=True):
        """What an earlier segment's frame shows that this shot doesn't: other clothes, people, props."""
        out = []
        for nm in sorted(old["people"] & now["people"]):
            a, b = old["worn"].get(nm), now["worn"].get(nm)
            if a and b and a["id"] != b["id"]:
                out.append(f"{nm} in {a['label']}")
        if people:
            out += [f"{nm} (not in this shot)" for nm in sorted(old["people"] - now["people"])]
            out += [f"no {nm}" for nm in sorted(now["people"] - old["people"])]
            out += [f"the {p} (not in this shot)" for p in sorted(old["props"] - now["props"])]
        return out

    for i, shot in enumerate(shots, start=1):
        sw = []
        roll = i > 1 and shot.get("join") == "roll" and _same_camera(shots[i - 2], shot)
        where = str(shot.get("where") or "").strip() or where_now or (story.get("places") or [{}])[0].get("name", "")
        place = _place(story, where)
        where_now = where
        said = " ".join(str(shot.get(k) or "") for k in ("action", "after"))
        for nm, items in robe.items():  # a shot that names an outfit (its card name, or "Anika ... outfit 1") puts it on, from here on
            for o in items:
                named = o["card"] and re.search(rf"\b{re.escape(o['card']['name'].strip())}\b", said, re.I)
                labelled = re.search(rf"\b{re.escape(o['label'])}\b", said, re.I) and re.search(rf"\b{re.escape(nm)}\b", said, re.I)
                if named or labelled:
                    worn[nm] = o
        for nm, oid in (shot.get("outfit") or {}).items():  # the shot card's 👗 dropdown
            o = next((o for o in robe.get(nm, []) if o["id"] == oid), None)
            if o:
                worn[nm] = o
        empty = shot.get("subject") == NOBODY
        shot_style = style_for(worn, (shot.get("framing") or "wide shot") in TIGHT, empty)
        line = _clean_line(shot.get("line"))
        if empty:
            if line and not _who_says(shot.get("speaker"))[1]:
                sw.append("🛑 Nobody is in this shot: its line can only be a voiceover. Pick the speaker's (V.O.).")
            named = [n for n in names if re.search(rf"\b{re.escape(n)}\b", said, re.I)]
            if named:
                sw.append(f"⚠️ Nobody is in this shot, but it names {', '.join(named)}: H3 may put them in. Leave them out of the action.")
        if line and not (shot.get("speaker") or len(names) == 1):
            sw.append("🛑 Pick who says the line.")
        elif line and shot.get("speaker") and _who_says(shot.get("speaker"))[0] not in names:
            sw.append(f"🛑 {_who_says(shot.get('speaker'))[0]} is not in the cast.")
        if (shot.get("framing") or "wide shot") in TIGHT and (shot.get("framing") or "") != "over the shoulder" and shot.get("subject") in names:
            others = [n for n in names if n != shot["subject"] and re.search(rf"\b{re.escape(n)}\b", said, re.I)]
            if others:
                # "Anika stares at Nova": H3 cut to Nova inside Anika's close-up, even with only Anika's pictures sent (test)
                sw.append(f"⚠️ This close-up names {', '.join(others)}: H3 may cut to them halfway through. Write \"looks across the "
                          "table\" (or \"at the person across from her\") instead.")
        sec, lw = _shot_seconds(shot, max_s)
        sw += lw
        typed = _typed_beats(str(shot.get("action") or ""))
        beats_all = typed or _auto_beats(shot, sec, names)
        if sec > max_s + 0.01:
            sw.append(f"⚠️ {sec:g} s is longer than max_shot_seconds ({max_s:g} s): made in parts, each carrying on from the one before.")
            parts = _split_beats(beats_all, sec, max_s, ovl_s)
        else:
            parts = [[0.0, sec, None]]
        segs = []
        for k, (p0, p1, pbeats) in enumerate(parts):
            cont = roll if k == 0 else True
            span = p1 - p0
            part_shot = dict(shot)
            if pbeats is not None:  # the part's own beats, written back as typed beats
                part_shot["action"] = " ".join(f"At {t:.2f} seconds, {b}" for t, b in pbeats) or "At 0.00 seconds, the moment carries on."
            elif not typed:
                part_shot["action"] = " ".join(f"At {t:.2f} seconds, {b}" for t, b in beats_all)
            # only the first part of a keep-rolling shot moves the camera from the shot before
            prev = shots[i - 2] if roll and k == 0 else None
            if not cont and _seated(shots, shot, where, story):
                part_shot["_seated"] = True
            text, tw = _shot_text(story, part_shot, place, span, ovl_s if cont else 0.0, names, shot, prev)
            sw += tw
            auto_text = text
            hand = shot.get("prompt_edit") if isinstance(shot.get("prompt_edit"), dict) else {}
            edited = str(hand.get(str(k + 1)) or "").strip()
            if edited:  # the user's own words: exactly what H3 gets for this part, whatever the cards say
                text = edited
            frames = snap_frames(span + (ovl_s if cont else 0.0))
            start = round(end - ovl_s, 3) if cont else round(end, 3)
            end = start + frames / FPS
            seg = {"index": len(plan_shots) + 1, "mode": "CONTINUE" if cont else "CUT", "action": text, "start_s": start,
                   "frames": frames, "prompt": f"{shot_style}\n\n{text}".strip(), "place_name": str(place.get("name")) if place else "",
                   "shot_id": shot.get("id") or f"s{i}", "shot": i, "part": k + 1, "parts": len(parts),
                   # a place whose picture is already a close view (cut from a 360 panorama) goes to tight shots too
                   "room": (shot.get("framing") or "wide shot") not in TIGHT or bool((place or {}).get("close_view"))}
            if (shot.get("framing") or "wide shot") in TIGHT:
                seg["tight"] = True  # Segment leaves the clothes out of frame out of the picture notes too
            if (shot.get("framing") or "wide shot") in TIGHT and shot.get("subject") in names:
                seg["subject"] = shot["subject"]  # Segment sends this one person's pictures only (others named stay words)
            if empty:
                seg["subject"] = NOBODY  # Segment sends no faces, voices or memory of the cast
            if _shot_view(shot, place):  # the shot's own angle out of the place's 360 panorama replaces the room picture
                seg["view_name"] = _view_name(place, shot)
            seg["sig"] = _sig(story, seg, shot, place)
            loras = shot_loras(shot)
            if loras:  # LC H3 Make adds them to the model for this shot only; a new LoRA or strength makes the take again
                seg["loras"] = loras
                seg["sig"] += "|loras:" + ",".join(f"{x['lora']}@{x['strength']:g}" for x in loras)
            if any(robe.values()):  # only the outfits worn now go to this shot (none = their look)
                seg["outfits"] = [o["id"] for o in worn.values() if o["picture"]]
                seg["sig"] += "".join(f"|outfit:{_pic_file(o['picture'])}:{(o['picture'] or {}).get('crop')}" for o in worn.values() if o["picture"])
            if shot_style != style:
                seg["style"] = shot_style
            setup = (str(place.get("name")) if place else "", shot.get("framing") or "wide shot", shot.get("subject") or "everyone")
            subj = shot.get("subject") or "everyone"
            who = set() if subj == NOBODY else set(names) if subj not in names else {subj}
            if subj in names and (shot.get("framing") or "wide shot") not in TIGHT:
                who |= {n for n in names if re.search(rf"\b{re.escape(n)}\b", said, re.I)}
            now = {"shot": i, "place": setup[0], "wide": (shot.get("framing") or "wide shot") not in TIGHT, "people": who,
                   "props": {p for p in prop_names if re.search(rf"\b{re.escape(p)}\b", said, re.I)},
                   "worn": {nm: o for nm, o in worn.items() if nm in who}}
            if k == 0 and _pic_file(shot.get("start")):
                if cont:
                    sw.append("⚠️ A keep-rolling shot starts from the shot before: its start frame is not used. Make it a cut to use it.")
                else:  # pinned as frame 0: the one way H3 keeps a background exactly
                    seg["start_frame"] = _pic_file(shot.get("start"))
            elif k == 0 and not cont and setup in first_seg:
                # a cut back to a setup already shown opens on that shot's first frame: the same camera, room and people.
                # It changes with that shot: remade when its anchor is remade
                seg["return_to"] = first_seg[setup]
                seg["sig"] += "|return:" + plan_shots[first_seg[setup] - 1]["sig"]
                diff = _mismatch(seen[first_seg[setup]], now)
                if diff:
                    sw.append(f"⚠️ Mismatched reference: this shot opens on shot {seen[first_seg[setup]]['shot']}'s first frame, "
                              f"which shows {', '.join(diff)}. H3 copies that frame. Pin a start frame, or change the camera.")
            elif k == 0 and not cont and now["wide"]:
                # a wide cut keeps the room from the place's first wide shot (its frame shows who wore what there)
                room = next((s for s in seen.values() if s["place"] == now["place"] and s["wide"]), None)
                diff = _mismatch(room, now, people=False) if room and room["shot"] != i else []
                if diff:
                    sw.append(f"⚠️ Mismatched reference: the room comes from shot {room['shot']}, which shows {', '.join(diff)}. "
                              "The clothes can carry over. Pin a start frame, or give this scene its own place card.")
            if k == len(parts) - 1 and _pic_file(shot.get("end")):  # the shot lands on its end frame (storyboard)
                seg["end_frame"] = _pic_file(shot.get("end"))
            if k == 0 and not cont:
                first_seg.setdefault(setup, seg["index"])
            seen[seg["index"]] = now
            plan_shots.append(seg)
            segs.append({"index": seg["index"], "mode": seg["mode"], "seconds": round(frames / FPS, 2), "text": text,
                         "edited": bool(edited), "auto": auto_text if edited else ""})
        previews.append({"shot": i, "seconds": sec, "segments": segs, "warnings": list(dict.fromkeys(sw))})
        warnings += [f"Shot {i}: {w}" for w in dict.fromkeys(sw)]
    plan = {"version": 1, "source": "story", "project": str(story.get("project") or "untitled").strip() or "untitled",
            "story": cast_story, "style": style, "fps": FPS, "segment_s": round(snap_frames(DEFAULT_SECONDS) / FPS, 3),
            "overlap_s": round(ovl_s, 3), "segment_frames": snap_frames(DEFAULT_SECONDS), "overlap_frames": OVERLAP_FRAMES,
            "segments": len(plan_shots), "shots": plan_shots, "warnings": warnings,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    return plan, previews, warnings


def build_refs(story: dict):
    """The cast and place cards as the Reference Assembler's items, built the same way it builds them."""
    items = []
    robe = wardrobe(story)
    for c in story.get("cast", []):
        nm = str(c.get("name") or "").strip()
        if not nm:
            continue
        pictured = any(o["picture"] for o in robe.get(nm, []))
        f = _pic_file(c.get("picture"))
        if f:
            look = str(c.get("look") or "")
            items.append({"id": f"{c.get('id')}_face", "kind": "image", "file": f, "role": "face", "name": nm,
                          "note": strip_clothes(look) if pictured else look, "crop": (c.get("picture") or {}).get("crop")})
        wf = _pic_file(c.get("outfit_picture"))
        if wf:  # the cast card's own outfit, named from its notes ("dark blue jumpsuit"; else "outfit 1")
            items.append({"id": f"{c.get('id')}_dress", "kind": "image", "file": wf, "role": "outfit", "name": nm,
                          "note": outfit_name(c.get("outfit_notes")), "crop": (c.get("outfit_picture") or {}).get("crop")})
        v = _pic_file(c.get("voice"))
        if v:
            items.append({"id": f"{c.get('id')}_voice", "kind": "audio", "file": v, "role": "voice", "name": nm,
                          "trim": (c.get("voice") or {}).get("trim")})
    for p in story.get("places", []):
        nm = str(p.get("name") or "").strip()
        f = _pic_file(p.get("picture"))
        if f:
            items.append({"id": f"{p.get('id')}_room", "kind": "image", "file": f, "role": "view" if p.get("close_view") else "setting", "name": nm,
                          "note": str(p.get("description") or ""), "crop": (p.get("picture") or {}).get("crop")})
        for k, a in enumerate(p.get("anchors") or []):
            af = _pic_file(a)
            if af:
                items.append({"id": f"{p.get('id')}_anchor{k}", "kind": "image", "file": af, "role": "layout", "name": nm,
                              "note": "anchor"})
    names = set(_names(story))
    for o in story.get("outfits", []) or []:  # worn by one person; the plan says which shots
        f = _pic_file(o.get("picture"))
        who = str(o.get("who") or "").strip()
        if f and who in names:
            label = next((x["label"] for x in robe.get(who, []) if x["card"] is o), "outfit")
            items.append({"id": f"{o.get('id')}_outfit", "kind": "image", "file": f, "role": "outfit", "name": who,
                          "note": label, "crop": (o.get("picture") or {}).get("crop")})
    for r in story.get("props", []) or []:  # a shot gets a prop's picture when its text names the prop
        nm = str(r.get("name") or "").strip()
        f = _pic_file(r.get("picture"))
        if nm and f:
            items.append({"id": f"{r.get('id')}_prop", "kind": "image", "file": f, "role": "prop", "name": nm,
                          "note": str(r.get("description") or ""), "crop": (r.get("picture") or {}).get("crop")})
    for shot, place, v, f in cut_views(story):
        items.append({"id": f"{shot.get('id')}_view", "kind": "image", "file": f, "role": "view",
                      "name": _view_name(place, shot), "note": v["behind"] or str(place.get("description") or "")})
    if not items:
        return None, ""
    refs, warn = _build_refs(json.dumps({"items": items}))
    return refs, warn


# ------------------------------------------------------------------ the project folder
def _story_path(project: str) -> str:
    return os.path.join(project_dir(project), "story.json")


def _write_json(path: str, data: dict) -> None:
    """Write to a temp file, then swap it in: a reader (or a crash) never sees half a file."""
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _set_aside(folder: str, indices: list) -> list:
    """Segments whose shot moved, was removed or was replaced: their files go to takes/moved_<time>/ (never deleted),
    so a different shot's take is never counted as done."""
    moved = []
    dest = os.path.join(work_dir(folder), "moved_" + time.strftime("%Y%m%d_%H%M%S"))
    for i in indices:
        if not _is_done(folder, i):
            continue
        prefix = os.path.basename(_seg_path(folder, i, ""))
        d = os.path.dirname(_seg_path(folder, i, ""))
        for f in os.listdir(d):
            if f.startswith(prefix):
                os.makedirs(dest, exist_ok=True)
                shutil.move(os.path.join(d, f), os.path.join(dest, f))
        moved.append(i)
    return moved


def save_project(story: dict, max_s: float = 20.0) -> dict:
    """story.json + plan.json. A segment whose shot changed place in the list is set aside; an edited shot keeps its
    take (the card offers ↻) and the shots after a removed one are never renumbered onto old takes."""
    plan, previews, warnings = build_plan(story, max_s)
    folder = project_dir(plan["project"])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "plan.json")
    moved = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            old_t = os.path.getmtime(path)
            old_ids = [(s.get("shot_id"), s.get("part")) for s in old.get("shots", [])]
            new_ids = [(s.get("shot_id"), s.get("part")) for s in plan["shots"]]
            if old.get("source") != "story":
                # a Showrunner project opened in Story for the first time: its renders were made from other text
                changed = set(range(1, len(old_ids) + 1))
            else:
                changed = {k + 1 for k, oid in enumerate(old_ids) if k >= len(new_ids) or new_ids[k] != oid}
                # a keep-rolling segment right after a set-aside one no longer joins either
                for k in range(1, len(new_ids)):
                    if k in changed and plan["shots"][k]["mode"] == "CONTINUE":
                        changed.add(k + 1)
            moved = _set_aside(folder, sorted(changed))
            for k, s in enumerate(plan["shots"], start=1):  # locks stay with their shot
                o = next((x for x in old.get("shots", []) if x.get("shot_id") == s["shot_id"] and x.get("part") == s["part"]), None)
                if o and o.get("locked"):
                    s["locked"] = o["locked"]
                # what the take on disk was made from: the plan saved before it was made (Story saves on every Run,
                # before the shot is made), kept until the shot is made again
                if o and k not in changed and _is_done(folder, k):
                    take_t = os.path.getmtime(_seg_path(folder, k, ".json"))
                    made = o.get("made")
                    if take_t >= old_t:  # made after the old plan was saved: made from it
                        made = {"t": take_t, "prompt": o.get("prompt"), "sig": o.get("sig")}
                    elif not made and "action" not in _take_meta(folder, k):
                        # an older take with no record of its own: the plan it sat under is the best guess
                        made = {"t": take_t, "prompt": o.get("prompt"), "sig": o.get("sig")}
                    if made:
                        s["made"] = made
        except Exception as exc:
            print(f"[LC H3] Story: could not read the old plan ({exc}), writing a new one.")
    _write_json(path, plan)
    story["saved"] = time.time()
    _write_json(_story_path(plan["project"]), story)
    cut_views(story)
    if moved:
        print(f"[LC H3] Story: segments {', '.join(map(str, moved))} moved to takes/ (their shot moved or changed place).")
    return {"plan": plan, "previews": previews, "warnings": warnings, "moved": moved, "saved": story["saved"]}


def _import_old(project: str) -> dict | None:
    """An old Shot List + Reference Assembler project (project.json) as a story."""
    p = os.path.join(project_dir(project), "project.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        old = json.load(f)
    sl = old.get("shot_list") or {}
    items = ((old.get("references") or {}).get("items") or [])

    def pic(file):
        file = str(file or "").replace("\\", "/")
        sub, _, name = file.rpartition("/")
        return {"type": "input", "subfolder": sub, "filename": name} if name else None

    cast = []
    for k, c in enumerate(sl.get("cast") or []):
        nm = str(c.get("name") or "").strip()
        face = next((it for it in items if it.get("name") == nm and it.get("role") in ("face", "full body") and it.get("on", True)), None)
        voice = next((it for it in items if it.get("name") == nm and it.get("role") == "voice" and it.get("on", True)), None)
        cast.append({"id": f"c{k + 1}", "name": nm, "who": c.get("who") or "person",
                     "look": (face or {}).get("note") or c.get("look") or "", "picture": pic(face["file"]) if face else None,
                     "voice": pic(voice["file"]) if voice else None})
    places = []
    for it in items:
        if it.get("role") == "setting" and it.get("on", True):
            places.append({"id": f"p{len(places) + 1}", "name": it.get("name") or f"place {len(places) + 1}",
                           "description": it.get("note") or "", "picture": pic(it.get("file")), "anchors": []})
    if not places:
        places.append({"id": "p1", "name": "the place", "description": str(sl.get("place") or ""), "picture": None, "anchors": []})
    for it in items:
        if it.get("role") == "layout" and it.get("on", True):
            target = _place({"places": places}, it.get("name")) or places[0]
            target["anchors"].append(pic(it.get("file")))
    shots = []
    for k, s in enumerate(sl.get("shots") or []):
        shots.append({"id": f"s{k + 1}", "framing": s.get("framing") or "wide shot", "subject": s.get("subject") or "everyone",
                      "move": s.get("move") or "still", "join": s.get("join") or "cut",
                      "where": s.get("where") or places[0]["name"], "action": s.get("action") or "", "speaker": s.get("speaker") or "",
                      "line": _clean_line(s.get("line")), "after": s.get("after") or "", "seconds": ""})
    return {"version": 1, "project": project, "look": "", "cast": cast, "places": places, "shots": shots, "imported": True}


def load_project(project: str) -> dict | None:
    p = _story_path(project)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return _import_old(project)


def list_images(where: str, limit: int = 240) -> list:
    root = folder_paths.get_output_directory() if where == "output" else folder_paths.get_input_directory()
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith((".", "_")) and x not in ("takes", "temp")]
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                full = os.path.join(d, f)
                try:
                    out.append((os.path.getmtime(full), os.path.relpath(d, root).replace("\\", "/"), f))
                except OSError:
                    pass
    out.sort(reverse=True)
    return [{"type": "output" if where == "output" else "input", "subfolder": "" if s == "." else s, "filename": f}
            for _, s, f in out[:limit]]


# ------------------------------------------------------------------ the board (what is made)
def _thumb(folder: str, index: int) -> str | None:
    """takes/seg_###_thumb.jpg, a frame from a third of the way in, made again when the take is newer. From the part
    the movie uses: a keep-rolling take opens on the shot before's tail, which isn't this shot."""
    src, out = _seg_path(folder, index, "_part.mp4"), _seg_path(folder, index, "_part_thumb.jpg")
    if not os.path.exists(src):
        return None
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        import subprocess

        from .lc_save_video import _ffmpeg_bin

        try:  # runs inside the board route: a stuck ffmpeg must not hang the server
            subprocess.run([_ffmpeg_bin(), "-v", "error", "-y", "-ss", "1.0", "-i", src, "-frames:v", "1", "-vf", "scale=320:-2", out],
                           capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            print(f"[LC H3] Story: no thumbnail for segment {index} (ffmpeg timed out).")
    return out if os.path.exists(out) else None


def board(project: str, now: dict | None = None) -> dict:
    """Per shot: its segments, made or not, changed since made (the text H3 gets now, from the cards as they are on
    screen when `now` (a fresh plan) is given, differs from what it got), locked, and the thumbnail / video as /view
    paths. Which take belongs to which shot comes from the saved plan.json."""
    folder = project_dir(project)
    cur = {(s.get("shot_id"), s.get("part")): s for s in (now or {}).get("shots", [])}
    try:
        with open(os.path.join(folder, "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
    except Exception:
        return {"shots": []}
    out_root = folder_paths.get_output_directory()
    rel = lambda p: {"type": "output", "subfolder": os.path.relpath(os.path.dirname(p), out_root).replace("\\", "/"),
                     "filename": os.path.basename(p), "t": int(os.path.getmtime(p))}
    shots = {}
    edited = changed_since_made(folder, plan, now)
    for k, s in enumerate(plan.get("shots", []), start=1):
        made = _is_done(folder, k)
        changed = k in edited
        if now is not None and (s.get("shot_id"), s.get("part")) not in cur:
            continue  # a shot taken out on screen, not saved yet
        th = _thumb(folder, k) if made else None
        # a made keep-rolling take whose opening came from an older ending of the segment before (that one was remade
        # since): the join jumps until this one is remade too. The take records the tail it started from.
        broken = False
        if made and s.get("mode") == "CONTINUE" and k > 1:
            try:
                with open(_seg_path(folder, k, ".json"), encoding="utf-8") as f:
                    from_t = float(json.load(f).get("from_tail_mtime") or 0)
                tail = _seg_path(folder, k - 1, "_tail.mp4")
                broken = bool(from_t) and os.path.exists(tail) and os.path.getmtime(tail) > from_t + 1.0
            except (OSError, ValueError, TypeError):
                broken = False
        seg = {"index": k, "part": s.get("part", 1), "made": made, "changed": changed, "locked": bool(s.get("locked")),
               "mode": s.get("mode"), "broken": broken,
               "seconds": round(s.get("frames", 0) / FPS, 2), "thumb": rel(th) if th else None,
               # the part the movie uses: a keep-rolling take's file opens on the shot before's last 1.6 s
               "video": rel(_seg_path(folder, k, "_part.mp4")) if made else None}
        shots.setdefault(s.get("shot_id") or f"s{k}", {"shot_id": s.get("shot_id"), "shot": s.get("shot", k), "segments": []})["segments"].append(seg)
    full = os.path.join(folder, f"{os.path.basename(folder)}.mp4")
    return {"shots": list(shots.values()), "movie": rel(full) if os.path.exists(full) else None}


def set_lock(project: str, shot_id: str, locked: bool) -> None:
    path = os.path.join(project_dir(project), "plan.json")
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    for s in plan.get("shots", []):
        if s.get("shot_id") == shot_id:
            s["locked"] = bool(locked)
    t = os.path.getmtime(path)
    _write_json(path, plan)
    os.utime(path, (t, t))  # a lock is not a new plan: "changed since made" compares takes with the plan's time


def _parse(story_json) -> dict:
    try:
        d = json.loads(story_json or "{}") if isinstance(story_json, str) else (story_json or {})
    except ValueError:
        raise ValueError("[LC H3] Story: the story is not valid. Undo the last change, or 📂 Load the project again.")
    return d if isinstance(d, dict) else {}


# ------------------------------------------------------------------ the node
class LCH3Story:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story": ("STRING", {"default": json.dumps(EXAMPLE), "tooltip": "The cast, places and shots as JSON. Fill it in on the node."}),
                "max_shot_seconds": ("FLOAT", {"default": 20.0, "min": 5.0, "max": 30.0, "step": 0.5,
                                               "tooltip": "The longest shot this machine renders in one go. A longer shot is made in parts, each carrying on from the one before."}),
            },
        }

    RETURN_TYPES = (PLAN_TYPE, REFS_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("plan", "refs", "project_name", "warnings")
    OUTPUT_TOOLTIPS = ("For LC H3 Make: one segment per shot.", "The cast and place pictures, for LC H3 Make's refs.",
                       "The project folder name.", "What to fix. Empty when it all looks good.")
    FUNCTION = "run"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("The whole movie on one node: cast, places and shots. Each shot shows its lines to read and edit; LC H3 Make "
                   "wraps the people, pictures and sound around them when the shot is made. Saves story.json and plan.json in "
                   "output/LC_H3/<project>/.")

    def run(self, story, max_shot_seconds=20.0):
        data = _parse(story)
        from .lc_h3_library import ensure_pics

        ensure_pics(data)  # library pictures a sample or shared story uses: downloaded once, before anything reads them
        res = save_project(data, float(max_shot_seconds))
        refs, rwarn = build_refs(data)
        warn = "\n".join(res["warnings"] + ([rwarn] if rwarn else []))
        if warn:
            print("[LC H3] Story: " + " | ".join(warn.splitlines()))
        return {"ui": {"text": [warn or "✅ Ready."], "lc_saved": [res["saved"]]}, "result": (res["plan"], refs, res["plan"]["project"], warn)}


def project_sizes(project: str) -> dict:
    """The size the project is made at (size.json: LC H3 Make / Segment write the size they run at), and the size of
    each made shot (from its seg_NNN.json)."""
    folder = project_dir(project)
    takes = os.path.join(folder, "takes")
    made = []
    if os.path.isdir(takes):
        for name in sorted(os.listdir(takes)):
            m = re.fullmatch(r"seg_(\d{3})\.json", name)
            if not m:
                continue
            path = os.path.join(takes, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                made.append({"index": int(m.group(1)), "width": int(j["width"]), "height": int(j["height"]), "mtime": os.path.getmtime(path)})
            except (OSError, ValueError, KeyError, TypeError):  # a shot still being written
                continue
    first = None
    try:
        with open(os.path.join(folder, "size.json"), "r", encoding="utf-8") as f:
            j = json.load(f)
        first = {"width": int(j["width"]), "height": int(j["height"])}
    except (OSError, ValueError, KeyError, TypeError):
        if made:  # an older project: the oldest shot on disk is the best guess
            old = min(made, key=lambda s: s["mtime"])
            first = {"width": old["width"], "height": old["height"], "guess": True}
    return {"first": first, "shots": [{k: v for k, v in s.items() if k != "mtime"} for s in made]}


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/lc_h3/story/size")
    async def _size_route(request):
        try:
            return web.json_response(project_sizes(request.query.get("project", "")))
        except Exception as exc:
            return web.json_response({"error": str(exc), "first": None, "shots": []}, status=400)

    @PromptServer.instance.routes.get("/lc_h3/story/load")
    async def _load_route(request):
        try:
            return web.json_response({"story": load_project(request.query.get("project", ""))})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/lc_h3/story/save")
    async def _save_route(request):
        try:
            body = await request.json()
            res = save_project(_parse(body.get("story")), float(body.get("max_shot_seconds") or 20.0))
            return web.json_response({"project": res["plan"]["project"], "moved": res["moved"], "warnings": res["warnings"], "saved": res["saved"]})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/lc_h3/story/save_as")
    async def _save_as_route(request):
        """A variant: the whole project folder (made shots, takes, views) copied under a new name, then the story
        saved there. The original is left as it was."""
        try:
            import asyncio

            body = await request.json()
            story = _parse(body.get("story"))
            name = str(body.get("name") or "").strip()
            if not name:
                raise ValueError("Give the copy a name.")
            src, dst = project_dir(story.get("project", "")), project_dir(name)
            if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst)):
                raise ValueError("That is this project's own name: pick a new one.")
            if os.path.exists(dst):
                raise ValueError(f'"{os.path.basename(dst)}" already exists: pick another name.')
            if os.path.isdir(src):
                await asyncio.get_running_loop().run_in_executor(None, shutil.copytree, src, dst)
                movie = os.path.join(dst, f"{os.path.basename(src)}.mp4")  # the finished movie is named after its folder
                if os.path.exists(movie):
                    os.replace(movie, os.path.join(dst, f"{os.path.basename(dst)}.mp4"))
            story["project"] = name
            res = save_project(story, float(body.get("max_shot_seconds") or 20.0))
            return web.json_response({"project": res["plan"]["project"], "saved": res["saved"], "copied": os.path.isdir(src)})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/lc_h3/story/preview")
    async def _preview_route(request):
        try:
            body = await request.json()
            _, previews, warnings = build_plan(_parse(body.get("story")), float(body.get("max_shot_seconds") or 20.0))
            return web.json_response({"shots": previews, "warnings": warnings})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.get("/lc_h3/story/images")
    async def _images_route(request):
        try:
            return web.json_response({"images": list_images(request.query.get("where", "input"))})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/lc_h3/story/board")
    async def _board_route(request):
        try:
            body = await request.json()
            story = _parse(body.get("story"))
            now, _, _ = build_plan(story, float(body.get("max_shot_seconds") or 20.0))
            return web.json_response(board(now["project"], now))
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/lc_h3/story/lock")
    async def _lock_route(request):
        try:
            body = await request.json()
            set_lock(body.get("project", ""), body.get("shot_id", ""), bool(body.get("locked")))
            return web.json_response({"ok": True})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.get("/lc_h3/story/projects")
    async def _projects_route(request):
        try:
            root = os.path.join(folder_paths.get_output_directory(), "LC_H3")
            names = sorted((d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("_")),
                           key=lambda d: -os.path.getmtime(os.path.join(root, d))) if os.path.isdir(root) else []
            return web.json_response({"projects": names})
        except Exception as exc:  # a folder removed while listing
            return web.json_response({"error": str(exc), "projects": []}, status=400)
except Exception as e:  # outside ComfyUI
    print(f"[LC AV] story routes not registered: {e}")


NODE_CLASS_MAPPINGS = {NODE_ID: LCH3Story}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: NODE_DISPLAY_NAME}
