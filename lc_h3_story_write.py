"""
LC H3 Story ✨ Write the story (no node): one LLM call turns a short pitch into the Story node's cards (look, cast,
places, shots). After that it is all plain text on the cards; nothing runs the model again unless ✨ is pressed again.

The model is any text GGUF under models/LLM (the LC Vision models work; the vision part is not needed here), picked
on the node. It loads for the one answer and unloads right after, so the video model gets the card back.

  GET  /lc_h3/story/models                         -> {"models": [...], "default": name}
  POST /lc_h3/story/write {pitch, model, story, max_shot_seconds} -> {"story": {...}} (pictures and voices of cast
       and places with the same name are kept)
"""
from __future__ import annotations

import gc
import json
import os
import re
import threading
import uuid

import folder_paths

from .lc_h3_story import FRAMINGS, MOVES, WHO, WORDS_PER_SECOND

N_CTX = 16384
MAX_TOKENS = 6000
_LOCK = threading.Lock()

SYSTEM = f"""You write short films for an AI video model (MiniMax H3) as JSON cards. Reply with one JSON object only.

Shape:
{{"look": "the visual style of the whole film, one sentence",
 "cast": [{{"name": "NAME", "who": "woman", "look": "in her twenties, wavy blue shoulder-length hair, white t-shirt, blue jeans"}}],
 "places": [{{"name": "the cafe", "description": "what is there: the room, the furniture, the light", "sound": "soft cafe chatter, cups clinking"}}],
 "shots": [{{"framing": "wide shot", "subject": "everyone", "move": "still", "join": "cut", "where": "the cafe",
            "action": "what happens before the line", "speaker": "NAME", "line": "the spoken line, word for word", "after": "what happens after the line", "seconds": ""}}]}}

Rules:
- cast: every person who appears, with the names the pitch gives them (NAME above is only a placeholder). Make up a short first name only for someone the pitch does not name. "name" starts with a capital letter. "who" is one of {", ".join(WHO)}.
- cast look: ONE clause, no full stops: age, hair, clothes (what they wear in the whole film). These words are what the video model follows for the person, so be concrete. Never write "detailed skin" or quality words.
- places: every place a shot happens in. "name" starts with "the". "sound" is only things you can hear: rain on the glass, footsteps, a door, cups clinking. Never write who speaks or does not speak, never "no voices", never music: naming speech makes the video model speak.
- framing is one of: {", ".join(f for f in FRAMINGS if f != "custom")}.
- move is one of: {", ".join(MOVES)}. Mostly "still".
- subject is "everyone" or one cast name.
- join: "cut" (a new shot) or "roll" (no cut: the camera carries straight on from the shot before). The first shot is always "cut".
  Use "roll" only between neighbouring framings (wide shot -> two-shot, medium shot -> close-up). From a wide shot to a close-up always "cut".
- where: one of the place names. Leave it "" when the shot stays in the place of the shot before.
- action: short and physical, in the present tense, using the cast names ("NAME sets her cup down."). One or two actions per shot.
- At most ONE spoken line per shot, in "line" (no quote marks), said by "speaker" (a cast name). No line: "speaker" and "line" are "".
  Keep lines natural and short (under 20 words). Never describe mouth movement, eating or drinking in the same shot as a line.
- after: optional, a short reaction after the line, or "".
- seconds: "" (the length is worked out from the line) unless the shot needs a long silent action: then a number of seconds as a string.
  A line takes about {WORDS_PER_SECOND:g} words a second. No shot longer than {{max_s:g}} seconds.
- Open with a wide shot that shows where everyone is. A close-up of one person is "subject": that person.
- shots: one card per shot, in order, telling the WHOLE pitch from the first moment to the ending. If the pitch asks for a number of shots or a length, keep to it; otherwise 4 to 8 shots.
- Follow the pitch: its people, places, mood and ending.
"""


def list_models() -> list:
    """Text GGUF files under models/LLM (the vision projectors left out), as paths relative to their LLM folder."""
    try:
        roots = list(folder_paths.get_folder_paths("LLM"))
    except Exception:
        roots = []
    roots = roots or [os.path.join(folder_paths.models_dir, "LLM")]
    out = []
    for root in roots:
        for d, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".gguf") and "mmproj" not in f.lower():
                    out.append(os.path.relpath(os.path.join(d, f), root).replace("\\", "/"))
    return sorted(set(out), key=str.lower)


def default_model(models: list) -> str:
    """A good writer that fits most cards: Qwen3-VL 8B, then any 8B, then Qwen3 4B Q8, then the first."""
    for want in (r"qwen3-vl-8b.*q5_k_m", r"qwen3-vl-8b.*q8", r"8b.*q8", r"8b", r"4b.*q8"):
        hit = next((m for m in models if re.search(want, m, re.I)), None)
        if hit:
            return hit
    return models[0] if models else ""


def _path(model: str) -> str:
    if model not in list_models():
        raise FileNotFoundError(f"[LC H3] Story: the model '{model}' is not under models/LLM.")
    try:
        roots = list(folder_paths.get_folder_paths("LLM"))
    except Exception:
        roots = []
    for root in roots or [os.path.join(folder_paths.models_dir, "LLM")]:
        p = os.path.join(root, model)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"[LC H3] Story: the model '{model}' is not under models/LLM.")


def _not_while_rendering() -> None:
    """A route runs beside the queue: loading a model (and freeing VRAM for it) mid-render pulled the video model out
    from under the sampler (CUDA illegal memory access, 2026-10-28). Wait for the queue instead."""
    try:
        from server import PromptServer

        running, _ = PromptServer.instance.prompt_queue.get_current_queue()
    except Exception:
        return
    if running:
        raise ValueError("A render is running. Try again when it's done: loading this model now would push the "
                         "video model out of VRAM mid-render.")


def _make_room(path: str) -> None:
    """Ask ComfyUI to drop the image / video model if the writer won't fit next to it."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        need = os.path.getsize(path) / 1024 ** 3 + 2.5  # the file plus the 16k context
        if torch.cuda.mem_get_info()[0] / 1024 ** 3 < need:
            import comfy.model_management as mm

            mm.unload_all_models()
            mm.soft_empty_cache()
    except Exception as exc:
        print(f"[LC H3] Story: could not make room in VRAM ({exc}).")


def mmproj_for(path: str) -> str:
    """The vision projector next to a model (the one sharing the longest filename start), or ""."""
    d, f = os.path.dirname(path), os.path.basename(path).lower()
    cands = [x for x in os.listdir(d) if x.lower().endswith(".gguf") and "mmproj" in x.lower()]
    if not cands:
        return ""
    share = lambda x: next((k for k, (a, b) in enumerate(zip(f, x.lower())) if a != b), min(len(f), len(x)))
    return os.path.join(d, max(sorted(cands), key=share))


def load_llm(path: str, vision: bool = False, n_ctx: int = N_CTX):
    """A llama.cpp model for one job. vision: with its mmproj, through LC Vision's own loader when that pack is in
    (it knows Qwen3-VL and Gemma 4), else the Qwen3-VL handler."""
    import inspect
    import sys

    from llama_cpp import Llama

    _not_while_rendering()
    _make_room(path)
    if vision:
        mm = mmproj_for(path)
        if not mm:
            raise ValueError(f"[LC H3] Story: {os.path.basename(path)} has no vision projector (mmproj) next to it. "
                             "Pick a vision model in the ✨ dropdown, e.g. Qwen3-VL 8B.")
        loader = next((m for n, m in list(sys.modules.items()) if n.endswith(".lc_vision_loader") and hasattr(m, "build_llama")), None)
        if loader:
            llm, _ = loader.build_llama(loader.LCVisionBuildParams(
                model_path=path, mmproj_path=mm, device_kind="cuda", n_gpu_layers=-1, n_ctx=n_ctx, n_batch=2048,
                image_min_tokens=256, image_max_tokens=1024, batch_max_tokens=1024, verbose=False))
            return llm
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler

        return Llama(model_path=path, chat_handler=Qwen3VLChatHandler(mmproj_path=mm, verbose=False), n_ctx=n_ctx,
                     n_gpu_layers=-1, n_batch=2048, n_ubatch=2048, verbose=False)
    kw = dict(model_path=path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    params = inspect.signature(Llama.__init__).parameters
    if "flash_attn_type" in params:
        kw["flash_attn_type"] = 1
    elif "flash_attn" in params:
        kw["flash_attn"] = True
    if "type_k" in params and "type_v" in params:
        kw["type_k"] = kw["type_v"] = 8
    return Llama(**kw)


def free_llm(llm) -> None:
    try:
        llm.close()
    except Exception:
        pass
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _ask(path: str, pitch: str, max_s: float) -> str:
    print(f"[LC H3] Story: ✨ writing with {os.path.basename(path)}")
    llm = load_llm(path)
    try:
        want = _wanted_shots(pitch)
        msgs = [{"role": "system", "content": SYSTEM.replace("{max_s:g}", f"{max_s:g}")},
                {"role": "user", "content": f"Pitch: {pitch.strip()}\n\nWrite the cards: the whole story from start to end, "
                                            f"{f'{want} shots' if want else '4 to 8 shots'}. /no_think"}]
        text = ""
        for _ in range(2):  # one more try when it stops short (a small model copies the one example shot)
            res = llm.create_chat_completion(messages=msgs, max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.9,
                                             response_format={"type": "json_object"})
            text = res["choices"][0]["message"]["content"] or ""
            try:
                got = len(_json(text).get("shots") or [])
            except Exception:
                got = 0
            need = want or 3
            if got >= min(need, want or need):
                break
            print(f"[LC H3] Story: ✨ wrote {got} shots, asking for {need}")
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"That is only {got} shots. Write the full JSON again with all {want or 'the'} shots "
                                                 f"that tell the whole pitch, start to end. /no_think"}]
        return text
    finally:
        free_llm(llm)
        print("[LC H3] Story: writer unloaded")


def _wanted_shots(pitch: str) -> int:
    m = re.search(r"\b(\d{1,2})\s*(?:shots?|scenes?|cuts?)\b", str(pitch or ""), re.I)
    return int(m.group(1)) if m else 0


def _json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("[LC H3] Story: the model did not answer with cards. Try again, or pick a bigger model.")
    return json.loads(m.group(0))


def _dicts(items) -> list:
    """The cards of one kind: a small model sometimes writes a list of names, or a string, instead of objects."""
    return [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []


def _clean(raw: dict, old: dict, max_s: float) -> dict:
    """The model's cards made safe for the node: known framings / moves only, ids, one-clause looks, real names, and
    the pictures and voices already on cards of the same name kept."""
    s = lambda v: re.sub(r"\s+", " ", str(v or "")).strip()
    uid = lambda p: p + uuid.uuid4().hex[:6]
    old_cast = {s(c.get("name")).lower(): c for c in old.get("cast", [])}
    old_places = {s(p.get("name")).lower(): p for p in old.get("places", [])}

    cast = []
    for c in _dicts(raw.get("cast")):
        name = s(c.get("name"))
        if not name:
            continue
        name = name[:1].upper() + name[1:]
        o = old_cast.get(name.lower(), {})
        who = s(c.get("who")).lower()
        cast.append({"id": o.get("id") or uid("c"), "name": name, "who": who if who in WHO else "person",
                     "look": s(c.get("look")).replace(". ", ", ").rstrip("."), "picture": o.get("picture"), "voice": o.get("voice")})
    names = [c["name"] for c in cast]

    places = []
    for p in _dicts(raw.get("places")):
        name = s(p.get("name"))
        if not name:
            continue
        o = old_places.get(name.lower(), {})
        places.append({**{k: o[k] for k in ("picture", "panorama", "front", "anchors", "close_view") if k in o},
                       "id": o.get("id") or uid("p"), "name": name, "description": s(p.get("description")),
                       "sound": s(p.get("sound")), "picture": o.get("picture"), "anchors": o.get("anchors") or []})
    pnames = {p["name"].lower(): p["name"] for p in places}

    shots = []
    for k, sh in enumerate(_dicts(raw.get("shots"))):
        framing = s(sh.get("framing")).lower()
        move = s(sh.get("move")).lower()
        subject = s(sh.get("subject"))
        speaker = s(sh.get("speaker"))
        line = s(sh.get("line")).strip("\"“”")
        where = pnames.get(s(sh.get("where")).lower(), "")
        sec = s(sh.get("seconds"))
        try:
            sec = f"{min(float(sec), max_s):g}" if sec else ""
        except ValueError:
            sec = ""
        shots.append({"id": uid("s"), "framing": framing if framing in FRAMINGS else "medium shot",
                      "subject": subject if subject in names else "everyone", "move": move if move in MOVES else "still",
                      "join": "roll" if k and s(sh.get("join")).lower() == "roll" else "cut", "where": where,
                      "action": s(sh.get("action")), "speaker": speaker if speaker in names and line else "",
                      "line": line if speaker in names else "", "after": s(sh.get("after")), "seconds": sec})
    if not shots:
        raise ValueError("[LC H3] Story: the model wrote no shots. Try again with a longer pitch, or a bigger model.")
    return {**{k: v for k, v in old.items() if k not in ("cast", "places", "shots", "look", "imported")},
            "version": 1, "look": s(raw.get("look")), "cast": cast, "places": places, "shots": shots}


def write_story(pitch: str, model: str, old: dict | None = None, max_s: float = 20.0) -> dict:
    _not_while_rendering()
    if not str(pitch or "").strip():
        raise ValueError("[LC H3] Story: write a pitch first (a few sentences: who, where, what happens).")
    with _LOCK:  # one writer at a time
        raw = _ask(_path(model), pitch, max_s)
    try:
        cards = _json(raw)
    except ValueError as exc:
        print(f"[LC H3] Story: unreadable answer: {raw[:400]}")
        raise ValueError(str(exc)) from exc
    return _clean(cards, old or {}, max_s)


try:
    import asyncio

    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/lc_h3/story/models")
    async def _models_route(request):
        models = list_models()
        return web.json_response({"models": models, "default": default_model(models)})

    @PromptServer.instance.routes.post("/lc_h3/story/write")
    async def _write_route(request):
        try:
            body = await request.json()
            story = await asyncio.get_running_loop().run_in_executor(
                None, write_story, body.get("pitch", ""), body.get("model", ""), body.get("story") or {},
                float(body.get("max_shot_seconds") or 20.0))
            return web.json_response({"story": story})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
except Exception as e:  # outside ComfyUI
    print(f"[LC AV] story write routes not registered: {e}")


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
