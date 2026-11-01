"""
LC H3 Story 🔗 join check (no node): a keep-rolling shot starts from wherever the shot before really ended, which is
not always what its card says ("she sits down" when she is already seated). The check looks at the last frames of the
shot before and suggests a new opening for the card's action. The card shows it with ✓ / ✗; nothing changes unless
you press ✓.

Two steps with the same model, like LC Vision Redirect: it first only looks (a small vision model describes well but
rewrites lazily while it looks), then rewrites from its own sentence as a text task.

  POST /lc_h3/story/join {story, shot, model, max_shot_seconds} -> {end, suggestion, rejected, tail_t, prev}
"""
from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import threading

from .lc_h3_segment import _seg_path, project_dir
from .lc_h3_story_write import _not_while_rendering, _path, free_llm, load_llm

_LOCK = threading.Lock()

LOOK = ("You are a film continuity supervisor. These frames, in time order, are the last moments of a shot. In one or two "
        "sentences, say exactly how the shot ends, judged by the LAST frame: where each person is, standing or sitting or "
        "walking, their pose and hands, which way they face, what they hold, and how the camera frames them. Call people by "
        "the names in the cast list. Facts only.")
REWRITE = ("You fix continuity on a shot card. The shot continues straight on from the END STATE with no cut, so its first "
           "moment must start from exactly that state.\n"
           "Read the PLANNED action. An action that has already happened by the end state, or is impossible from it (sitting "
           "down while already seated, opening a door already open, walking in while already inside), is replaced by one "
           "that gets from the end state toward the rest of the plan. Actions that already fit stay word for word. If the "
           "plan already fits, output it unchanged.\n"
           "Keep the order of events, keep every name and who does what, keep any \"At X seconds\" timestamps exactly, keep "
           "it about as short as the plan, and describe actions only (never appearance or clothes).\n"
           "Example. END STATE: Nova sits at the table holding her cup. PLANNED: Nova walks to the table and sits down. She "
           "sips her coffee. REWRITE: Nova sets her cup down and leans back. She sips her coffee.\n"
           "Output only the rewritten action.")


def _tail(project: str, prev_id: str) -> tuple:
    """(tail mp4 of the shot before's last segment, its mtime) from the saved plan, or (None, 0)."""
    import json

    folder = project_dir(project)
    try:
        with open(os.path.join(folder, "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
    except Exception:
        return None, 0
    idx = [k for k, s in enumerate(plan.get("shots", []), start=1) if s.get("shot_id") == prev_id]
    if not idx:
        return None, 0
    p = _seg_path(folder, max(idx), "_tail.mp4")
    return (p, os.path.getmtime(p)) if os.path.exists(p) else (None, 0)


def _frames(path: str, picks: int = 3, side: int = 768) -> list:
    """Base64 JPEGs spread over the tail, the last frame included."""
    from PIL import Image

    from .lc_save_video import _ffmpeg_bin

    out = subprocess.run([_ffmpeg_bin(), "-v", "error", "-i", path, "-vf", f"scale={side}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True).stdout
    pngs, k = [], 0
    while True:  # split the PNG stream
        a = out.find(b"\x89PNG", k)
        if a < 0:
            break
        b = out.find(b"\x89PNG", a + 8)
        pngs.append(out[a:b if b > 0 else len(out)])
        k = a + 8
    if not pngs:
        return []
    pick = sorted({round(x * (len(pngs) - 1) / max(1, picks - 1)) for x in range(picks)})
    res = []
    for i in pick:
        buf = io.BytesIO()
        Image.open(io.BytesIO(pngs[i])).convert("RGB").save(buf, "JPEG", quality=90)
        res.append(base64.b64encode(buf.getvalue()).decode())
    return res


def _names(story: dict) -> list:
    return [str(c.get("name") or "").strip() for c in story.get("cast", []) if str(c.get("name") or "").strip()]


def _problems(plan: str, new: str, names: list) -> list:
    out = []
    if not new:
        return ["empty"]
    for n in names:
        if re.search(rf"\b{re.escape(n)}\b", plan) and not re.search(rf"\b{re.escape(n)}\b", new):
            out.append(f"dropped {n}")
    stamps = lambda t: re.findall(r"\bAt (\d+(?:\.\d+)?) seconds\b", t)
    if stamps(plan) and stamps(plan) != stamps(new):
        out.append("changed the timestamps")
    if len(new) > len(plan) * 1.8 + 80:
        out.append("grew too long")
    if re.search(r"[\"“]", new) and not re.search(r"[\"“]", plan):
        out.append("added a spoken line")
    return out


def check(story: dict, shot_i: int, model: str) -> dict:
    _not_while_rendering()
    shots = [s for s in story.get("shots", []) if isinstance(s, dict)]
    if not (0 < shot_i < len(shots)):
        raise ValueError("[LC H3] Story: the first shot has nothing before it to join to.")
    shot, prev = shots[shot_i], shots[shot_i - 1]
    tail, tail_t = _tail(str(story.get("project") or ""), prev.get("id"))
    if not tail:
        raise ValueError(f"[LC H3] Story: shot {shot_i} isn't made yet, so there is nothing to look at. Make it first.")
    frames = _frames(tail)
    if not frames:
        raise RuntimeError("[LC H3] Story: could not read the end of the shot before.")
    names = _names(story)
    cast = "; ".join(f"{c['name']}: a {c.get('who') or 'person'}, {c.get('look') or ''}".strip(", ") for c in story.get("cast", []) if c.get("name"))
    planned = str(shot.get("action") or "").strip() or "The moment carries on."
    with _LOCK:
        llm = load_llm(_path(model), vision=True, n_ctx=8192)
        try:
            def ask(system, content, tokens=400):
                res = llm.create_chat_completion(messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                                                 max_tokens=tokens, temperature=0.3, top_p=0.9, repeat_penalty=1.05)
                text = (res["choices"][0]["message"]["content"] or "").strip()
                return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

            content = [{"type": "text", "text": f"/no_think\nCast: {cast}"}]
            for k, b in enumerate(frames, start=1):
                content += [{"type": "text", "text": f"\n<Frame {k} of {len(frames)}>"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}]
            end = ask(LOOK, content)
            new = ask(REWRITE, f"/no_think\nEND STATE: {end}\nPLANNED: {planned}\nREWRITE:", 500).split("REWRITE:", 1)[-1].strip().strip('"')
        finally:
            free_llm(llm)
    bad = _problems(planned, new, names)
    same = re.sub(r"\W+", " ", new).strip().lower() == re.sub(r"\W+", " ", planned).strip().lower()
    print(f"[LC H3] Story: 🔗 shot {shot_i} ends: {end}" + (f" | rejected: {', '.join(bad)}" if bad else ""))
    return {"end": end, "suggestion": "" if bad or same else new, "same": same and not bad, "rejected": bad,
            "tail_t": tail_t, "prev": shot_i, "from": planned}


try:
    import asyncio

    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/lc_h3/story/join")
    async def _join_route(request):
        try:
            body = await request.json()
            res = await asyncio.get_running_loop().run_in_executor(None, check, body.get("story") or {}, int(body.get("shot", 0)),
                                                                   body.get("model", ""))
            return web.json_response(res)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/lc_h3/story/join_state")
    async def _join_state_route(request):
        """When the shot before each keep-rolling shot was last made (the card flags a check as out of date)."""
        try:
            body = await request.json()
            story = body.get("story") or {}
            shots = [s for s in story.get("shots", []) if isinstance(s, dict)]
            out = {}
            for i, s in enumerate(shots):
                if i and s.get("join") == "roll":
                    out[s.get("id")] = _tail(str(story.get("project") or ""), shots[i - 1].get("id"))[1]
            return web.json_response({"tails": out})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
except Exception as e:  # outside ComfyUI
    print(f"[LC AV] story join routes not registered: {e}")


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
