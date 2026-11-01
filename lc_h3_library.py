"""
LC H3 library (no node): ready-made cast headshots, outfits, props and 360 panoramas for the Story cards.

The pack ships the catalog and small thumbnails (assets/library/); the full pictures come from the online library on the
first pick and stay in input/lc_h3_library/ (a picture in the pack's assets/library/full/ is used first, if there).

  GET  /lc_h3/library?category=cast|outfits|props|360  -> {"items": [...]}, each with its card text and a thumbnail URL
  GET  /lc_h3/library/thumb?stem=...                    -> the thumbnail
  POST /lc_h3/library/get {stem}                        -> {"pic": {...}, "item": {...}}: the full picture (downloaded once)
  POST /lc_h3/library/ensure {story}                    -> {"fetched": n}: downloads the library pictures a story uses
"""
from __future__ import annotations

import json
import os
import re

import folder_paths

PACK = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(PACK, "assets", "library")
BASE_URL = "https://huggingface.co/datasets/Lonecatone23/lc-h3-library/resolve/main/"
CACHE_SUB = "lc_h3_library"
_catalog = {}


def catalog() -> dict:
    if not _catalog:
        try:
            with open(os.path.join(LIB, "catalog.json"), encoding="utf-8") as f:
                _catalog.update(json.load(f))
        except Exception as exc:
            print(f"[LC H3] library: no catalog ({exc}).")
    return _catalog


def _safe_rel(rel: str) -> str:
    """A catalog path as a relative path that can't step out of its folder."""
    rel = str(rel or "").replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/") or re.match(r"^[A-Za-z]:", rel):
        raise ValueError("bad library path")
    return rel


def _local(rel: str) -> str:
    return os.path.join(folder_paths.get_input_directory(), CACHE_SUB, *rel.split("/"))


def fetch(stem: str, announce: bool = False) -> tuple:
    """(pic, item) for a library entry, downloading the full picture the first time."""
    item = catalog().get(stem)
    if not item:
        raise ValueError(f"'{stem}' is not in the library.")
    rel = _safe_rel(item["file"])
    dest = _local(rel)
    if not os.path.isfile(dest):
        shipped = os.path.join(LIB, "full", *rel.split("/"))
        if os.path.isfile(shipped):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            import shutil

            shutil.copy(shipped, dest)
        else:
            from .lc_first_download import download

            download(BASE_URL + rel, dest, f"library {item.get('name') or stem}", cancel=False, announce=announce)
    sub = "/".join([CACHE_SUB] + rel.split("/")[:-1])
    return {"type": "input", "subfolder": sub, "filename": rel.split("/")[-1]}, item


def ensure_pics(data) -> int:
    """Downloads every library picture a story points at that isn't on disk yet (the sample story, a shared one).
    Returns how many came down."""
    by_file = {it.get("file"): stem for stem, it in catalog().items()}
    got = 0
    stack = [data]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack += x
            continue
        if not isinstance(x, dict):
            continue
        sub = str(x.get("subfolder") or "").replace("\\", "/")
        if x.get("type") == "input" and x.get("filename") and (sub == CACHE_SUB or sub.startswith(CACHE_SUB + "/")):
            rel = "/".join(sub.split("/")[1:] + [str(x["filename"])])
            stem = by_file.get(rel)
            if stem and not os.path.isfile(_local(rel)):
                fetch(stem, announce=True)
                got += 1
            continue
        stack += list(x.values())
    return got


try:
    import asyncio

    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/lc_h3/library")
    async def _list(request):
        cat = request.query.get("category", "")
        out = []
        for stem, it in catalog().items():
            if cat and it.get("category") != cat:
                continue
            rel = it.get("file", "")
            try:
                have = os.path.isfile(_local(_safe_rel(rel)))
            except ValueError:
                continue
            out.append({**it, "stem": stem, "have": have, "thumb": f"/lc_h3/library/thumb?stem={stem}"})
        return web.json_response({"items": out})

    @PromptServer.instance.routes.get("/lc_h3/library/thumb")
    async def _thumb(request):
        stem = re.sub(r"[^A-Za-z0-9_-]", "", request.query.get("stem", ""))
        path = os.path.join(LIB, "thumbs", stem + ".webp")
        if not stem or not os.path.isfile(path):
            return web.Response(status=404)
        return web.FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    @PromptServer.instance.routes.post("/lc_h3/library/get")
    async def _get(request):
        try:
            data = await request.json()
            pic, item = await asyncio.get_running_loop().run_in_executor(None, fetch, str(data.get("stem") or ""))
            return web.json_response({"pic": pic, "item": item})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/lc_h3/library/ensure")
    async def _ensure(request):
        try:
            data = await request.json()
            got = await asyncio.get_running_loop().run_in_executor(None, ensure_pics, data.get("story"))
            return web.json_response({"fetched": got})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
except Exception as _e:  # pragma: no cover - outside ComfyUI
    print(f"[LC H3] library routes not registered: {_e}")
