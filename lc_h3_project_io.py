"""
LC H3 project export / import (no node)
---------------------------------------
📦 Export and 📥 Import on LC H3 Story: a whole movie project as one zip, to carry it to another computer.

A project folder (output/LC_H3/<project>/) only points at most of its pictures: the cast, outfit, place, prop and
start-frame pictures live in ComfyUI's input folder (input/lc_h3_refs/...). Export packs the folder plus every
picture, video and sound its files point at; Import puts them back on the other machine and rewrites the paths to
that machine's folders.

  GET  /lc_h3/story/export?project=NAME&renders=0|1   -> <project>.zip (renders = takes/ and the finished movie)
  POST /lc_h3/story/import  multipart: file, name?, overwrite?  -> {project, pictures, renamed, missing}
                                                                 409 {exists, suggest} when the name is taken

Zip layout: lc_export.json (manifest), project/<files>, input/<rel path>, output/<rel path>.
"""

from __future__ import annotations

import asyncio
import datetime
import filecmp
import json
import os
import re
import shutil
import tempfile
import zipfile

import folder_paths


def _root() -> str:
    return os.path.join(folder_paths.get_output_directory(), "LC_H3")


def _name(project: str) -> str:
    """A project name as its folder name (the same rule as project_dir), "" for no usable name."""
    safe = re.sub(r"[^A-Za-z0-9 _.-]", "_", (project or "").strip())
    return safe if safe.strip(" .") else ""


def _folder(project: str, any_file: bool = True) -> str | None:
    """The project's folder when it holds a story, a plan or a project file."""
    name = _name(project)
    path = os.path.join(_root(), name)
    if not name:
        return None
    files = ("story.json", "plan.json", "project.json", os.path.join("assets", "assets.json"))
    return path if any(os.path.isfile(os.path.join(path, f)) for f in files) else None


FORMAT = "lc_h3_project"
MEDIA = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".mp4", ".webm", ".mov", ".mkv", ".wav", ".mp3", ".flac",
         ".ogg", ".m4a", ".safetensors", ".npy", ".pt", ".latent")
STORED = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov", ".mkv", ".mp3", ".ogg", ".m4a", ".flac")
_EXPORTS = os.path.join(tempfile.gettempdir(), "lc_h3_exports")


class NameTaken(Exception):
    def __init__(self, name: str, suggest: str):
        super().__init__(f"A project called '{name}' already exists.")
        self.name, self.suggest = name, suggest


def _dirs() -> dict:
    return {"input": os.path.realpath(folder_paths.get_input_directory()),
            "output": os.path.realpath(folder_paths.get_output_directory())}


def _inside(path: str, base: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), base]) == base
    except ValueError:  # another drive
        return False


def _rel(path: str, base: str) -> str:
    return os.path.relpath(os.path.realpath(path), base).replace("\\", "/")


def _renders(rel: str) -> bool:
    """takes/ (every render, latent and snapshot) and the finished movie at the top of the folder."""
    first = rel.split("/", 1)[0]
    return first == "takes" or ("/" not in rel and rel.lower().endswith((".mp4", ".webm", ".mov", ".mkv")))


def _json_files(folder: str, renders: bool) -> list:
    out = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            rel = _rel(p, os.path.realpath(folder))
            if f.lower().endswith(".json") and (renders or not _renders(rel)):
                out.append(p)
    return out


def _refs(data, dirs: dict, found: dict) -> None:
    """Every file a project's JSON points at, as found[(base, rel)] = absolute path. Three shapes: a picture slot
    {filename, subfolder, type}, a card's input-relative "file", and any absolute path inside input / output."""
    if isinstance(data, dict):
        fn = data.get("filename")
        if isinstance(fn, str) and fn and not os.path.isabs(fn):
            base = "output" if data.get("type") == "output" else "input"
            sub = str(data.get("subfolder") or "").strip("/\\")
            p = os.path.join(dirs[base], sub, fn)
            if os.path.isfile(p) and _inside(p, dirs[base]):
                found[(base, _rel(p, dirs[base]))] = p
        for v in data.values():
            _refs(v, dirs, found)
    elif isinstance(data, list):
        for v in data:
            _refs(v, dirs, found)
    elif isinstance(data, str) and data.lower().endswith(MEDIA) and len(data) < 1024:
        cands = [data] if os.path.isabs(data) else [os.path.join(dirs["input"], data), os.path.join(dirs["output"], data)]
        for p in cands:
            if os.path.isfile(p):
                for base in ("input", "output"):
                    if _inside(p, dirs[base]):
                        found[(base, _rel(p, dirs[base]))] = p
                        break
                break


def export_zip(project: str, renders: bool = False) -> tuple[str, str]:
    """(zip path, download name)."""
    folder = _folder(project, any_file=True)
    if not folder:
        raise ValueError(f"No project called '{project}'.")
    name = os.path.basename(folder)
    rfolder = os.path.realpath(folder)
    dirs = _dirs()
    found: dict = {}
    for jf in _json_files(folder, renders):
        try:
            with open(jf, encoding="utf-8") as f:
                _refs(json.load(f), dirs, found)
        except Exception:
            continue
    # a file inside the project folder travels with the folder itself
    extra = {k: v for k, v in found.items() if not _inside(v, rfolder)}
    os.makedirs(_EXPORTS, exist_ok=True)
    for old in os.listdir(_EXPORTS):  # one export at a time is plenty: clear the last one
        try:
            os.remove(os.path.join(_EXPORTS, old))
        except OSError:
            pass
    out = os.path.join(_EXPORTS, f"{name}.zip")
    files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for root, _dirs_, fs in os.walk(folder):
            for f in fs:
                p = os.path.join(root, f)
                rel = _rel(p, rfolder)
                if (not renders and _renders(rel)) or f.endswith(".tmp"):
                    continue
                z.write(p, "project/" + rel, compress_type=zipfile.ZIP_STORED if f.lower().endswith(STORED) else None)
                files += 1
        for (base, rel), p in sorted(extra.items()):
            z.write(p, f"{base}/{rel}", compress_type=zipfile.ZIP_STORED if p.lower().endswith(STORED) else None)
        z.writestr("lc_export.json", json.dumps({
            "format": FORMAT, "version": 1, "project": name, "renders": bool(renders),
            "made": datetime.datetime.now().isoformat(timespec="seconds"),
            "input_dir": dirs["input"], "output_dir": dirs["output"],
            "project_files": files, "pictures": sorted(f"{b}/{r}" for b, r in extra)}, indent=1, ensure_ascii=False))
    print(f"[LC AV] exported {name}: {files} project files + {len(extra)} pictures -> {out}")
    return out, f"{name}.zip"


def _free_name(name: str) -> str:
    k = 2
    while os.path.exists(os.path.join(_root(), f"{name} {k}")):
        k += 1
    return f"{name} {k}"


def _safe_member(m: str) -> str | None:
    """A zip entry as a clean relative path under one of the three tops, or None (absolute, .., drive, odd top)."""
    m = m.replace("\\", "/")
    if m.startswith("/") or re.match(r"^[A-Za-z]:", m) or any(p in ("..", "") for p in m.rstrip("/").split("/")):
        return None
    return m if m == "lc_export.json" or m.split("/", 1)[0] in ("project", "input", "output") else None


def _fix_paths(data, rename: dict, old_dirs: dict, dirs: dict):
    """Rewrite one JSON value for this machine: renamed pictures and the old machine's input / output folders."""
    if isinstance(data, dict):
        fn, sub = data.get("filename"), str(data.get("subfolder") or "").strip("/\\").replace("\\", "/")
        if isinstance(fn, str) and fn:
            base = "output" if data.get("type") == "output" else "input"
            key = f"{base}/{sub}/{fn}" if sub else f"{base}/{fn}"
            if key in rename:
                data["filename"] = rename[key].rsplit("/", 1)[-1]
        return {k: _fix_paths(v, rename, old_dirs, dirs) for k, v in data.items()}
    if isinstance(data, list):
        return [_fix_paths(v, rename, old_dirs, dirs) for v in data]
    if isinstance(data, str) and data.lower().endswith(MEDIA):
        s = data.replace("\\", "/")
        for base in ("input", "output"):
            old = (old_dirs.get(base) or "").replace("\\", "/").rstrip("/")
            if old and s.lower().startswith(old.lower() + "/"):
                rel = s[len(old) + 1:]
                rel = rename.get(f"{base}/{rel}", f"{base}/{rel}").split("/", 1)[1]
                return os.path.join(dirs[base], *rel.split("/"))
        if not os.path.isabs(data) and f"input/{s}" in rename:  # a card's input-relative file
            return rename[f"input/{s}"].split("/", 1)[1]
    return data


def import_zip(path: str, name: str = "", overwrite: bool = False) -> dict:
    dirs = _dirs()
    with zipfile.ZipFile(path) as z:
        try:
            man = json.loads(z.read("lc_export.json").decode("utf-8"))
        except KeyError:
            raise ValueError("This zip isn't an LC H3 project export (no lc_export.json).")
        if man.get("format") != FORMAT:
            raise ValueError("This zip isn't an LC H3 project export.")
        members = []
        for info in z.infolist():
            if info.is_dir():
                continue
            m = _safe_member(info.filename)
            if not m:
                raise ValueError(f"Unsafe path in the zip: {info.filename}")
            members.append((m, info))
        need = sum(i.file_size for _, i in members)
        free = shutil.disk_usage(_root() if os.path.isdir(_root()) else dirs["output"]).free
        if need > free - 512 * 1024 * 1024:
            raise ValueError(f"Not enough disk space: the project needs {need / 2**30:.1f} GB.")
        target = _name(name or man.get("project") or "")
        if not target or len(target) > 80:
            raise ValueError("The project needs a short folder name.")
        folder = os.path.join(_root(), target)
        if os.path.exists(folder):
            if not overwrite:
                raise NameTaken(target, _free_name(target))
            keep = os.path.join(_root(), "_replaced", f"{target} {datetime.datetime.now():%Y-%m-%d %H%M%S}")
            os.makedirs(os.path.dirname(keep), exist_ok=True)
            shutil.move(folder, keep)  # never deleted: an overwrite parks the old folder in output/LC_H3/_replaced/
        os.makedirs(folder, exist_ok=True)
        rename, pictures, same = {}, 0, 0
        for m, info in members:
            top, rel = m.split("/", 1) if "/" in m else (m, "")
            if top == "project":
                dst = os.path.join(folder, *rel.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(info) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
            elif top in ("input", "output"):
                dst = os.path.join(dirs[top], *rel.split("/"))
                if not _inside(dst, dirs[top]):
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = dst + ".lcimport"
                with z.open(info) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                if os.path.exists(dst):
                    if filecmp.cmp(tmp, dst, shallow=False):  # already here, the same bytes
                        os.remove(tmp)
                        same += 1
                        continue
                    stem, ext = os.path.splitext(dst)  # a different picture by that name stays: this one gets a new name
                    k, new = 2, f"{stem}_imported{ext}"
                    while os.path.exists(new) and not filecmp.cmp(tmp, new, shallow=False):
                        new, k = f"{stem}_imported{k}{ext}", k + 1
                    if os.path.exists(new):  # imported before: the same bytes are already there under that name
                        os.remove(tmp)
                        same += 1
                    else:
                        os.replace(tmp, new)
                        pictures += 1
                    rename[m] = f"{top}/{_rel(new, dirs[top])}"
                    continue
                else:
                    os.replace(tmp, dst)
                pictures += 1
    old_dirs = {"input": man.get("input_dir", ""), "output": man.get("output_dir", "")}
    for jf in _json_files(folder, True):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        data = _fix_paths(data, rename, old_dirs, dirs)
        rel = _rel(jf, os.path.realpath(folder))
        if isinstance(data, dict) and rel in ("project.json", "story.json", "plan.json", "assets/assets.json") \
                and "project" in data:
            data["project"] = target  # the folder's own name (a copied project kept the old one)
        with open(jf + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(jf + ".tmp", jf)
    missing = []
    for p in man.get("pictures", []):
        base = p.split("/", 1)[0]
        r = rename.get(p, p).split("/", 1)[1]
        if not os.path.isfile(os.path.join(dirs[base], *r.split("/"))):
            missing.append(p)
    print(f"[LC AV] imported {target}: {pictures} new pictures, {same} already here, {len(rename)} renamed")
    return {"project": target, "pictures": pictures, "already_here": same,
            "renamed": [f"{a} -> {b}" for a, b in rename.items()], "missing": missing,
            "from": man.get("project"), "renders": man.get("renders", False)}


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/lc_h3/story/export")
    async def _export_route(request):
        try:
            renders = request.query.get("renders", "0") in ("1", "true", "yes")
            path, fname = await asyncio.to_thread(export_zip, request.query.get("project", ""), renders)
        except Exception as exc:
            print(f"[LC AV] movie/export: {exc}")
            return web.json_response({"error": str(exc)}, status=400)
        quoted = fname.encode("ascii", "replace").decode().replace('"', "_")
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{quoted}"',
                                               "Cache-Control": "no-store"})

    @PromptServer.instance.routes.post("/lc_h3/story/import")
    async def _import_route(request):
        tmp = None
        try:
            fields = {}
            reader = await request.multipart()  # streamed to disk: a project with its renders can be gigabytes
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "file":
                    os.makedirs(_EXPORTS, exist_ok=True)
                    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=_EXPORTS)
                    with os.fdopen(fd, "wb") as out:
                        while True:
                            chunk = await part.read_chunk(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                else:
                    fields[part.name] = (await part.text()).strip()
            if not tmp:
                return web.json_response({"error": "No file was sent."}, status=400)
            if not zipfile.is_zipfile(tmp):
                return web.json_response({"error": "That file isn't a zip."}, status=400)
            res = await asyncio.to_thread(import_zip, tmp, fields.get("name", ""),
                                          fields.get("overwrite", "") in ("1", "true", "yes"))
            return web.json_response(res)
        except NameTaken as exc:
            return web.json_response({"error": str(exc), "exists": exc.name, "suggest": exc.suggest}, status=409)
        except Exception as exc:
            print(f"[LC AV] movie/import: {exc}")
            return web.json_response({"error": str(exc)}, status=400)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
except Exception as e:  # running outside ComfyUI (tests)
    print(f"[LC AV] project export / import routes not registered: {e}")
