"""
LC H3 anchors (no node): the Story board's 📌 Set anchor. Grabs one frame of a saved segment and puts it in ComfyUI's
input folder (input/lc_h3_refs/anchors/), where the Reference Assembler loads it like any picture you drop in.

  POST /lc_h3/anchor {project, index, time} -> {"file": "lc_h3_refs/anchors/<project>_seg003_2.40s.png"}
"""
from __future__ import annotations

import os
import re
import subprocess

import folder_paths

from .lc_h3_segment import _is_done, _seg_path, project_dir
from .lc_save_video import _ffmpeg_bin


def capture(project: str, index: int, time_s: float) -> str:
    folder = project_dir(project)
    if not _is_done(folder, index):
        raise ValueError(f"Segment {index} of {project} isn't made yet.")
    src = _seg_path(folder, index, ".mp4")
    sub = os.path.join("lc_h3_refs", "anchors")
    dest_dir = os.path.join(folder_paths.get_input_directory(), sub)
    os.makedirs(dest_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.basename(folder))
    name = f"{safe}_seg{index:03d}_{max(0.0, time_s):.2f}s.png"
    out = subprocess.run([_ffmpeg_bin(), "-v", "error", "-y", "-ss", f"{max(0.0, time_s):.3f}", "-i", src, "-frames:v", "1",
                          os.path.join(dest_dir, name)], capture_output=True, timeout=60)  # runs in the route: never hang the server
    if out.returncode or not os.path.exists(os.path.join(dest_dir, name)):
        raise RuntimeError(f"Could not grab that frame: {out.stderr.decode(errors='ignore')[-200:]}")
    return f"{sub}/{name}".replace("\\", "/")


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/lc_h3/anchor")
    async def _anchor_route(request):
        try:
            body = await request.json()
            return web.json_response({"file": capture(body.get("project", ""), int(body.get("index", 0)), float(body.get("time", 0)))})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
except Exception as e:  # outside ComfyUI
    print(f"[LC AV] anchor route not registered: {e}")
