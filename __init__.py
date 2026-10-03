"""
ComfyUI_LC_AV_nodes — LC Audio_Video Nodes by lonecatone23

Companion pack to ComfyUI_LC123_nodes. Separate repo, separate Registry name.

https://github.com/lonecatone23
https://ko-fi.com/lonecatone
"""

import os as _os

_PACK_DIR = _os.path.dirname(_os.path.abspath(__file__))
_nested = _os.path.join(_PACK_DIR, "ComfyUI_LC_AV_nodes", "__init__.py")
if _os.path.isfile(_nested):
    print(
        "[LC AV] WARNING: nested pack folder detected. "
        f"{_nested} will be ignored. Unzip so __init__.py sits in {_PACK_DIR}"
    )

NODE_CLASS_MAPPINGS = {}
_FAILED = []  # modules that did not load (shown in full, and counted on the banner line)
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load(module_name: str) -> None:
    """Import a submodule and merge its mappings. Log and skip on failure."""
    import importlib
    import traceback

    try:
        mod = importlib.import_module(f".{module_name}", __name__)
        maps = getattr(mod, "NODE_CLASS_MAPPINGS", None) or {}
        disp = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", None) or {}
        NODE_CLASS_MAPPINGS.update(maps)
        NODE_DISPLAY_NAME_MAPPINGS.update(disp)
    except Exception as e:
        _FAILED.append(module_name)
        print(f"[LC AV] ! failed to load {module_name}: {e}")
        traceback.print_exc()


_load("lc_av_pipe")
_load("lc_create_video")
_load("lc_save_video")
_load("lc_final_frame")
_load("lc_load_audio")
_load("lc_audio_duration")
_load("lc_audio_crop")
_load("lc_audio_channel")
_load("lc_audio_separate")
_load("lc_audio_eq")
_load("lc_audio_volume")
_load("lc_vae_decode")
_load("lc_latent_size")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:  # the Lonecat banner (first LC pack to load) or this pack's one line under it
    from .lc_banner import pack_version as _pv, show as _banner

    _banner("LC AV", _pv(_PACK_DIR), len(NODE_CLASS_MAPPINGS), len(_FAILED))
except Exception as _banner_e:
    print(f"[LC AV] {len(NODE_CLASS_MAPPINGS)} nodes ({_banner_e})")
