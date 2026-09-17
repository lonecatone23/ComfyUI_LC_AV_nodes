"""Shared metadata helpers for LC Save Video (sidecar + container tags)."""

from __future__ import annotations

import json


# Same type string as LC123 LC Save Metadata so one metadata node feeds
# both LC Save Image and LC Save Video.
META_TYPE = "LC_SAVE_META"
META_TYPE_LEGACY = "LC_SAVE_VIDEO_META"


def txt(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def pipe_get(pipe, *keys):
    if not isinstance(pipe, dict):
        return None
    for k in keys:
        if k in pipe and pipe[k] is not None and pipe[k] != "":
            return pipe[k]
    return None


def as_meta(metadata) -> dict:
    if isinstance(metadata, dict):
        out = dict(metadata)
        out["_type"] = META_TYPE
        return out
    return {"_type": META_TYPE}


def civitai_resources_payload(air: str):
    """Parse a pasted Civitai URN / URL. Do not invent version or file ids."""
    air = txt(air)
    if not air:
        return None
    # Exact paste only. CiviScribe-style: store the string; no lookup.
    return [{"air": air}] if air.lower().startswith("urn:air:") else [{"url": air}]


def build_parameters(meta: dict, width: int = 0, height: int = 0, fps=None, duration=None) -> str:
    positive = txt(meta.get("positive"))
    negative = txt(meta.get("negative"))
    steps = meta.get("steps")
    sampler = txt(meta.get("sampler_name") or meta.get("sampler"))
    scheduler = txt(meta.get("scheduler"))
    cfg = meta.get("cfg")
    seed = meta.get("seed")
    models = txt(meta.get("models"))
    extra = txt(meta.get("extra_params"))
    air = txt(meta.get("civitai_air"))
    denoise = meta.get("denoise")

    lines = []
    if positive:
        lines.append(positive)
    if negative:
        lines.append(f"Negative prompt: {negative}")
    bits = []
    if steps not in (None, "", 0, "0"):
        try:
            bits.append(f"Steps: {int(steps)}")
        except (TypeError, ValueError):
            bits.append(f"Steps: {steps}")
    if sampler:
        bits.append(f"Sampler: {sampler}")
    if scheduler:
        bits.append(f"Schedule type: {scheduler}")
    if cfg not in (None, "", 0, 0.0, "0"):
        try:
            bits.append(f"CFG scale: {float(cfg):g}")
        except (TypeError, ValueError):
            bits.append(f"CFG scale: {cfg}")
    if seed not in (None, "", -1, "-1"):
        try:
            bits.append(f"Seed: {int(seed)}")
        except (TypeError, ValueError):
            bits.append(f"Seed: {seed}")
    if width and height:
        bits.append(f"Size: {int(width)}x{int(height)}")
    if fps not in (None, "", 0, 0.0):
        try:
            bits.append(f"FPS: {float(fps):g}")
        except (TypeError, ValueError):
            pass
    if duration not in (None, "", 0, 0.0):
        try:
            bits.append(f"Duration: {float(duration):g}")
        except (TypeError, ValueError):
            pass
    if models:
        bits.append(f"Model: {models}")
    if denoise not in (None, "", 0, 0.0, "0"):
        try:
            bits.append(f"Denoising strength: {float(denoise):g}")
        except (TypeError, ValueError):
            pass
    if air:
        resources = civitai_resources_payload(air)
        if resources:
            bits.append("Civitai resources: " + json.dumps(resources, separators=(",", ":")))
    if extra:
        bits.append(extra.lstrip(", "))
    if bits:
        lines.append(", ".join(bits))
    return "\n".join(lines).strip()


def sidecar_fields(meta: dict, extra: dict | None = None) -> dict:
    """Same field names as LC Save Image metadata where possible."""
    out = {}
    for key in (
        "positive",
        "negative",
        "steps",
        "sampler",
        "sampler_name",
        "scheduler",
        "cfg",
        "denoise",
        "seed",
        "models",
        "civitai_air",
        "hashes",
        "parameters",
        "width",
        "height",
        "fps",
        "frame_rate",
        "duration",
        "frame_count",
        "sample_rate",
        "total_steps",
    ):
        if key in meta and meta[key] not in (None, ""):
            out[key] = meta[key]
    if extra:
        for k, v in extra.items():
            if v not in (None, ""):
                out[k] = v
    return out
