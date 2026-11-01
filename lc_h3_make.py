"""
LC H3 Make (BETA, working name): the engine for LC H3 Story in one node. It builds the next shot (LC H3 Segment),
samples it (the same Basic Guider + Basic Scheduler + Sampler Custom Advanced the workflows used) and saves it
(LC H3 Segment Save: the takes, the tail the next shot starts from, the joined movie).

The model, VAEs, size, steps, sampler and seed come in on one LC MiniMax H3 Pipe V2. make: all remaining chains
shot after shot like the Segment did (Stop ends it); the Story board's ↻ points it at one shot.
"""
from __future__ import annotations

import json
import os

import comfy.samplers
from comfy_execution.graph import ExecutionBlocker

from .lc_h3_refs import REFS_TYPE
from .lc_h3_segment import PLAN_TYPE, LCH3Segment, LCH3SegmentSave, _is_done, _locked, project_dir
from .lc_h3_story import changed_since_made

NODE_ID = "LCH3Make"
NODE_DISPLAY_NAME = "LC H3 Make (BETA) 🎥"
PIPE_TYPE_V2 = "LC_H3_PIPE_V2"


class LCH3Make:
    @classmethod
    def INPUT_TYPES(cls):
        seg = LCH3Segment.INPUT_TYPES()
        req, opt = seg["required"], seg["optional"]
        return {
            "required": {
                "plan": (PLAN_TYPE, {"tooltip": "From LC H3 Story 🎬."}),
                "pipe": (PIPE_TYPE_V2, {"tooltip": "LC MiniMax H3 Pipe V2 with the H3 model (fl2va_model), its text encoder, the video "
                                                   "and audio VAEs, width / height, steps, sampler, scheduler and seed."}),
                "make": opt["make"],
                "index": req["index"],
                "crf": ("INT", {"default": 16, "min": 0, "max": 40, "tooltip": "mp4 quality. Lower = better and bigger. 16 is close to lossless."}),
            },
            "optional": {
                "refs": (REFS_TYPE, {"tooltip": "From LC H3 Story 🎬: the cast and place pictures."}),
                "redo_after": opt["redo_after"],
                "refs_on": opt["refs_on"],
                "hold_scene": opt["hold_scene"],
                "memory": req["memory"],
                "memory_gap": req["memory_gap"],
                "color_match": req["color_match"],
                "carry_on_line": req["carry_on_line"],
                "prompt_format": opt["prompt_format"],
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "AUDIO", "STRING")
    RETURN_NAMES = ("video", "audio", "info")
    OUTPUT_TOOLTIPS = ("The movie so far: every made shot joined.", "This shot's sound.",
                       "What was made and saved, the face match and how smooth each join is.")
    FUNCTION = "make_shot"
    OUTPUT_NODE = True
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = ("Makes the shots of an LC H3 Story: builds the next one, samples it and saves it (takes, the tail the next "
                   "shot starts from, the joined movie). make: all remaining chains every shot not made yet.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # the next shot depends on what is saved on disk, never on the inputs alone

    def make_shot(self, plan, pipe, make, index, crf, refs=None, redo_after="this + the CONTINUE ones after it", refs_on="cuts only",
                  hold_scene=True, memory="clip", memory_gap=5.0, color_match=0.0, carry_on_line=True,
                  prompt_format="sections (H3 trained)", prompt=None, extra_pnginfo=None):
        p = pipe if isinstance(pipe, dict) else {}
        need = {"fl2va_model": "the H3 model", "fl2va_clip": "the H3 text encoder", "video_vae": "the video VAE", "audio_vae": "the audio VAE"}
        missing = [v for k, v in need.items() if p.get(k) is None]
        if missing:
            raise ValueError(f"[LC H3] Make: the pipe has no {', '.join(missing)}. Wire them into the LC MiniMax H3 Pipe V2.")
        model = p["fl2va_model"]
        width, height = int(p.get("width") or 960), int(p.get("height") or 544)
        steps = int(p.get("total_steps") or 8)
        sampler_name = p.get("sampler_name") or "res_multistep"
        scheduler = p.get("scheduler") or "simple"
        seed = int(p.get("seed") or 0)

        # all remaining with every shot made: the shots edited since they were made are made again (Run = "make my
        # changes"), one after another; nothing edited: the finished movie is shown
        more = []
        if make == "all remaining":
            folder = project_dir(plan.get("project", ""))
            shots = plan.get("shots") or []
            if shots and all(_is_done(folder, k) for k in range(1, len(shots) + 1)):
                with open(os.path.join(folder, "plan.json"), encoding="utf-8") as f:
                    disk = json.load(f)
                edited = sorted(changed_since_made(folder, disk) - _locked(folder))
                if not edited:
                    movie = os.path.join(folder, f"{os.path.basename(folder)}.mp4")
                    msg = (f"🎬 Every shot of {plan.get('project')} is made, and none was changed since. Edit a shot (its tile "
                           "on the Story turns ✏️) and Run to make it again, or ↻ a tile.")
                    print(f"[LC H3] Make: {msg}")
                    return {"ui": {"text": [msg]}, "result": (movie if os.path.exists(movie) else "", None, msg)}
                make, index, more = "redo", edited[0], edited[1:]
                print(f"[LC H3] Make: shots changed since made: segment {', '.join(map(str, edited))}. Making segment {index} again.")
            elif shots:
                # shots still to make (new ones, or the keep-rolling ones a redo set aside): the edited ones that are
                # made stay on the list, so the chain comes back for them once the rest is made
                with open(os.path.join(folder, "plan.json"), encoding="utf-8") as f:
                    disk = json.load(f)
                more = sorted(changed_since_made(folder, disk) - _locked(folder))

        built = LCH3Segment().build(plan, p["fl2va_clip"], p["video_vae"], p["audio_vae"], index, width, height, seed, carry_on_line,
                                    color_match, memory, memory_gap, None, None, prompt_format, redo_after, refs, refs_on, make,
                                    hold_scene)
        res = built["result"] if isinstance(built, dict) else built
        seg_text = (built.get("ui", {}).get("text") or [""])[0] if isinstance(built, dict) else ""
        if isinstance(res[0], ExecutionBlocker):  # every shot is made: nothing to do
            msg = res[4]
            return {"ui": {"text": [msg]}, "result": ("", None, msg)}
        positive, latent, seed, segment, info, _prompt = res

        from comfy_extras import nodes_custom_sampler as cs

        guider = cs.Guider_Basic(model)
        guider.set_conds(positive)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
        out = cs.SamplerCustomAdvanced.execute(cs.Noise_RandomNoise(seed), guider, comfy.samplers.sampler_object(sampler_name), sigmas, latent)
        samples = out.args[0] if hasattr(out, "args") else out[0]

        segment["_more"] = more  # Save tells the chain these edited ones are still to make
        saved = LCH3SegmentSave().save(segment, samples, p["video_vae"], p["audio_vae"], crf, prompt=prompt, extra_pnginfo=extra_pnginfo)
        audio, text, video = saved["result"]
        text = f"{info}\n\n{text}" if info else text
        return {"ui": {"text": [text or seg_text]}, "result": (video, audio, text)}


NODE_CLASS_MAPPINGS = {NODE_ID: LCH3Make}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: NODE_DISPLAY_NAME}
