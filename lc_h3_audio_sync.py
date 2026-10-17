"""
LC H3 Audio Sync (BETA)
-----------------------
Makes a MiniMax H3 video perform YOUR track. H3 makes the picture and the sound together in one latent, so left
alone it invents its own audio and the mouth moves to a song that doesn't exist. This node encodes your recording
into the sound half and locks it there with a per-stream noise mask (1 = denoise the video, 0 = keep the audio as
is; ComfyUI's sampler blends a nested mask stream by stream), so the only thing the model decides is the picture
that fits it.

The track is fitted in the waveform, not the latent: a zero audio latent isn't silence (it decodes to a tone or
hiss), real zeros are. Numbers from comfy_extras/nodes_minimax_h3.py: 24 fps, 40 audio latent frames per second.
Idea from Pixaroma's H3 Audio Sync (ComfyUI-Pixaroma, MIT); written for LC_AV.
"""
from __future__ import annotations

import torch

FPS = 24
AUDIO_LATENT_FPS = 40


def _frames(video_t):
    """Video latent length -> frame count (inverse of 2 if f <= 5 else ((f - 5) // 17) * 5 + 2)."""
    t = int(video_t)
    return 5 if t <= 2 else ((t - 2) // 5) * 17 + 5


def fit_waveform(waveform, sample_rate, start_s, want, when_short):
    """[C, L] or [B, C, L] cut from start_s to exactly `want` samples; short tracks get silence or a loop."""
    start = max(0, int(round(float(start_s) * sample_rate)))
    piece = waveform[..., start:start + want]
    take = piece.shape[-1]
    if take >= want:
        return piece
    if when_short == "loop" and take > 0:
        idx = torch.arange(want - take, device=piece.device) % take
        return torch.cat([piece, piece.index_select(-1, idx)], dim=-1)
    pad = torch.zeros(*piece.shape[:-1], want - take, dtype=piece.dtype, device=piece.device)
    return torch.cat([piece, pad], dim=-1)


class LCH3AudioSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The picture-and-sound latent from an H3 node (Empty MiniMax H3 AV Latent, "
                                                 "Image to Video, Reference to Video...)."}),
                "audio_vae": ("VAE", {"tooltip": "H3's audio VAE, the same one the H3 nodes use."}),
                "audio": ("AUDIO", {"tooltip": "The recording the video should perform: a song, a line of dialogue, anything."}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                                            "tooltip": "Where in the track the clip starts. The clip's own length decides how much is used."}),
                "when_short": (["silence", "loop"], {"default": "silence",
                                                     "tooltip": "The track is shorter than the clip: fill the rest with silence, or loop the track."}),
            },
            "optional": {
                "positive": ("CONDITIONING", {"tooltip": "The H3 positive conditioning. Wire it through here: the track is "
                                                         "also handed to H3 as a clean guide, so the mouth follows the words "
                                                         "from the first step (the lock alone is too noisy early on)."}),
            },
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "STRING", "CONDITIONING")
    RETURN_NAMES = ("latent", "audio", "info", "positive")
    OUTPUT_TOOLTIPS = (
        "The latent with your track locked into its sound half. Wire it into the sampler.",
        "Your track cut to exactly the clip's length, for the save node, so picture and sound end together.",
        "Clip and track lengths, and whether the track was filled.",
    )
    FUNCTION = "sync"
    CATEGORY = "LC AV/H3 Long Video"
    DESCRIPTION = (
        "Makes a MiniMax H3 video perform your track instead of sound the model invents. Put it between the latent and "
        "the sampler. Your recording goes into the sound half and is held still, so the model only decides the picture "
        "that fits it, mouth included. The clip's length comes from the latent; a shorter track is filled with silence "
        "or looped. H3 was trained to about 15 seconds."
    )

    def sync(self, latent, audio_vae, audio, start_seconds=0.0, when_short="silence", positive=None):
        import comfy.audio
        import comfy.nested_tensor

        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None or not getattr(samples, "is_nested", False) or len(samples.unbind()) < 2:
            raise ValueError("[LC H3 Audio Sync] This needs a MiniMax H3 latent (picture and sound together), e.g. from "
                             "Empty MiniMax H3 AV Latent or MiniMax H3 Image to Video.")
        video, audio_tpl = samples.unbind()[:2]
        audio_t = audio_tpl.shape[-1]
        frames = _frames(video.shape[2])
        clip_s = frames / FPS

        waveform = audio["waveform"][:1]  # H3 runs batch 1
        sr = int(audio["sample_rate"])
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sr != vae_sr:
            waveform = comfy.audio.resample(waveform, sr, vae_sr)
        track_s = max(0.0, waveform.shape[-1] / vae_sr - float(start_seconds))
        want = int(round(audio_t / AUDIO_LATENT_FPS * vae_sr))
        fitted = fit_waveform(waveform, vae_sr, start_seconds, want, when_short)

        z = audio_vae.encode(fitted.movedim(1, -1))  # [1, 32, 2, T]
        if z.shape[-1] > audio_t:
            z = z[..., :audio_t]
        elif z.shape[-1] < audio_t:  # the encoder rounds: repeat the last frame (a zero frame isn't silence)
            z = torch.cat([z, z[..., -1:].repeat_interleave(audio_t - z.shape[-1], dim=-1)], dim=-1)
        z = z.to(device=audio_tpl.device, dtype=audio_tpl.dtype)

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, z))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(z)))

        # The lock (noise mask) keeps the saved sound exactly your track, but the sampler re-noises it to each step's
        # level, so in the first steps (where the motion and the mouth are decided) H3 hears mostly noise. Handing the
        # same track over as a guide keyframe at frame 0 (what core's MiniMax H3 Add Guide does with audio) gives H3
        # the clean words from the start.
        if positive is not None:
            import node_helpers
            keyframes = list(positive[0][1].get("minimax_keyframes", [])) if positive else []
            keyframes.append({"resolved_frame_index": 0, "audio_latent": z.clone()})
            positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})

        info = f"Clip {clip_s:.2f} s ({frames} frames), track {track_s:.2f} s from {float(start_seconds):.1f} s"
        if track_s + 0.02 < clip_s:
            info += f": {clip_s - track_s:.2f} s short, {'looped' if when_short == 'loop' else 'filled with silence'}"
        if clip_s > 15.25:
            info += ". ⚠️ Past 15 s: H3 was trained to about 15 s and often falls apart beyond it"
        if positive is not None:
            info += ". Track also handed to H3 as a guide"
        print(f"[LC H3 Audio Sync] {info}")
        return (out, {"waveform": fitted, "sample_rate": vae_sr}, info, positive)


NODE_CLASS_MAPPINGS = {"LCH3AudioSync": LCH3AudioSync}
NODE_DISPLAY_NAME_MAPPINGS = {"LCH3AudioSync": "LC H3 Audio Sync (BETA) 🎤"}
