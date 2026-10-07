"""Keyframe stills with SDXL for consistent characters.

1. ``portraits``: one reference image per character from its fixed
   description (plain background, full figure).
2. ``keyframes``: one still per shot. IP-Adapter feeds the main character's
   reference image into every shot, so look and outfit carry across cuts far
   better than text alone. LTX then animates each still.

Hyper-SD's 8-step CFG LoRA keeps a still at ~10-20 s on Apple Silicon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from .model import DEVICE, Cancelled, free_memory

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"
IP_REPO = "h94/IP-Adapter"
IP_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
HYPER = ("ByteDance/Hyper-SD", "Hyper-SDXL-8steps-CFG-lora.safetensors")

# SDXL-native sizes (~1 MP) per aspect; resized to the video size later.
SIZES = {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (1024, 1024)}
NEGATIVE = ("blurry, lowres, deformed, bad anatomy, extra limbs, extra fingers, text, watermark, "
            "logo, frame, border, cropped, duplicate, clones, many copies, collage, ugly")
PORTRAIT_NEGATIVE = NEGATIVE + ", multiple subjects, group, crowd, flock, character sheet, several views, pattern"


def ready() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(r, f), str) for r, f in [
        (SDXL, "unet/diffusion_pytorch_model.fp16.safetensors"),
        (VAE_FIX, "diffusion_pytorch_model.safetensors"),
        (IP_REPO, f"sdxl_models/{IP_WEIGHT}"),
        (IP_REPO, "models/image_encoder/model.safetensors"),
        HYPER,
    ])


class KeyframeEngine:
    def __init__(self) -> None:
        from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline
        from huggingface_hub import hf_hub_download
        from transformers import CLIPVisionModelWithProjection

        dtype = torch.float16
        vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=dtype)
        encoder = CLIPVisionModelWithProjection.from_pretrained(
            IP_REPO, subfolder="models/image_encoder", torch_dtype=dtype
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL, vae=vae, image_encoder=encoder, torch_dtype=dtype, variant="fp16", use_safetensors=True
        )
        self.pipe.load_lora_weights(hf_hub_download(*HYPER))
        self.pipe.fuse_lora()
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe.load_ip_adapter(IP_REPO, subfolder="sdxl_models", weight_name=IP_WEIGHT)
        self.pipe.to(DEVICE)
        self.pipe.vae.enable_tiling()

    def close(self) -> None:
        del self.pipe
        free_memory()

    def _run(self, prompt: str, size: tuple[int, int], seed: int, ref: Image.Image | None, ip_scale: float,
             negative: str = NEGATIVE) -> Image.Image:
        w, h = size
        self.pipe.set_ip_adapter_scale(ip_scale if ref is not None else 0.0)
        # The adapter always expects an image; a blank one at scale 0 is a no-op.
        ip_image = ref if ref is not None else Image.new("RGB", (224, 224), "white")
        with torch.no_grad():
            return self.pipe(
                prompt=prompt, negative_prompt=negative, width=w, height=h,
                num_inference_steps=8, guidance_scale=5.0,
                ip_adapter_image=ip_image,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]

    def portrait(self, look: str, style: str, aspect: str, seed: int, setting: str = "") -> Image.Image:
        # Drawn inside the film's own world and style: IP-Adapter copies the
        # reference's palette and background, so a grey studio backdrop would
        # leak grey studio backdrops into every shot.
        where = f"in {setting}" if setting else "in a softly blurred natural environment"
        prompt = (f"a single {look}, one subject only, full body, centered, clearly visible, {where}, "
                  f"shallow depth of field, sharp focus on the subject, {style}")
        return self._run(prompt, SIZES["1:1"], seed, None, 0.0, PORTRAIT_NEGATIVE + ", plain background, studio backdrop")

    def keyframe(self, prompt: str, aspect: str, seed: int, ref: Image.Image | None) -> Image.Image:
        return self._run(prompt, SIZES[aspect], seed, ref, 0.25)


def make_portraits(
    characters: list[dict], style: str, out_dir: Path, seed: int,
    engine: KeyframeEngine, should_cancel: Callable[[], bool] | None = None, setting: str = "",
) -> dict[str, Path]:
    """Reference image per character (reused if already on disk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = {}
    for k, ch in enumerate(characters):
        p = out_dir / f"{_slug(ch['name'])}.png"
        if not p.exists():
            if should_cancel and should_cancel():
                raise Cancelled()
            engine.portrait(ch["look"], style, "1:1", seed + 1000 + k, setting).save(p)
        refs[ch["name"]] = p
    return refs


def _slug(name: str) -> str:
    s = "".join(c if c.isalnum() else "_" for c in name.strip().lower())
    return s or "character"
