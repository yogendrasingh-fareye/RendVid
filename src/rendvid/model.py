"""LTX-Video engine for Apple Silicon.

16 GB of unified memory cannot hold the T5-XXL text encoder (~9.5 GB in bf16)
next to the video transformer, so work happens in two phases:

1. ``encode_prompts`` loads only the text encoder, embeds every scene prompt,
   moves the embeddings to CPU and frees the encoder.
2. ``VideoEngine`` loads the transformer + VAE (+ optional latent upscaler)
   and renders scenes from those cached embeddings.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image

from . import config

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16


class Cancelled(Exception):
    pass


def free_memory() -> None:
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


@dataclass
class PromptEmbeds:
    embeds: torch.Tensor
    mask: torch.Tensor
    neg_embeds: torch.Tensor
    neg_mask: torch.Tensor

    def as_kwargs(self, use_cfg: bool) -> dict:
        kw = {
            "prompt_embeds": self.embeds.to(DEVICE),
            "prompt_attention_mask": self.mask.to(DEVICE),
        }
        if use_cfg:
            kw["negative_prompt_embeds"] = self.neg_embeds.to(DEVICE)
            kw["negative_prompt_attention_mask"] = self.neg_mask.to(DEVICE)
        return kw


def encode_prompts(prompts: list[str], negative: str) -> list[PromptEmbeds]:
    from diffusers import LTXConditionPipeline

    pipe = LTXConditionPipeline.from_pretrained(
        config.BASE_REPO, transformer=None, vae=None, torch_dtype=DTYPE
    )
    pipe.text_encoder.to(DEVICE)
    out = []
    try:
        with torch.no_grad():
            for p in prompts:
                e, m, ne, nm = pipe.encode_prompt(
                    p, negative, do_classifier_free_guidance=True, device=DEVICE, dtype=DTYPE
                )
                out.append(PromptEmbeds(e.cpu(), m.cpu(), ne.cpu(), nm.cpu()))
    finally:
        del pipe
        free_memory()
    return out


class VideoEngine:
    def __init__(self, multiscale: bool):
        from diffusers import (
            AutoencoderKLLTXVideo,
            LTXConditionPipeline,
            LTXLatentUpsamplePipeline,
            LTXVideoTransformer3DModel,
        )
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(config.CKPT_REPO, config.CKPT_FILE)
        transformer = LTXVideoTransformer3DModel.from_single_file(ckpt, torch_dtype=DTYPE)
        vae = AutoencoderKLLTXVideo.from_single_file(ckpt, torch_dtype=DTYPE)
        self.pipe = LTXConditionPipeline.from_pretrained(
            config.BASE_REPO,
            transformer=transformer,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        self.pipe.vae.enable_tiling()

        self.upsampler = None
        if multiscale:
            self.upsampler = LTXLatentUpsamplePipeline.from_pretrained(
                config.UPSCALER_REPO, vae=self.pipe.vae, torch_dtype=DTYPE
            ).to(DEVICE)

    def close(self) -> None:
        del self.pipe, self.upsampler
        free_memory()

    def render(
        self,
        emb: PromptEmbeds,
        size: tuple[int, int],
        *,
        seed: int,
        guidance: float = 1.0,
        cond_image: Image.Image | None = None,
        cond_video: list[Image.Image] | None = None,
        num_frames: int = config.CLIP_FRAMES,
        on_step: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[Image.Image]:
        """Render one scene. ``size`` is the final (w, h); with multiscale the
        first pass runs at half that size and is refined after a 2x upscale."""
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        w, h = size
        use_cfg = guidance > 1.0
        conditions = None
        if cond_video:
            conditions = [LTXVideoCondition(video=cond_video, frame_index=0, strength=1.0)]
        elif cond_image is not None:
            conditions = [LTXVideoCondition(image=cond_image, frame_index=0, strength=1.0)]

        passes = 2 if self.upsampler else 1
        total = len(config.FIRST_PASS_TIMESTEPS) + (len(config.REFINE_TIMESTEPS) if passes == 2 else 0)
        done = 0

        def cb(_pipe, _i, _t, kw):
            nonlocal done
            done += 1
            if on_step:
                on_step(done, total)
            if should_cancel and should_cancel():
                raise Cancelled()
            return kw

        common = dict(
            conditions=conditions,
            num_frames=num_frames,
            frame_rate=config.FPS,
            guidance_scale=guidance,
            guidance_rescale=0.7 if use_cfg else 0.0,
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            image_cond_noise_scale=0.0,
            callback_on_step_end=cb,
            **emb.as_kwargs(use_cfg),
        )
        gen = torch.Generator("cpu").manual_seed(seed)

        with torch.no_grad():
            if passes == 1:
                frames = self.pipe(
                    width=w, height=h, timesteps=config.FIRST_PASS_TIMESTEPS,
                    generator=gen, output_type="pil", **common,
                ).frames[0]
            else:
                bw, bh = w // 2, h // 2
                latents = self.pipe(
                    width=bw, height=bh, timesteps=config.FIRST_PASS_TIMESTEPS,
                    generator=gen, output_type="latent", **common,
                ).frames
                latents = self.upsampler(
                    latents=latents, adain_factor=1.0, output_type="latent"
                ).frames
                frames = self.pipe(
                    width=bw * 2, height=bh * 2, latents=latents,
                    denoise_strength=0.999, timesteps=config.REFINE_TIMESTEPS,
                    generator=gen, output_type="pil", **common,
                ).frames[0]
        free_memory()
        return frames
