"""Wan 2.2 TI2V-5B engine (few-step distill) for Apple Silicon.

Same two-phase shape as ``model.py``: the umT5-XXL text encoder (~11 GB bf16)
is loaded alone, embeds every prompt and is freed; ``WanEngine`` then loads
the 5B transformer + Wan 2.2 VAE and renders from the cached embeddings.

TI2V-5B conditions with per-token timesteps: latent frames marked clean get
t=0 and are re-imposed every step. ``cond_image`` pins latent frame 0;
``cond_video`` (e.g. the previous clip's 9-frame tail) pins the first
(n-1)//4+1 latent frames, so continuations chain like LTX's video conditioning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps

from . import config
from .model import DEVICE, Cancelled, free_memory

DTYPE = torch.bfloat16
MAX_TOKENS = 512


@dataclass
class WanEmbeds:
    embeds: torch.Tensor
    neg_embeds: torch.Tensor


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def encode_prompts(prompts: list[str], negative: str, repo: str | None = None) -> list[WanEmbeds]:
    from transformers import AutoTokenizer, UMT5EncoderModel

    repo = repo or config.WAN_REPO
    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE)

    def embed(text: str) -> torch.Tensor:
        t = tok([_clean(text)], padding="max_length", max_length=MAX_TOKENS, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.sum())
        e = enc(t.input_ids.to(DEVICE), t.attention_mask.to(DEVICE)).last_hidden_state[0, :n]
        out = torch.zeros(1, MAX_TOKENS, e.shape[-1], dtype=DTYPE)
        out[0, :n] = e.to("cpu", DTYPE)
        return out

    try:
        with torch.no_grad():
            neg = embed(negative)
            out = [WanEmbeds(embed(p), neg) for p in prompts]
    finally:
        del enc
        free_memory()
    return out


def _to_tensor(frames: list[Image.Image], size: tuple[int, int]) -> torch.Tensor:
    """PIL frames -> (1, 3, F, H, W) in [-1, 1], center-cropped to ``size``."""
    arr = np.stack([np.asarray(ImageOps.fit(f.convert("RGB"), size, Image.LANCZOS)) for f in frames])
    x = torch.from_numpy(arr).float().div(127.5).sub(1.0)
    return x.permute(3, 0, 1, 2).unsqueeze(0)


class WanEngine:
    def __init__(self, repo: str | None = None, sampler: str | None = None, steps: int | None = None,
                 shift: float | None = None):
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel

        self.repo = repo or config.WAN_REPO
        self.sampler = sampler or config.WAN_SAMPLER
        self.steps = steps or config.WAN_STEPS
        self.shift = shift or config.WAN_SHIFT
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.repo, subfolder="transformer", torch_dtype=DTYPE).to(DEVICE)
        self.vae = AutoencoderKLWan.from_pretrained(self.repo, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
        self.vae.enable_tiling()
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.repo, subfolder="scheduler", flow_shift=self.shift)
        c = self.vae.config
        self.z = c.z_dim
        self.mean = torch.tensor(c.latents_mean).view(1, -1, 1, 1, 1).to(DEVICE, torch.float32)
        self.std = torch.tensor(c.latents_std).view(1, -1, 1, 1, 1).to(DEVICE, torch.float32)
        self.sf_s, self.sf_t = c.scale_factor_spatial, c.scale_factor_temporal

    def close(self) -> None:
        del self.transformer, self.vae
        free_memory()

    def _encode(self, frames: list[Image.Image], size: tuple[int, int]) -> torch.Tensor:
        x = _to_tensor(frames, size).to(DEVICE, self.vae.dtype)
        lat = self.vae.encode(x).latent_dist.mode().float()
        return (lat - self.mean) / self.std

    def render(
        self,
        emb: WanEmbeds,
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
        w, h = size
        m = self.sf_s * 2  # VAE stride x patch size
        w, h = w // m * m, h // m * m
        num_frames = (num_frames - 1) // self.sf_t * self.sf_t + 1
        T = (num_frames - 1) // self.sf_t + 1
        lh, lw = h // self.sf_s, w // self.sf_s
        use_cfg = guidance > 1.0

        gen = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn((1, self.z, T, lh, lw), generator=gen, dtype=torch.float32).to(DEVICE)

        mask = torch.ones(1, 1, T, lh, lw, device=DEVICE)
        cond = None
        with torch.no_grad():
            if cond_video:
                n = (len(cond_video) - 1) // self.sf_t * self.sf_t + 1
                cond = self._encode(cond_video[-n:], (w, h))
            elif cond_image is not None:
                cond = self._encode([cond_image], (w, h))
        k = 0
        if cond is not None:
            k = min(cond.shape[2], T - 1)
            pad = torch.zeros_like(latents)
            pad[:, :, :k] = cond[:, :, :k]
            cond = pad
            mask[:, :, :k] = 0

        def compose(x: torch.Tensor) -> torch.Tensor:
            return x if cond is None else (1 - mask) * cond + mask * x

        pe = emb.embeds.to(DEVICE, DTYPE)
        ne = emb.neg_embeds.to(DEVICE, DTYPE) if use_cfg else None
        tok_mask = mask[0, 0][:, ::2, ::2].flatten()  # one entry per transformer token

        def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            inp = compose(x).to(DTYPE)
            ts = (tok_mask * t).unsqueeze(0)
            v = self.transformer(hidden_states=inp, timestep=ts, encoder_hidden_states=pe, return_dict=False)[0]
            if use_cfg:
                vu = self.transformer(hidden_states=inp, timestep=ts, encoder_hidden_states=ne, return_dict=False)[0]
                v = vu + guidance * (v - vu)
            return v.float()

        def tick(i: int, total: int) -> None:
            if on_step:
                on_step(i + 1, total)
            if should_cancel and should_cancel():
                raise Cancelled()

        with torch.no_grad():
            if self.sampler == "dmd":
                sigmas = config.WAN_DMD_SIGMAS
                x = latents
                for i, s in enumerate(sigmas):
                    x0 = x - s * velocity(x, torch.tensor(s * 1000.0, device=DEVICE))
                    if i + 1 < len(sigmas):
                        noise = torch.randn(x.shape, generator=gen, dtype=torch.float32).to(DEVICE)
                        x = (1 - sigmas[i + 1]) * x0 + sigmas[i + 1] * noise
                    else:
                        x = x0
                    tick(i, len(sigmas))
                latents = x
            else:
                sch = self.scheduler
                sch.set_timesteps(self.steps, device=DEVICE)
                for i, t in enumerate(sch.timesteps):
                    v = velocity(latents, t)
                    latents = sch.step(v, t, latents, return_dict=False)[0]
                    tick(i, len(sch.timesteps))

            latents = compose(latents)
            del pe, ne
            video = self.vae.decode((latents * self.std + self.mean).to(self.vae.dtype), return_dict=False)[0]
        video = ((video[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        del latents, cond
        free_memory()
        return [Image.fromarray(f) for f in video]
