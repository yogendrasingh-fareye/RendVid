"""Local SDXL LoRA training on Apple Silicon (MPS, 16 GB).

Two kinds of LoRA:

* ``character`` (also props / locations): locks one recurring subject. Captions
  are ``"<trigger> <class word>, <what varies>"`` so the trigger soaks up the
  look and the caption explains pose and background.
* ``style``: a house look. Captions are ``"<trigger>, <content>"`` so the
  trigger soaks up the rendering style.

Memory plan (fits 16 GB): captions -> text embeddings and images -> VAE
latents are computed up front, then the text encoders and VAE are freed.
Training holds only the UNet (bf16, frozen, gradient checkpointing) plus a
small fp32 LoRA on its attention projections, batch 1, torch AdamW.

If there are no photos of a character, ``make_training_set`` draws a varied
set (poses, angles, backgrounds) from its description with the keyframe
engine: an anchor portrait first, then IP-Adapter-guided variations of it,
keeping the ones that look most like the anchor.

API (used by the library and the CLI)::

    paths = make_training_set(description, style, out_dir, count=16, seed=0)
    info = train_lora("character", "crow", paths, out_path, trigger="rvd_crow", class_word="crow")
"""

from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps

from .model import DEVICE, Cancelled, free_memory

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]

# Per-kind defaults: steps, rank, learning rate, resolution.
DEFAULTS = {
    "character": {"steps": 800, "rank": 16, "lr": 1e-4, "resolution": 768},
    "style": {"steps": 800, "rank": 16, "lr": 1e-4, "resolution": 768},
}


def default_trigger(kind: str, name: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_") or "x"
    return f"rvd_style_{slug}" if kind == "style" else f"rvd_{slug}"


# --- Training set generation -------------------------------------------------

_VIEWS = [
    "full body, front view", "full body, side view, profile", "three-quarter view", "close-up of the head",
    "seen from behind, looking back over the shoulder", "medium shot", "low angle shot", "high angle shot from above",
    "in mid-motion, dynamic pose", "resting, calm pose", "full body, facing left", "full body, facing right",
    "looking up", "looking down", "wide shot, small in frame", "three-quarter view, looking at the camera",
]
_PLACES = [
    "in a sunny village courtyard", "in a lush green garden", "on a rooftop at golden hour", "by a calm riverbank",
    "in a forest clearing", "against a soft pastel background", "inside a cozy room with warm light",
    "on a dusty country road at noon", "under a clear blue sky", "at night under soft moonlight",
    "on a wooden fence", "near a whitewashed wall", "in a field of tall grass", "on a stone well",
    "in light rain", "at dawn with mist",
]
_LOCATION_VIEWS = [
    "wide establishing shot", "eye-level view", "aerial view", "close detail view", "at golden hour",
    "at noon", "at dusk", "at night", "in the rain", "in morning mist", "from a low angle", "from a distance",
    "with soft overcast light", "seen through a doorway", "panoramic view", "early morning",
]
_STYLE_SUBJECTS = [
    "a crow on a branch", "an old man reading a newspaper", "a village street with small houses", "a cup of tea on a table",
    "a girl flying a kite", "a bullock cart on a dirt road", "a temple by a lake", "a cat sleeping on a windowsill",
    "a market stall with vegetables", "a mountain landscape at sunrise", "a boy running in the rain", "a clay pot in a garden",
    "an elephant in a forest", "a fisherman's boat on a river", "a woman cooking in a kitchen", "a banyan tree at dusk",
]


def make_training_set(description: str, style: str, out_dir: Path, count: int = 16, seed: int = 0,
                      on_progress: Callable[[int, int], None] | None = None, *, kind: str = "character",
                      engine=None, should_cancel: Callable[[], bool] | None = None) -> list[Path]:
    """Draw ``count`` varied images of a character / prop / location / style
    from its description into ``out_dir`` (PNG + .txt caption of what
    varies). ``engine``: an open KeyframeEngine to reuse (with IP-Adapter);
    otherwise one is created and closed here. Returns the image paths."""
    from .keyframes import PORTRAIT_NEGATIVE, SIZES, KeyframeEngine

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    own = engine is None
    if own:
        engine = KeyframeEngine(loras=[])
    try:
        rng = random.Random(seed)
        style_s = f", {style}" if style else ""
        if kind == "style":
            subjects = (_STYLE_SUBJECTS * (count // len(_STYLE_SUBJECTS) + 1))[:count]
            jobs = [(f"{s}, {description}", s) for s in subjects]
            anchor = None
            extra = 0
        else:
            if kind == "location":
                varies = rng.sample(_LOCATION_VIEWS * 2, count + count // 4)
            else:
                views, places = _VIEWS[:], _PLACES[:]
                rng.shuffle(views)
                rng.shuffle(places)
                varies = [f"{views[k % len(views)]}, {places[(k * 7) % len(places)]}" for k in range(count + count // 4)]
            one = "" if kind == "location" else "a single "
            jobs = [(f"{one}{description}, {v}, one subject only, clearly visible{style_s}", v) for v in varies]
            extra = len(jobs) - count
            # Anchor: the identity every variation is guided towards.
            anchor = engine.portrait(description, style, "1:1", seed, loras=[])
            anchor.save(out_dir / "_anchor.png")
        total = len(jobs)
        made = []  # (image, caption, score)
        anchor_emb = _clip_embed(engine, anchor) if anchor is not None else None
        for k, (prompt, caption) in enumerate(jobs):
            if should_cancel and should_cancel():
                raise Cancelled()
            aspect = ["1:1", "1:1", "16:9", "9:16"][k % 4]
            if anchor is not None and engine.ip_adapter:
                img = engine._run(prompt, SIZES[aspect], seed + 1 + k, anchor, 0.45, PORTRAIT_NEGATIVE, loras=[])
            else:
                img = engine._run(prompt, SIZES[aspect], seed + 1 + k, None, 0.0, PORTRAIT_NEGATIVE, loras=[])
            score = float(_clip_embed(engine, img) @ anchor_emb.T) if anchor_emb is not None else 0.0
            made.append((img, caption, score))
            if on_progress:
                on_progress(k + 1, total)
        # Keep the variations closest to the anchor (drops off-model draws).
        if extra > 0 and anchor_emb is not None:
            made = sorted(made, key=lambda m: -m[2])[:count]
        paths = []
        for k, (img, caption, score) in enumerate(made):
            p = out_dir / f"{k:03d}.png"
            img.save(p)
            p.with_suffix(".txt").write_text(caption)
            paths.append(p)
        (out_dir / "_set.json").write_text(json.dumps({
            "description": description, "style": style, "kind": kind, "seed": seed,
            "scores": [round(m[2], 4) for m in made]}, indent=2))
        return paths
    finally:
        if own:
            engine.close()


@torch.no_grad()
def _clip_embed(engine, img: Image.Image | None) -> torch.Tensor | None:
    enc = getattr(engine.pipe, "image_encoder", None)
    if img is None or enc is None:
        return None
    px = engine.pipe.feature_extractor(img, return_tensors="pt").pixel_values.to(DEVICE, enc.dtype)
    e = enc(px).image_embeds.float().cpu()
    return e / e.norm(dim=-1, keepdim=True)


# --- Training ----------------------------------------------------------------

def _buckets(res: int) -> list[tuple[int, int]]:
    """(w, h) multiples of 64 with area <= res^2, aspect 1:2 .. 2:1."""
    out = set()
    for w in range(res // 2, res * 2 + 1, 64):
        h = (res * res // w) // 64 * 64
        if h >= 64 and 0.5 <= w / h <= 2.0:
            out.add((w, h))
    return sorted(out)


def _fit(img: Image.Image, res: int) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    """Resize-and-center-crop to the nearest aspect bucket.
    Returns (image, original (h, w), crop (top, left))."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    W, H = img.size
    bw, bh = min(_buckets(res), key=lambda b: abs(math.log((b[0] / b[1]) / (W / H))))
    s = max(bw / W, bh / H)
    rw, rh = max(bw, round(W * s)), max(bh, round(H * s))
    img = img.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - bw) // 2, (rh - bh) // 2
    return img.crop((left, top, left + bw, top + bh)), (H, W), (top, left)


def _caption(kind: str, trigger: str, class_word: str, img_path: Path) -> str:
    side = img_path.with_suffix(".txt")
    extra = side.read_text().strip() if side.exists() else ""
    head = trigger if kind == "style" or not class_word else f"{trigger} {class_word}"
    return f"{head}, {extra}" if extra else head


def _peak(cur: float) -> float:
    if DEVICE == "mps":
        return max(cur, torch.mps.driver_allocated_memory() / 1e9)
    return cur


def train_lora(kind: str, name: str, images: list[Path], out_path: Path, *, trigger: str, class_word: str = "",
               steps: int | None = None, on_progress: Callable[[int, int], None] | None = None,
               should_cancel: Callable[[], bool] | None = None, rank: int | None = None,
               lr: float | None = None, resolution: int | None = None, base: str | None = None,
               flip: bool = True, seed: int = 0, log: Callable[[str], None] | None = None) -> dict:
    """Train a UNet LoRA on ``images`` and save it to ``out_path`` (.safetensors)
    plus a .json sidecar. A ``<image>.txt`` next to an image adds to its
    caption. Returns {"path", "trigger", "steps", "seconds", ...}."""
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from .keyframes import VAE_FIX, base_repo

    d = DEFAULTS["style" if kind == "style" else "character"]
    steps, rank = steps or d["steps"], rank or d["rank"]
    lr, res = lr or d["lr"], resolution or d["resolution"]
    repo = base_repo(base)
    images = [Path(p) for p in images if Path(p).suffix.lower() in IMAGE_EXTS]
    if not images:
        raise ValueError("no training images")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    say = log or (lambda m: None)
    t0 = time.time()
    peak = 0.0
    if on_progress:
        on_progress(0, steps)

    # 1. Captions -> text embeddings (then free the text encoders).
    captions = [_caption(kind, trigger, class_word, p) for p in images]
    pipe = StableDiffusionXLPipeline.from_pretrained(repo, unet=None, vae=None, torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
    pipe.text_encoder.to(DEVICE)
    pipe.text_encoder_2.to(DEVICE)
    embeds = {}
    with torch.no_grad():
        for c in set(captions):
            e, _, pooled, _ = pipe.encode_prompt(c, device=DEVICE, num_images_per_prompt=1,
                                                 do_classifier_free_guidance=False)
            embeds[c] = (e.cpu(), pooled.cpu())
    peak = _peak(peak)
    del pipe
    free_memory()

    # 2. Images -> VAE latent distributions (+ mirrored copies), then free the VAE.
    vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=torch.float16).to(DEVICE)
    samples = []
    with torch.no_grad():
        for p, c in zip(images, captions):
            img, orig, crop = _fit(Image.open(p), res)
            variants = [img, ImageOps.mirror(img)] if flip else [img]
            for v in variants:
                x = torch.from_numpy(np.array(v)).permute(2, 0, 1)[None].float() / 127.5 - 1
                dist = vae.encode(x.to(DEVICE, torch.float16)).latent_dist
                time_ids = torch.tensor([[orig[0], orig[1], crop[0], crop[1], v.height, v.width]], dtype=torch.float32)
                samples.append((dist.mean.cpu(), dist.std.cpu(), c, time_ids))
    scaling = vae.config.scaling_factor
    peak = _peak(peak)
    del vae
    free_memory()
    say(f"precomputed {len(samples)} latents / {len(embeds)} captions in {time.time() - t0:.0f}s")

    # 3. UNet + LoRA.
    dtype = torch.bfloat16
    unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", variant="fp16", torch_dtype=dtype)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=TARGET_MODULES))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    unet.to(DEVICE)
    unet.train()
    noise_sched = DDPMScheduler.from_pretrained(repo, subfolder="scheduler")
    acp = noise_sched.alphas_cumprod.to(DEVICE)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    gen = torch.Generator().manual_seed(seed)
    order: list[int] = []
    t_train = time.time()
    losses = []
    for step in range(steps):
        if should_cancel and should_cancel():
            raise Cancelled()
        if not order:
            order = torch.randperm(len(samples), generator=gen).tolist()
        mean, std, cap, time_ids = samples[order.pop()]
        lat = (mean + std * torch.randn(mean.shape, generator=gen)) * scaling
        noise = torch.randn(lat.shape, generator=gen)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), generator=gen)
        lat, noise, t = lat.to(DEVICE), noise.to(DEVICE), t.to(DEVICE)
        a = acp[t].view(-1, 1, 1, 1)
        noisy = a.sqrt() * lat + (1 - a).sqrt() * noise
        e, pooled = embeds[cap]
        pred = unet(noisy.to(dtype), t, encoder_hidden_states=e.to(DEVICE, dtype),
                    added_cond_kwargs={"text_embeds": pooled.to(DEVICE, dtype), "time_ids": time_ids.to(DEVICE, dtype)}).sample
        # Min-SNR (gamma 5) weighting for epsilon prediction.
        snr = a / (1 - a)
        w = torch.clamp(snr, max=5.0) / snr
        loss = (w * (pred.float() - noise).pow(2).mean(dim=(1, 2, 3), keepdim=True)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 10 == 0:
            peak = _peak(peak)
        if on_progress:
            on_progress(step + 1, steps)
        if (step + 1) % 50 == 0:
            el = time.time() - t_train
            say(f"step {step + 1}/{steps} loss {sum(losses[-50:]) / 50:.4f} "
                f"{el / (step + 1) * 100:.0f}s/100 steps peak {peak:.1f} GB")
    train_s = time.time() - t_train

    # 4. Save (diffusers LoRA format, loadable by pipe.load_lora_weights).
    lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    lora = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in lora.items()}
    StableDiffusionXLPipeline.save_lora_weights(out_path.parent, unet_lora_layers=lora, weight_name=out_path.name,
                                                safe_serialization=True)
    del unet, opt, params
    free_memory()
    info = {
        "path": str(out_path), "name": name, "kind": kind, "trigger": trigger, "class_word": class_word,
        "steps": steps, "rank": rank, "lr": lr, "resolution": res, "base": repo, "images": len(images),
        "seconds": round(time.time() - t0, 1), "train_seconds": round(train_s, 1),
        "sec_per_100_steps": round(train_s / steps * 100, 1), "peak_gb": round(peak, 2),
        "final_loss": round(sum(losses[-50:]) / max(1, len(losses[-50:])), 4), "weight": 1.0,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return info
