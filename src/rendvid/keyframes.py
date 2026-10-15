"""Keyframe stills with SDXL for consistent characters.

1. ``portraits``: one reference image per character from its fixed
   description (plain background, full figure).
2. ``keyframes``: one still per shot. IP-Adapter feeds the main character's
   reference image into every shot, so look and outfit carry across cuts far
   better than text alone. LTX then animates each still.
3. Optional LoRAs trained locally (``training.py``) lock a recurring
   character or a house style in far more tightly than IP-Adapter: pass
   ``loras=[...]`` to ``keyframe``/``portrait``. Their trigger words are added
   to the prompt automatically.

Hyper-SD's 8-step CFG LoRA keeps a still at ~10-20 s on Apple Silicon.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image

from .model import DEVICE, Cancelled, free_memory

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
# Selectable SDXL bases (diffusers repos with fp16 variants). All are SDXL-1.0
# fine-tunes, so Hyper-SD, IP-Adapter SDXL and SDXL LoRAs work on each.
BASES = {
    "sdxl": SDXL,                                # CreativeML Open RAIL++-M
    "realvis": "SG161222/RealVisXL_V5.0",        # Open RAIL++-M
    "dreamshaper": "Lykon/dreamshaper-xl-1-0",   # Open RAIL++-M
    "juggernaut": "RunDiffusion/Juggernaut-XL-v9",  # Open RAIL-M + no paid-API deployment without a licence
}
# Default base for keyframes (and for LoRA training: a LoRA should be used on
# the base it was trained on). Override with RENDVID_SDXL_BASE=<key or repo>.
DEFAULT_BASE = "sdxl"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"
IP_REPO = "h94/IP-Adapter"
IP_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
HYPER = ("ByteDance/Hyper-SD", "Hyper-SDXL-8steps-CFG-lora.safetensors")

# Trained LoRAs (CLI default location): outputs/.loras/<kind>/<name>.safetensors + .json
LORA_DIR = Path(__file__).resolve().parents[2] / "outputs" / ".loras"

# SDXL-native sizes (~1 MP) per aspect; resized to the video size later.
SIZES = {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (1024, 1024)}
NEGATIVE = ("blurry, lowres, deformed, bad anatomy, extra limbs, extra fingers, text, watermark, "
            "logo, frame, border, cropped, duplicate, clones, many copies, collage, ugly")
PORTRAIT_NEGATIVE = NEGATIVE + ", multiple subjects, group, crowd, flock, character sheet, several views, pattern"


def base_repo(base: str | None = None) -> str:
    """Repo id for a base key ("realvis"), a repo id, or the default."""
    b = base or os.environ.get("RENDVID_SDXL_BASE") or DEFAULT_BASE
    return BASES.get(b, b)


def ready(base: str | None = None) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(r, f), str) for r, f in [
        (base_repo(base), "unet/diffusion_pytorch_model.fp16.safetensors"),
        (VAE_FIX, "diffusion_pytorch_model.safetensors"),
        (IP_REPO, f"sdxl_models/{IP_WEIGHT}"),
        (IP_REPO, "models/image_encoder/model.safetensors"),
        HYPER,
    ])


# --- LoRAs -------------------------------------------------------------------

def available_loras(root: Path = LORA_DIR) -> list[dict]:
    """Trained LoRAs under outputs/.loras: [{name, kind, path, trigger, weight, base, steps, created}]."""
    out = []
    for p in sorted(Path(root).glob("*/*.safetensors")):
        meta = {}
        if p.with_suffix(".json").exists():
            try:
                meta = json.loads(p.with_suffix(".json").read_text())
            except ValueError:
                pass
        out.append({"name": p.stem, "kind": p.parent.name, "path": str(p),
                    "trigger": meta.get("trigger", ""), "weight": meta.get("weight", 1.0), **meta})
    return out


def _lora_spec(item) -> dict:
    """Normalise a LoRA reference: a name from available_loras(), a path, a
    (name_or_path, weight) tuple, or a dict {path, trigger?, weight?}."""
    if isinstance(item, (tuple, list)):
        spec = _lora_spec(item[0])
        spec["weight"] = float(item[1])
        return spec
    if isinstance(item, dict):
        spec = dict(item)
        if "path" not in spec and "name" in spec:
            spec = {**_lora_spec(spec["name"]), **{k: v for k, v in spec.items() if k != "name"}}
    else:
        p = Path(str(item))
        if p.suffix == ".safetensors" and p.exists():
            spec = {"path": str(p)}
        else:
            hits = [l for l in available_loras() if l["name"] == str(item)]
            if not hits:
                raise FileNotFoundError(f"no trained LoRA named {item!r}")
            spec = dict(hits[0])
    path = Path(spec["path"])
    if (not spec.get("trigger") or not spec.get("kind")) and path.with_suffix(".json").exists():
        try:
            meta = json.loads(path.with_suffix(".json").read_text())
            spec["trigger"] = spec.get("trigger") or meta.get("trigger", "")
            spec["kind"] = spec.get("kind") or meta.get("kind", "")
        except ValueError:
            pass
    return {"path": str(path), "trigger": spec.get("trigger") or "", "weight": float(spec.get("weight", 1.0)),
            "kind": spec.get("kind", "")}


def _with_triggers(prompt: str, specs: list[dict]) -> str:
    words = [s["trigger"] for s in specs if s["trigger"] and s["trigger"] not in prompt]
    return ", ".join(words + [prompt]) if words else prompt


class KeyframeEngine:
    def __init__(self, base: str | None = None, loras: Iterable | None = None, ip_adapter: bool = True) -> None:
        """``base``: key of BASES or a repo id (default DEFAULT_BASE).
        ``loras``: default LoRAs applied to every image (see ``keyframe``).
        ``ip_adapter``: False skips loading IP-Adapter (saves ~1.5 GB; use
        with a character LoRA)."""
        from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline
        from huggingface_hub import hf_hub_download

        dtype = torch.float16
        self.base = base_repo(base)
        vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=dtype)
        extra = {}
        if ip_adapter:
            from transformers import CLIPVisionModelWithProjection

            extra["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
                IP_REPO, subfolder="models/image_encoder", torch_dtype=dtype
            )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base, vae=vae, torch_dtype=dtype, variant="fp16", use_safetensors=True, **extra
        )
        # Fuse Hyper-SD into the weights, then drop its adapter so trained
        # LoRAs can be switched freely without touching the speed-up.
        self.pipe.load_lora_weights(hf_hub_download(*HYPER), adapter_name="hyper")
        self.pipe.fuse_lora(adapter_names=["hyper"])
        self.pipe.unload_lora_weights()
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        self.ip_adapter = ip_adapter
        if ip_adapter:
            self.pipe.load_ip_adapter(IP_REPO, subfolder="sdxl_models", weight_name=IP_WEIGHT)
        self.pipe.to(DEVICE)
        self.pipe.vae.enable_tiling()
        self._adapters: dict[str, str] = {}  # adapter name -> path (loaded)
        self.default_loras = [_lora_spec(x) for x in (loras or [])]

    def close(self) -> None:
        del self.pipe
        free_memory()

    # LoRA adapters stay loaded once used (a few MB each); switching between
    # calls is just set_adapters / disable_lora.
    MAX_ADAPTERS = 8

    def _activate(self, specs: list[dict]) -> None:
        if not specs:
            if self._adapters:
                self.pipe.disable_lora()
            return
        names = []
        for s in specs:
            p = Path(s["path"])
            key = hashlib.sha1(f"{p.resolve()}:{p.stat().st_mtime_ns}".encode()).hexdigest()[:12]
            name = f"l{key}"
            if name not in self._adapters:
                if len(self._adapters) >= self.MAX_ADAPTERS:
                    stale = [n for n in self._adapters if n not in names][:1]
                    self.pipe.delete_adapters(stale)
                    for n in stale:
                        self._adapters.pop(n)
                self.pipe.load_lora_weights(str(p), adapter_name=name)
                self._adapters[name] = str(p)
            names.append(name)
        self.pipe.enable_lora()
        self.pipe.set_adapters(names, adapter_weights=[s["weight"] for s in specs])

    def _run(self, prompt: str, size: tuple[int, int], seed: int, ref: Image.Image | None, ip_scale: float,
             negative: str = NEGATIVE, loras: Iterable | None = None) -> Image.Image:
        w, h = size
        specs = self.default_loras if loras is None else [_lora_spec(x) for x in loras]
        self._activate(specs)
        prompt = _with_triggers(prompt, specs)
        kw = {}
        if self.ip_adapter:
            self.pipe.set_ip_adapter_scale(ip_scale if ref is not None else 0.0)
            # The adapter always expects an image; a blank one at scale 0 is a no-op.
            kw["ip_adapter_image"] = ref if ref is not None else Image.new("RGB", (224, 224), "white")
        with torch.no_grad():
            pe, ne, pp, npool = self._encode(prompt, negative)
            return self.pipe(
                prompt_embeds=pe, negative_prompt_embeds=ne,
                pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=npool, width=w, height=h,
                num_inference_steps=8, guidance_scale=5.0,
                generator=torch.Generator("cpu").manual_seed(seed), **kw,
            ).images[0]

    # --- Long prompts -----------------------------------------------------------
    # CLIP sees 77 tokens; film prompts (shot + character look) run far longer
    # and the look at the end was silently cut. Encode in 75-token chunks
    # (split at a comma where possible), each wrapped in BOS/EOS, concatenate
    # the penultimate hidden states of both encoders, and take the pooled
    # embedding from the first chunk. Positive and negative are padded to the
    # same number of chunks with empty-prompt chunks.

    def _chunks(self, tok, text: str) -> list[list[int]]:
        ids = tok(text, add_special_tokens=False, truncation=False, verbose=False).input_ids
        comma = tok(",", add_special_tokens=False).input_ids[-1]
        out = []
        while len(ids) > 75:
            cut = 75
            for k in range(74, 54, -1):  # prefer ending a chunk after a comma
                if ids[k] == comma:
                    cut = k + 1
                    break
            out.append(ids[:cut])
            ids = ids[cut:]
        out.append(ids)
        return out

    def _encode_one(self, text: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.pipe
        hidden, pooled = [], None
        for tok, enc in ((p.tokenizer, p.text_encoder), (p.tokenizer_2, p.text_encoder_2)):
            chunks = self._chunks(tok, text)
            chunks += [[]] * (n - len(chunks))
            states = []
            for j, c in enumerate(chunks):
                ids = [tok.bos_token_id] + c + [tok.eos_token_id]
                ids += [tok.pad_token_id] * (77 - len(ids))
                out = enc(torch.tensor([ids], device=enc.device), output_hidden_states=True)
                states.append(out.hidden_states[-2])
                if enc is p.text_encoder_2 and j == 0:
                    pooled = out[0]
            hidden.append(torch.cat(states, dim=1))
        return torch.cat(hidden, dim=-1).to(torch.float16), pooled.to(torch.float16)

    def _encode(self, prompt: str, negative: str):
        p = self.pipe
        n = max(len(self._chunks(t, s)) for t in (p.tokenizer, p.tokenizer_2) for s in (prompt, negative))
        pe, pp = self._encode_one(prompt, n)
        ne, npool = self._encode_one(negative, n)
        return pe, ne, pp, npool

    def portrait(self, look: str, style: str, aspect: str, seed: int, setting: str = "",
                 loras: Iterable | None = None) -> Image.Image:
        # Drawn inside the film's own world and style: IP-Adapter copies the
        # reference's palette and background, so a grey studio backdrop would
        # leak grey studio backdrops into every shot.
        where = f"in {setting}" if setting else "in a softly blurred natural environment"
        prompt = (f"a single {look}, one subject only, full body, centered, clearly visible, {where}, "
                  f"shallow depth of field, sharp focus on the subject, {style}")
        return self._run(prompt, SIZES["1:1"], seed, None, 0.0, PORTRAIT_NEGATIVE + ", plain background, studio backdrop",
                         loras=loras)

    def keyframe(self, prompt: str, aspect: str, seed: int, ref: Image.Image | None,
                 loras: Iterable | None = None, ip_scale: float | None = None) -> Image.Image:
        """``loras``: list of names (see ``available_loras``), paths, (name, weight)
        tuples or dicts {"path", "trigger", "weight"}; None = engine defaults,
        [] = none. With a character LoRA active the reference image is ignored
        unless ``ip_scale`` is given (the LoRA already carries the look, and
        IP-Adapter would pull its palette/background into every shot)."""
        specs = self.default_loras if loras is None else [_lora_spec(x) for x in loras]
        if ip_scale is None:
            ip_scale = 0.0 if any(s["kind"] not in ("", "style") for s in specs) else 0.25
        return self._run(prompt, SIZES[aspect], seed, ref if ip_scale > 0 else None, ip_scale, loras=specs)


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
