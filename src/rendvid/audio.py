"""Voice-over (Kokoro), generated music (MusicGen) and subtitle timing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from . import config
from .model import free_memory


def synth_narration(
    lines: list[tuple[float, float, str]], voice: str, total_s: float, out_wav: Path,
    cache_dir: Path | None = None, lead: float = 0.15,
) -> list[tuple[float, float, str]]:
    """Render narration lines onto one track.

    ``lines`` is ``(start, room, text)``: each line starts ``lead`` s after
    ``start`` and is sped up (max 1.3x, pitch kept) if longer than ``room``.
    Lines are recorded through ``tts`` and cached in ``cache_dir``.
    Returns subtitle cues ``(start, end, text)``.
    """
    from . import tts

    cache_dir = cache_dir or out_wav.parent / "tts"
    lines = [(s, d, t.strip()) for s, d, t in lines if t.strip()]
    paths = tts.synth_many([t for _, _, t in lines], voice, cache_dir)
    sr = tts.SR
    track = np.zeros(int((total_s + 2) * sr), dtype=np.float32)
    cues: list[tuple[float, float, str]] = []
    cursor = 0.0
    for (start, room, text), path in zip(lines, paths):
        length = tts.duration(path)
        avail = max(room - lead - 0.1, 0.8)
        speed = min(1.3, length / avail) if length > avail else 1.0
        audio = tts.load(path, speed)
        t0 = max(start + lead, cursor + 0.1)
        i0 = int(t0 * sr)
        if i0 >= len(track):
            break
        seg = audio[: len(track) - i0]
        track[i0 : i0 + len(seg)] += seg
        t1 = t0 + len(seg) / sr
        cues += _split_cues(text, t0, t1)
        cursor = t1
    sf.write(out_wav, np.clip(track[: int(total_s * sr)], -1, 1), sr)
    return cues


def _split_cues(text: str, t0: float, t1: float, max_words: int = 7) -> list[tuple[float, float, str]]:
    """Chunk a line into short captions timed proportionally to their length.

    Breaks at sentence/clause marks first (। , ? !), then balances long
    phrases by word count, so captions never straddle two sentences."""
    import re

    chunks = []
    for phrase in (p.strip() for p in re.split(r"(?<=[।,!?.;])\s+", text) if p.strip()):
        words = phrase.split()
        n = -(-len(words) // max_words)
        size = -(-len(words) // n)
        chunks += [" ".join(words[i : i + size]) for i in range(0, len(words), size)]
    # Glue a tiny trailing fragment onto its neighbour.
    merged: list[str] = []
    for c in chunks:
        if merged and (len(c.split()) <= 2 or len(merged[-1].split()) <= 1) and len((merged[-1] + " " + c).split()) <= max_words + 2:
            merged[-1] = f"{merged[-1]} {c}"
        else:
            merged.append(c)
    total = sum(len(c) for c in merged) or 1
    cues, t = [], t0
    for c in merged:
        d = (t1 - t0) * len(c) / total
        cues.append((t, t + d, c))
        t += d
    return cues


def generate_music(prompt: str, out_wav: Path, seconds: int = 30) -> None:
    """Generate a loopable music bed with MusicGen-small (CC-BY-NC weights)."""
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    proc = AutoProcessor.from_pretrained(config.MUSICGEN_REPO)
    model = MusicgenForConditionalGeneration.from_pretrained(config.MUSICGEN_REPO)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model.to(device)
        inputs = proc(text=[prompt], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            audio = model.generate(**inputs, max_new_tokens=int(seconds * 50), do_sample=True, guidance_scale=3.0)
    except Exception:  # fall back to CPU if an op is missing on MPS
        model.to("cpu")
        inputs = proc(text=[prompt], padding=True, return_tensors="pt")
        with torch.no_grad():
            audio = model.generate(**inputs, max_new_tokens=int(seconds * 50), do_sample=True, guidance_scale=3.0)
    sr = model.config.audio_encoder.sampling_rate
    sf.write(out_wav, audio[0, 0].float().cpu().numpy(), sr)
    del model, proc
    free_memory()
