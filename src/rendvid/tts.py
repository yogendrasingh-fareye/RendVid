"""Text-to-speech front end: Indic Parler-TTS (natural Hindi) or Kokoro (fast).

Voices are "parler:<Speaker>:<style>" or a Kokoro id like "af_heart".
Every line is cached as 44.1 kHz mono WAV under ``cache_dir`` keyed by
(text, voice), so re-exports and edits never re-record unchanged lines.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import config
from .model import free_memory

SR = 44100
TTS_PY = config.ROOT / ".tts-venv" / "bin" / "python"
WORKER = Path(__file__).with_name("parler_worker.py")

PARLER_SPEAKERS = {"Divya": "female", "Rani": "female", "Rohit": "male", "Aman": "male"}
PARLER_STYLES = {
    "storyteller": "speaks in a warm, expressive storytelling tone at a moderate pace",
    "calm": "speaks in a calm, soothing tone at a slightly slow pace",
    "dramatic": "speaks in a dramatic, emotional tone with varied pitch at a moderate pace",
    "news": "speaks in a clear, confident and neutral tone at a moderate pace",
}


def parler_voices() -> dict[str, str]:
    return {
        f"parler:{sp}:{st}": f"{sp} ({g}) · {st}"
        for sp, g in PARLER_SPEAKERS.items() for st in PARLER_STYLES
    }


def parler_ready() -> bool:
    return TTS_PY.exists()


def all_voices() -> dict[str, str]:
    voices = {k: f"Hindi HQ · {v}" for k, v in parler_voices().items()} if parler_ready() else {}
    voices.update({k: f"Kokoro · {v}" for k, v in config.VOICES.items()})
    return voices


def _key(text: str, voice: str) -> str:
    return hashlib.sha1(f"{voice}\n{text}".encode()).hexdigest()[:20]


def _to_cache(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SR),
         "-af", "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                "silenceremove=start_periods=1:start_threshold=-50dB,areverse",
         str(dst)],
        check=True,
    )


def synth_many(
    texts: list[str], voice: str, cache_dir: Path, on_progress: Callable[[int, int], None] | None = None
) -> list[Path]:
    """Record every text (cached). Returns one WAV path per text."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{_key(t, voice)}.wav" for t in texts]
    todo = [(t, p) for t, p in zip(texts, paths) if t.strip() and not p.exists()]
    todo = list(dict.fromkeys(todo))  # de-duplicate repeated lines
    if todo:
        if voice.startswith("parler:"):
            _parler(todo, voice, on_progress)
        else:
            _kokoro(todo, voice, on_progress)
    return paths


def _parler(todo, voice: str, on_progress) -> None:
    if not parler_ready():
        raise RuntimeError("Hindi HQ voice not installed (.tts-venv missing)")
    _, speaker, style = (voice.split(":") + ["storyteller"])[:3]
    desc = (f"{speaker} {PARLER_STYLES.get(style, PARLER_STYLES['storyteller'])} "
            "in a close-sounding recording with very clear audio and no background noise.")
    raw = [p.with_suffix(".raw.wav") for _, p in todo]
    job = {"description": desc, "items": [{"text": t, "out": str(r)} for (t, _), r in zip(todo, raw)]}
    proc = subprocess.Popen(
        [str(TTS_PY), str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    proc.stdin.write(json.dumps(job))
    proc.stdin.close()
    done = 0
    for line in proc.stdout:
        if line.startswith("done "):
            done += 1
            if on_progress:
                on_progress(done, len(todo))
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"Parler TTS failed: {err[-1500:]}")
    for (_, p), r in zip(todo, raw):
        _to_cache(r, p)
        r.unlink(missing_ok=True)


def _kokoro(todo, voice: str, on_progress) -> None:
    from kokoro import KPipeline

    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M", device="cpu")
    for k, (t, p) in enumerate(todo):
        parts = [r.audio.numpy() for r in pipe(t, voice=voice) if r.audio is not None]
        raw = p.with_suffix(".raw.wav")
        sf.write(raw, np.concatenate(parts) if parts else np.zeros(2400, dtype=np.float32), 24000)
        _to_cache(raw, p)
        raw.unlink(missing_ok=True)
        if on_progress:
            on_progress(k + 1, len(todo))
    del pipe
    free_memory()


def duration(path: Path) -> float:
    return sf.info(path).duration


def load(path: Path, speed: float = 1.0) -> np.ndarray:
    """Load a cached line, optionally sped up (pitch-preserving atempo)."""
    if abs(speed - 1.0) < 0.01:
        return sf.read(path, dtype="float32")[0]
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-af", f"atempo={speed:.3f}",
         "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()
