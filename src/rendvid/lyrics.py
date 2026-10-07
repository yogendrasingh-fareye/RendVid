"""Song transcription with faster-whisper (CPU, int8) for lyric-aware scenes."""

from __future__ import annotations

from pathlib import Path

from . import config
from .model import free_memory


def transcribe(path: Path) -> dict:
    """Return ``{"language": str, "segments": [[start, end, text], ...]}``."""
    from faster_whisper import WhisperModel

    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    try:
        segs, info = model.transcribe(str(path), vad_filter=True, beam_size=5, word_timestamps=True)
        segments = _lines([w for s in segs for w in (s.words or [])])
    finally:
        del model
        free_memory()
    return {"language": info.language, "segments": segments}


def _lines(words, max_words: int = 8, gap: float = 0.6) -> list:
    """Regroup words into subtitle-sized lines: break on pauses, sentence ends
    and length (whisper segments can span a whole verse)."""
    lines, cur = [], []
    for w in words:
        if cur and (w.start - cur[-1].end > gap or len(cur) >= max_words):
            lines.append(cur)
            cur = []
        cur.append(w)
        if w.word.strip().endswith((".", "!", "?", ",")) and len(cur) >= 3:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        text = "".join(w.word for w in ln).strip()
        if text:
            out.append([round(float(ln[0].start), 2), round(float(ln[-1].end), 2), text])
    return out


def in_window(segments: list, start: float, end: float) -> str:
    """Lyrics overlapping ``[start, end)``."""
    return " ".join(t for s, e, t in segments if s < end and e > start)
