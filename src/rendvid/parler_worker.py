"""Indic Parler-TTS worker, run with the isolated .tts-venv interpreter.

Parler pins transformers==4.46.1, which the video stack can't use, so it lives
in its own venv and talks to the app over stdin/stdout:

  stdin:  {"description": str, "items": [{"text": str, "out": path}, ...]}
  stdout: one line per finished item: "done <index>"

Standalone on purpose: imports nothing from rendvid.
"""

import json
import re
import sys

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

REPO = "ai4bharat/indic-parler-tts"


def chunks(text: str, max_chars: int = 180) -> list[str]:
    """Parler degrades on long inputs: split at sentence/clause marks."""
    parts = [p.strip() for p in re.split(r"(?<=[।!?.,;])\s+", text) if p.strip()]
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out or [text]


def main() -> None:
    job = json.loads(sys.stdin.read())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(REPO).to(device)
    tok = AutoTokenizer.from_pretrained(REPO)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    desc = desc_tok(job["description"], return_tensors="pt").to(device)
    gap = np.zeros(int(sr * 0.18), dtype=np.float32)

    for i, item in enumerate(job["items"]):
        pieces = []
        for text in chunks(item["text"]):
            torch.manual_seed(1234)  # same seed every line: steadier timbre
            prompt = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                audio = model.generate(
                    input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                    prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask,
                )
            pieces += [audio.cpu().float().numpy().squeeze(), gap]
        sf.write(item["out"], np.concatenate(pieces[:-1]), sr)
        print(f"done {i}", flush=True)


if __name__ == "__main__":
    main()
