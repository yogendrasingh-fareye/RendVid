# RendVid

Local text-to-video on Apple Silicon. Type an idea, get a 40–60 s video with
voice-over, subtitles and music — everything runs on your Mac.

## Run

```sh
uv run rendvid               # http://127.0.0.1:8000
```

First run downloads the models into `~/.cache/huggingface`. Put `HF_TOKEN=...`
in `.env` (needed for the gated Hindi voice).

The "brain" that writes scenes and storyboards is Claude, called through the
Claude Code CLI (`claude --print`) with your existing Claude login, no API key.
Set `RENDVID_CLAUDE_MODEL` (default `sonnet`) to change the model. Story text is
sent to Anthropic for planning; all rendering stays on this Mac.

Benchmark one scene: `uv run rendvid bench --preset max`

## How it works

1. **Scene writer** – Claude (via the Claude Code CLI) turns the idea into ~5 s
   scenes with fixed character descriptions and voice-over lines. Edit them in the UI.
2. **Prompt encoding** – T5-XXL embeds all prompts, then is freed (it doesn't
   fit in 16 GB next to the video model).
3. **Scenes** – LTX-Video 2B 0.9.8 distilled renders each scene. The last 9
   frames of a scene condition the next one, so motion and characters carry
   over; "New shot" scenes start fresh. *Max* renders at half size, upscales
   2× in latent space and refines.
4. **Audio** – Kokoro-82M voice-over (English/Hindi), your MP3/WAV or
   MusicGen-small music, auto-ducked under the voice.
5. **Assembly** – ffmpeg stitches scenes, interpolates to 48 fps and upscales
   to 1080p (*max*), burns subtitles, mixes audio.

Jobs live in `outputs/<id>/` (`final.mp4`, per-scene clips, `subtitles.srt`).
Interrupted or cancelled jobs resume from the last finished scene.

## Licenses

LTX-Video: LTX Open Weights License. Kokoro: Apache 2.0.
MusicGen weights: CC-BY-NC 4.0 (non-commercial) — use your own music for
commercial videos.
