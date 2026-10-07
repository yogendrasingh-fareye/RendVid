"""Turn one idea into a scene-by-scene script.

The "brain" is Claude through the Claude Code CLI (headless ``claude --print``,
using the machine's existing Claude login, the same way the cowork project
does). Without the CLI, a template fills in scenes so the app still works.
RENDVID_CLAUDE_MODEL picks the model (default sonnet). Story text is sent to
Anthropic when planning; rendering stays fully local.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

from . import config

SYSTEM = """You are a film director writing shot lists for an AI video model.
The model renders each scene as a separate ~{secs:.0f}-second clip, so every
scene prompt must stand alone. Rules:
- Return ONLY JSON: {{"title": str, "characters": str, "scenes": [{{"prompt": str, "narration": str, "new_shot": bool}}]}}
- Exactly {n} scenes, in story order.
- "characters": one sentence fixing the look of every recurring character/object
  (age, clothing, colors, hair). Copy those exact words into every scene prompt where they appear.
- Each "prompt": 50-90 words, one flowing paragraph, present tense. Describe the main
  action first, then specific movements and gestures, then appearance, then the
  environment, then camera angle/movement, then lighting and colors. Literal and visual;
  no dialogue, no on-screen text, no sound descriptions.
- Keep motion simple and continuous inside a scene; one action per scene.
- Stay strictly on the idea. Never invent characters, people or plot the idea does not
  mention. If the idea is a pure visual description (a planet, a landscape, an object),
  every scene shows that subject from different angles and moments, with no people.
- "new_shot": false when the scene continues directly from the previous one (same place,
  same moment); true for a cut to a new location, time or camera angle. Scene 1 is true.
  Prefer continuity: use at most one new_shot every 3 scenes.
- "narration": {narration_rule}
"""

NARRATION_ON = "a voice-over line of 8-14 words that fits inside the scene, in {lang}."
NARRATION_OFF = 'always "".'


def brain() -> str | None:
    """"claude" when the Claude Code CLI is installed, else None (template only)."""
    return "claude" if shutil.which("claude") else None


def brain_label() -> str:
    return f"Claude {os.environ.get('RENDVID_CLAUDE_MODEL', 'sonnet')}" if brain() else "none"


def llm_available() -> bool:
    return brain() is not None


def _chat(system: str, user: str) -> dict:
    """One JSON reply from the brain."""
    return _claude(system, user)


def _claude(system: str, user: str) -> dict:
    """Headless Claude Code call: no tools, no MCP, no saved session, run outside
    any project so no CLAUDE.md is picked up. Uses the existing Claude login."""
    args = [
        "claude", "--print", user,
        "--system-prompt", system + "\n\nReply with ONLY one valid JSON object. No prose, no code fences.",
        "--model", os.environ.get("RENDVID_CLAUDE_MODEL", "sonnet"),
        "--output-format", "json", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
    ]
    p = subprocess.run(args, cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=600)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Claude CLI failed: {(p.stderr or p.stdout)[-500:]}")
    if out.get("is_error"):
        raise RuntimeError(f"Claude: {str(out.get('result'))[:500]}")
    return _parse_json(out.get("result") or "")


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def plan(idea: str, duration: int, *, narration: bool, language: str = "English", style: str = "none") -> dict:
    n = config.scene_count(duration)
    if llm_available():
        try:
            return _plan_llm(idea, n, narration, language, style)
        except Exception as e:  # malformed JSON, timeout...
            result = _plan_template(idea, n, narration)
            result["warning"] = f"LLM planner failed ({e}); used simple template."
            return result
    result = _plan_template(idea, n, narration)
    result["warning"] = "No AI brain available (the Claude CLI); used simple template. Edit scenes before rendering."
    return result


def _plan_llm(idea: str, n: int, narration: bool, language: str, style: str) -> dict:
    secs = (config.CLIP_FRAMES - config.OVERLAP_FRAMES) / config.FPS
    rule = NARRATION_ON.format(lang=language) if narration else NARRATION_OFF
    system = SYSTEM.format(n=n, secs=secs, narration_rule=rule)
    style_hint = config.STYLES.get(style, "")
    user = f"Idea: {idea}\nVisual style: {style_hint or 'natural'}\nWrite the {n} scenes."
    data = _chat(system, user)
    scenes = [
        {
            "prompt": str(s.get("prompt", "")).strip(),
            "narration": str(s.get("narration", "")).strip() if narration else "",
            "new_shot": bool(s.get("new_shot", False)),
        }
        for s in data.get("scenes", [])
        if str(s.get("prompt", "")).strip()
    ]
    if not scenes:
        raise ValueError("no scenes returned")
    scenes = (scenes + scenes[-1:] * n)[:n]  # pad/trim to exactly n
    scenes[0]["new_shot"] = True
    return {
        "title": data.get("title") or idea[:60],
        "characters": data.get("characters", ""),
        "scenes": scenes,
        "source": "llm",
    }


def _plan_template(idea: str, n: int, narration: bool) -> dict:
    beats = ["The scene opens", "The action continues", "The moment builds", "The story unfolds", "The scene concludes"]
    scenes = []
    for i in range(n):
        beat = beats[0] if i == 0 else beats[-1] if i == n - 1 else beats[1 + (i - 1) % 3]
        scenes.append({
            "prompt": f"{beat}: {idea}. Smooth continuous motion, consistent characters and setting.",
            "narration": "",
            "new_shot": i == 0,
        })
    return {"title": idea[:60], "characters": "", "scenes": scenes, "source": "template"}


# --- Music video -------------------------------------------------------------

SONG_BIBLE = """You are the director of a music video. From the brief and lyrics,
fix the visual identity that every shot will share.
Return ONLY JSON: {"title": str, "characters": str, "look": str}
- "characters": one sentence fixing the exact look of each recurring person/object
  (age, clothing, colors, hair). Empty string if the video has no recurring characters.
- "look": one sentence for setting, color palette, lighting and camera style.
"""

SONG_BATCH = """You are the director of a music video shot as ~{secs:.0f}-second clips
by an AI video model. Each clip prompt must stand alone.
Characters (copy these exact words whenever they appear): {characters}
Look (apply to every scene): {look}
Return ONLY JSON: {{"scenes": [{{"prompt": str, "new_shot": bool}}]}} with exactly {n} scenes, in order.
- Each "prompt": 50-90 words, one paragraph, present tense: main action, movements,
  appearance, environment, camera angle/movement, lighting. Visual only; no text, no sound.
- Match the mood and imagery of the lyrics playing during that scene, and the user's brief.
  Instrumental parts: atmospheric shots, landscapes, details, performance.
- One simple continuous action per scene. "new_shot": true for a cut to a new place, angle
  or moment (good on a new verse or chorus), false to continue the previous shot.
"""


def plan_song(context: str, duration: float, segments: list, *, style: str = "none") -> dict:
    from .lyrics import in_window

    n = config.scene_count(duration)
    windows = config.scene_windows(n)
    lyric = [in_window(segments, s, e) for s, e in windows]
    if not llm_available():
        result = _song_template(context, n, lyric)
        result["warning"] = "No AI brain available (the Claude CLI); used simple template. Edit scenes before rendering."
        return result
    try:
        return _song_llm(context, n, windows, lyric, segments, style)
    except Exception as e:
        result = _song_template(context, n, lyric)
        result["warning"] = f"LLM planner failed ({e}); used simple template."
        return result


def _song_llm(context, n, windows, lyric, segments, style, batch=8) -> dict:
    style_hint = config.STYLES.get(style, "") or "natural"
    all_lyrics = " / ".join(t for _, _, t in segments)[:3000] or "(instrumental)"
    bible = _chat(SONG_BIBLE, f"Brief: {context}\nVisual style: {style_hint}\nLyrics: {all_lyrics}")
    characters = str(bible.get("characters", "")).strip() or "none"
    look = f"{str(bible.get('look', '')).strip()} {style_hint}".strip()
    secs = (config.CLIP_FRAMES - config.OVERLAP_FRAMES) / config.FPS
    scenes: list[dict] = []
    for b0 in range(0, n, batch):
        idx = list(range(b0, min(n, b0 + batch)))
        last = len(idx) - 1 if b0 + batch >= n else None
        lines = [
            f"Scene {i + 1} ({windows[i][0]:.0f}-{windows[i][1]:.0f}s): lyrics: {lyric[i] or '(instrumental)'}"
            for i in idx
        ]
        prev = f"Previous scene: {scenes[-1]['prompt']}\n" if scenes else "This is the opening of the video.\n"
        user = (f"Brief: {context}\n{prev}"
                + ("The final scene should feel like an ending.\n" if last is not None else "")
                + "Write these scenes:\n" + "\n".join(lines))
        system = SONG_BATCH.format(secs=secs, characters=characters, look=look, n=len(idx))
        data = _chat(system, user)
        got = [
            {"prompt": str(s.get("prompt", "")).strip(), "narration": "", "new_shot": bool(s.get("new_shot", False))}
            for s in data.get("scenes", []) if str(s.get("prompt", "")).strip()
        ]
        if not got:
            raise ValueError(f"no scenes returned for scenes {idx[0] + 1}-{idx[-1] + 1}")
        scenes += (got + got[-1:] * len(idx))[: len(idx)]
    scenes[0]["new_shot"] = True
    return {
        "title": bible.get("title") or context[:60],
        "characters": characters if characters != "none" else "",
        "scenes": scenes,
        "source": "llm",
    }


def _song_template(context: str, n: int, lyric: list[str]) -> dict:
    scenes = [
        {
            "prompt": f"{context}. {('Imagery: ' + lyric[i] + '.') if lyric[i] else 'Atmospheric instrumental moment.'} "
                      "Smooth continuous motion, consistent characters and setting.",
            "narration": "",
            "new_shot": i % 4 == 0,
        }
        for i in range(n)
    ]
    return {"title": context[:60], "characters": "", "scenes": scenes, "source": "template"}
