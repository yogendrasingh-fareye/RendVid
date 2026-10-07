"""Narrated-story planner (Studio mode).

Produces one plan shape from either source:

  {"title", "language", "characters": [{"name", "look"}], "setting",
   "lines": [{"narration", "visual", "characters": [names], "new_shot", "chapter"}],
   "chapters": [str]}

- From a script: the user's text is narrated verbatim. It is split into
  lines at sentence marks (। ? ! .), and the LLM only writes the visuals.
- From an idea: the LLM writes a story bible, a chapter outline, then the
  narration + visuals chapter by chapter (keeps a 5-10 min story coherent).

Narration may be Hindi; visuals are always English (what SDXL/LTX understand).
"""

from __future__ import annotations

import re

from . import config
from .planner import _chat, llm_available

WORDS_PER_MIN = {"Hindi": 125, "English": 140}


# --- script -> lines -------------------------------------------------------------

def split_script(script: str, max_chars: int = 190, min_chars: int = 28) -> list[dict]:
    """Paragraphs become chapters; sentences become narration lines."""
    lines = []
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", script) if p.strip()]
    for ch, para in enumerate(paras):
        sents = [s.strip() for s in re.split(r"(?<=[।!?.])\s+", para) if s.strip()]
        merged: list[str] = []
        for s in sents:
            if merged and (len(merged[-1]) < min_chars or len(s) < min_chars // 2) and len(merged[-1]) + len(s) < max_chars:
                merged[-1] = f"{merged[-1]} {s}"
            else:
                merged.append(s)
        for s in merged:
            # Over-long sentence: split at commas so each shot stays short.
            if len(s) > max_chars:
                parts, cur = re.split(r"(?<=[,،])\s+", s), ""
                for p in parts:
                    if cur and len(cur) + len(p) > max_chars:
                        lines.append({"narration": cur, "chapter": ch})
                        cur = p
                    else:
                        cur = f"{cur} {p}".strip()
                if cur:
                    lines.append({"narration": cur, "chapter": ch})
            else:
                lines.append({"narration": s, "chapter": ch})
    return lines


BIBLE = """You are the director of an illustrated narrated film. Read the story and fix
its visual identity. Return ONLY JSON:
{"title": str, "characters": [{"name": str, "look": str}], "setting": str, "look": str}
- "title": short title in the story's own language.
- "characters": ONLY characters, animals or key props that are actually named or described
  in the story text (max 4). Never invent people or creatures that the text does not mention.
  If the text describes only a scene or object (e.g. a planet), list just that subject, or
  return []. Never materials or substances (water, stones, sand, fire).
  "name": a short English id taken from the text. "look": ENGLISH, 15-30 words, concrete and
  fixed: species/age, body, colors, clothing, distinctive marks. Never vague.
- Stay faithful to the text's world: translate objects exactly, never substitute a different
  object. A story written in Hindi about everyday life is set in India unless the text says
  otherwise (घड़ा/मटका = round earthen clay pot). Space, fantasy or other settings stay as written.
- "setting": ENGLISH, where/when the story happens.
- "look": ENGLISH, one line on art direction: palette, lighting, mood.
"""

SHOTS = """You storyboard a narrated film. For each narration line write the shot shown
while it is spoken. Return ONLY JSON: {{"shots": [{{"visual": str, "characters": [str], "new_shot": bool}}]}}
with exactly {n} shots, in order.
Characters (use these exact descriptions whenever they appear): {characters}
Setting: {setting}. Art direction: {look}.
- Refer to characters and props by their fixed description; never swap in a different object.
- Show only what the narration and character list contain; do not add new people or creatures.
- "visual": ENGLISH, 40-70 words, present tense: subject and action first, then pose and
  expression, environment, camera framing (wide/medium/close-up) and angle, lighting.
  Show what the line says; literal and visual; no text, no speech, no sound.
- Vary framing across shots (establishing wide, medium, close-up details).
- "characters": names (from the list) visible in the shot.
- "new_shot": false only if the shot directly continues the previous one (same framing,
  same moment); true otherwise.
"""


def _bible(text: str, style: str) -> dict:
    data = _chat(BIBLE, f"Visual style: {config.STYLES.get(style) or 'natural'}\nStory:\n{text[:6000]}")
    chars = [
        {"name": str(c.get("name", "")).strip()[:30], "look": str(c.get("look", "")).strip()[:300]}
        for c in data.get("characters", []) if str(c.get("name", "")).strip()
    ][:4]
    return {"title": str(data.get("title", "")).strip(), "characters": chars,
            "setting": str(data.get("setting", "")).strip(), "look": str(data.get("look", "")).strip()}


def _shots(lines: list[dict], bible: dict, batch: int = 6) -> None:
    chars = "; ".join(f"{c['name']}: {c['look']}" for c in bible["characters"]) or "none"
    names = {c["name"].lower(): c["name"] for c in bible["characters"]}
    for b0 in range(0, len(lines), batch):
        part = lines[b0 : b0 + batch]
        prev = f"Previous shot: {lines[b0 - 1]['visual']}\n" if b0 else ""
        user = prev + "Narration lines:\n" + "\n".join(f"{k + 1}. {ln['narration']}" for k, ln in enumerate(part))
        data = _chat(SHOTS.format(n=len(part), characters=chars, setting=bible["setting"], look=bible["look"]),
                     user)
        shots = data.get("shots") or []
        for k, ln in enumerate(part):
            s = shots[k] if k < len(shots) else (shots[-1] if shots else {})
            ln["visual"] = str(s.get("visual", "")).strip() or ln.get("visual", "")
            ln["characters"] = [names[n.lower()] for n in s.get("characters", []) if isinstance(n, str) and n.lower() in names]
            ln["new_shot"] = bool(s.get("new_shot", True))


def plan_from_script(script: str, *, language: str, style: str) -> dict:
    lines = split_script(script)
    if not lines:
        raise ValueError("The script is empty")
    if not llm_available():
        for ln in lines:
            ln.update(visual=f"Illustration of: {ln['narration']}", characters=[], new_shot=True)
        return _finish({"title": script.strip().split("\n")[0][:40], "characters": [], "setting": "", "look": ""},
                       lines, language, warning="No AI brain available (the Claude CLI): visuals are placeholders. Write them yourself.")
    bible = _bible(script, style)
    _shots(lines, bible)
    # Drop characters no shot uses (small LLMs sometimes add unrelated ones).
    used = {n for ln in lines for n in ln["characters"]}
    bible["characters"] = [c for c in bible["characters"] if c["name"] in used]
    return _finish(bible, lines, language)


# --- idea -> story -------------------------------------------------------------

OUTLINE = """You are a screenwriter. Plan a narrated story film of about {minutes} minutes in {lang}.
Return ONLY JSON: {{"title": str, "characters": [{{"name": str, "look": str}}], "setting": str,
"look": str, "chapters": [{{"title": str, "summary": str}}]}}
- "title" and chapter titles in {lang}{script_note}.
- {n} chapters with a clear beginning, rising conflict, climax and a satisfying ending.
  "summary": 2-3 sentences in English.
- "characters": max 4 recurring characters/animals/objects. "name": short English id.
  "look": ENGLISH, 15-30 words, fixed concrete appearance (age, body, colors, clothing).
- "setting" and "look" (art direction: palette, lighting, mood) in ENGLISH.
"""

CHAPTER = """Write chapter {idx} of {total} of a narrated story film in {lang}{script_note}.
Return ONLY JSON: {{"lines": [{{"narration": str, "visual": str, "characters": [str], "new_shot": bool}}]}}
Story so far: {so_far}
This chapter: "{title}": {summary}
Characters (exact descriptions): {characters}
Setting: {setting}. Art direction: {look}.
- About {words} words of narration in total, split into {n_lines} lines of 10-22 words.
  Narration is spoken by a storyteller: vivid, simple, emotional, natural {lang}.
- "visual": ENGLISH, 40-70 words: the shot shown while the line is spoken: subject and action,
  pose, environment, framing (wide/medium/close-up), angle, lighting. No text or speech.
- "characters": names visible in the shot. "new_shot": false only to continue the previous shot.
"""


def plan_from_idea(idea: str, *, minutes: float, language: str, style: str) -> dict:
    if not llm_available():
        raise ValueError("Writing a story from an idea needs an AI brain (the Claude CLI), or paste your own script.")
    note = " (Devanagari script)" if language == "Hindi" else ""
    n_ch = max(3, min(10, round(minutes * 1.4)))
    out = _chat(OUTLINE.format(minutes=minutes, lang=language, script_note=note, n=n_ch),
                f"Idea: {idea}\nVisual style: {config.STYLES.get(style) or 'natural'}")
    bible = {
        "title": str(out.get("title", "")).strip() or idea[:40],
        "characters": [{"name": str(c.get("name", "")).strip()[:30], "look": str(c.get("look", "")).strip()[:300]}
                       for c in out.get("characters", []) if str(c.get("name", "")).strip()][:4],
        "setting": str(out.get("setting", "")).strip(), "look": str(out.get("look", "")).strip(),
    }
    chapters = [c for c in out.get("chapters", []) if isinstance(c, dict)][:n_ch] or [{"title": bible["title"], "summary": idea}]
    words = WORDS_PER_MIN.get(language, 130) * minutes / len(chapters)
    names = {c["name"].lower(): c["name"] for c in bible["characters"]}
    chars = "; ".join(f"{c['name']}: {c['look']}" for c in bible["characters"]) or "none"
    lines: list[dict] = []
    so_far = "This is the beginning."
    for k, ch in enumerate(chapters):
        data = _chat(CHAPTER.format(
            idx=k + 1, total=len(chapters), lang=language, script_note=note, so_far=so_far,
            title=ch.get("title", ""), summary=ch.get("summary", ""), characters=chars,
            setting=bible["setting"], look=bible["look"], words=round(words), n_lines=max(2, round(words / 16)),
        ), f"Idea: {idea}")
        for ln in data.get("lines", []):
            if str(ln.get("narration", "")).strip():
                lines.append({
                    "narration": str(ln["narration"]).strip(), "visual": str(ln.get("visual", "")).strip(),
                    "characters": [names[n.lower()] for n in ln.get("characters", []) if isinstance(n, str) and n.lower() in names],
                    "new_shot": bool(ln.get("new_shot", True)), "chapter": k,
                })
        so_far = " ".join(c.get("summary", "") for c in chapters[: k + 1])[-1200:]
    if not lines:
        raise ValueError("The LLM returned no narration; try again or paste a script.")
    result = _finish(bible, lines, language)
    result["chapters"] = [str(c.get("title", "")) for c in chapters]
    return result


def _finish(bible: dict, lines: list[dict], language: str, warning: str | None = None) -> dict:
    if lines:
        lines[0]["new_shot"] = True
    for ln in lines:
        ln.setdefault("characters", [])
        ln.setdefault("new_shot", True)
        ln.setdefault("visual", "")
    out = {**bible, "language": language, "lines": lines,
           "chapters": sorted({ln["chapter"] for ln in lines}), "source": "llm" if not warning else "template"}
    if warning:
        out["warning"] = warning
    out["estimated_seconds"] = round(sum(len(ln["narration"].split()) for ln in lines)
                                     / WORDS_PER_MIN.get(language, 130) * 60 + 0.5 * len(lines))
    return out
