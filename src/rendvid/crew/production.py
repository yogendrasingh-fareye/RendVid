"""A production: one chat thread in which the crew makes (and keeps changing) one film.

Stages:
  briefing -> awaiting_brief (user approves) -> planning -> awaiting_storyboard
  (user approves) -> rendering (keyframes checked by the QA Critic inside the
  render) -> reviewing (QA on the finished shots) -> done.
At any point the user can chat; the Producer answers questions and routes
change requests to the right members.

Stored as outputs/.crew/productions/<id>.json.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from .. import config, tts
from ..studio import split_script
from . import runner
from .members import CREW

PROD_DIR = runner.CREW_DIR / "productions"
_locks: dict[str, threading.Lock] = {}
_busy: set[str] = set()
_jobs = None  # JobManager, injected by the server
_submit: Callable[[dict], dict] | None = None


def setup(jobs, submit: Callable[[dict], dict]) -> None:
    global _jobs, _submit
    _jobs, _submit = jobs, submit
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    runner.init_memory(CREW)
    # Productions that were mid-render when the server stopped: keep watching.
    for p in list_all():
        if p.get("stage") in ("rendering", "reviewing") and p.get("job_id"):
            threading.Thread(target=_watch_render, args=(p["id"],), daemon=True).start()
        elif p.get("busy"):
            _update(p["id"], busy=False, activity="")


# --- storage --------------------------------------------------------------------

def _path(pid: str) -> Path:
    return PROD_DIR / f"{pid}.json"


def _lock(pid: str) -> threading.Lock:
    return _locks.setdefault(pid, threading.Lock())


def load(pid: str) -> dict | None:
    p = _path(pid)
    return json.loads(p.read_text()) if p.exists() else None


def _save(prod: dict) -> None:
    prod["updated"] = time.time()
    tmp = _path(prod["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(prod, ensure_ascii=False, indent=1))
    tmp.replace(_path(prod["id"]))


def _update(pid: str, **fields) -> dict:
    with _lock(pid):
        prod = load(pid)
        prod.update(fields)
        _save(prod)
        return prod


def list_all() -> list[dict]:
    out = []
    for f in PROD_DIR.glob("*.json"):
        try:
            p = json.loads(f.read_text())
            out.append({k: p.get(k) for k in ("id", "title", "stage", "updated", "created", "job_id", "busy", "activity")})
        except json.JSONDecodeError:
            pass
    return sorted(out, key=lambda p: p["updated"], reverse=True)


def post(pid: str, sender: str, text: str, kind: str = "text", data: dict | None = None) -> None:
    with _lock(pid):
        prod = load(pid)
        prod["messages"].append({"id": uuid.uuid4().hex[:8], "from": sender, "text": text, "kind": kind,
                                 "data": data or {}, "ts": time.time()})
        _save(prod)


def _activity(pid: str, text: str) -> None:
    _update(pid, activity=text)


# --- entry points ----------------------------------------------------------------

def create(text: str) -> dict:
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    prod = {"id": pid, "title": text.strip().split("\n")[0][:60] or "New film", "created": time.time(),
            "stage": "briefing", "busy": False, "activity": "", "messages": [], "cost": 0.0}
    _save(prod)
    user_message(pid, text)
    return load(pid)


def user_message(pid: str, text: str) -> None:
    post(pid, "user", text)
    _spawn(pid, _producer_turn, text)


def approve(pid: str, what: str) -> None:
    prod = load(pid)
    if what == "brief" and prod["stage"] == "awaiting_brief":
        post(pid, "user", "✓ Brief approved")
        _spawn(pid, _plan)
    elif what == "storyboard" and prod["stage"] == "awaiting_storyboard":
        post(pid, "user", "✓ Storyboard approved, roll camera")
        _spawn(pid, _render)
    else:
        raise ValueError("Nothing to approve right now")


def _spawn(pid: str, fn, *args) -> None:
    if pid in _busy:
        raise ValueError("The crew is still working on the last step; wait a moment")

    def work():
        _busy.add(pid)
        _update(pid, busy=True)
        try:
            fn(pid, *args)
        except Exception as e:
            traceback.print_exc()
            post(pid, "system", f"Something went wrong: {e}", "error")
        finally:
            _busy.discard(pid)
            _update(pid, busy=False, activity="")

    threading.Thread(target=work, daemon=True).start()


# --- member calls --------------------------------------------------------------------

def _call(pid: str, key: str, prompt: str, activity: str, read_dir: Path | None = None) -> dict:
    m = CREW[key]
    _activity(pid, f"{m['role']} {activity}")
    out = runner.run(key, m, prompt, m["schema"], read_dir=read_dir)
    runner.save_memories(key, m["role"], pid, out.pop("memories", None))
    with _lock(pid):
        prod = load(pid)
        prod["cost"] = round(prod.get("cost", 0) + (out.get("_cost") or 0), 4)
        _save(prod)
    return out


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=1)


def _storyboard_text(prod: dict, visuals: bool = True) -> str:
    lines = (prod.get("script") or {}).get("lines") or []
    shots = {s["line"]: s for s in prod.get("shots") or []}
    out = []
    for i, ln in enumerate(lines, 1):
        row = f"{i}. [ch {ln.get('chapter', 0)}] {ln.get('narration') or '(silent) ' + ln.get('beat', '')}"
        if visuals and i in shots:
            row += f"\n   SHOT: {shots[i]['visual']} | chars: {', '.join(shots[i].get('characters', []))}"
        out.append(row)
    return "\n".join(out)


# --- producer ----------------------------------------------------------------------------

def _producer_turn(pid: str, text: str) -> None:
    prod = load(pid)
    state = [f"Stage: {prod['stage']}"]
    if prod.get("brief"):
        state.append(f"Current brief:\n{_j(prod['brief'])}")
    if prod.get("shots"):
        state.append(f"Storyboard (lines numbered from 1):\n{_storyboard_text(prod)}")
    if prod.get("job_id"):
        job = _jobs.load(prod["job_id"]) or {}
        state.append(f"Render job: {job.get('status')} ({round(100 * job.get('progress', 0))}%) {job.get('message', '')}")
    history = "\n".join(f"{m['from']}: {m['text'][:600]}" for m in prod["messages"][-14:])
    prompt = (
        "PRODUCTION STATE\n" + "\n\n".join(state) +
        f"\n\nRECENT CHAT\n{history}\n\nNEW MESSAGE FROM THE USER\n{text}\n\n"
        "If the brief is awaiting approval and the user adjusts it, return kind \"production\" with the "
        "full updated brief. If no film exists yet, any request to make something is kind \"production\"."
    )
    out = _call(pid, "producer", prompt, "is reading your message…")
    kind = out.get("kind")
    if kind == "production" and out.get("brief"):
        brief = out["brief"]
        _update(pid, brief=brief, title=brief.get("title") or prod["title"], stage="awaiting_brief",
                script=None, director=None, design=None, shots=None, sound=None, editor=None)
        post(pid, "producer", out.get("reply", ""), "brief", {"brief": brief})
    elif kind == "change" and out.get("change"):
        post(pid, "producer", out.get("reply", ""))
        _apply_change(pid, out["change"])
    else:
        post(pid, "producer", out.get("reply", ""))


# --- planning -------------------------------------------------------------------------------

def _plan(pid: str, notes: str = "") -> None:
    _update(pid, stage="planning")
    prod = load(pid)
    brief = prod["brief"]
    narrated = brief.get("format") == "narrated"

    # 1. Script
    if narrated and brief.get("source") == "script" and brief.get("script", "").strip():
        lines = split_script(brief["script"])
        script = {"title": brief.get("title", ""), "chapters": [], "lines": lines}
        post(pid, "writer", f"Your script is narrated word for word: {len(lines)} lines in "
                            f"{len({ln['chapter'] for ln in lines})} parts.")
    else:
        target = (f"about {brief.get('minutes', 1)} minutes; Hindi ≈ 125 words/min, English ≈ 140"
                  if narrated else
                  f"{max(2, round(brief.get('minutes', 0.5) * 60 / 5))} silent visual beats of about 5 s; "
                  "narration empty, describe each beat in `beat`, set `seconds`")
        out = _call(pid, "writer", f"BRIEF\n{_j(brief)}\n\nLENGTH: {target}\n"
                    + (f"USER'S MATERIAL (use it faithfully):\n{brief.get('script')}\n" if brief.get("script") else "")
                    + (f"\nNOTES: {notes}" if notes else ""), "is writing the script…")
        script = {"title": out.get("title") or brief.get("title"), "chapters": out.get("chapters", []),
                  "lines": [ln for ln in out.get("lines", []) if (ln.get("narration") or ln.get("beat") or "").strip()]}
        words = sum(len(ln.get("narration", "").split()) for ln in script["lines"])
        post(pid, "writer", f"Script ready: “{script['title']}”, {len(script['lines'])} lines"
                            + (f", {words} words" if narrated else "") + ".",
             "script", {"lines": script["lines"][:200]})
    _update(pid, script=script)
    script_text = "\n".join(f"{i}. [ch {ln.get('chapter', 0)}] {ln.get('narration') or ln.get('beat', '')}"
                            for i, ln in enumerate(script["lines"], 1))

    # 2. Director
    director = _call(pid, "director", f"BRIEF\n{_j(brief)}\n\nSCRIPT\n{script_text}", "is shaping the vision…")
    director.pop("_cost", None)
    post(pid, "director", f"{director.get('vision', '')}\nTone: {director.get('tone', '')}")
    _update(pid, director=director)

    # 3. Production design
    design = _call(pid, "designer", f"BRIEF\n{_j(brief)}\n\nDIRECTOR\n{_j(director)}\n\nSCRIPT\n{script_text}",
                   "is designing the look…")
    design["characters"] = [c for c in design.get("characters", []) if c.get("name") and c.get("look")][:5]
    post(pid, "designer", "Design bible:\n" + "\n".join(f"• {c['name']}: {c['look']}" for c in design["characters"])
         + f"\n• Setting: {design.get('setting', '')}\n• Art direction: {design.get('look', '')}")
    _update(pid, design=design)

    # 4. Cinematographer, in batches so long films stay sharp
    bible = _j({k: design.get(k) for k in ("characters", "setting", "look")})
    shots: list[dict] = []
    lines = script["lines"]
    for b0 in range(0, len(lines), 14):
        part = lines[b0:b0 + 14]
        listing = "\n".join(f"{b0 + k + 1}. {ln.get('narration') or '(silent) ' + ln.get('beat', '')}" for k, ln in enumerate(part))
        prev = f"\nPREVIOUS SHOT\n{shots[-1]['visual']}\n" if shots else ""
        out = _call(pid, "dop", f"BRIEF\n{_j(brief)}\n\nDIRECTOR\n{_j(director)}\n\nDESIGN BIBLE\n{bible}\n{prev}\n"
                    f"LINES (write exactly one shot per line, keep the line numbers)\n{listing}",
                    f"is designing shots {b0 + 1}–{b0 + len(part)}…")
        got = {s["line"]: s for s in out.get("shots", []) if isinstance(s.get("line"), int)}
        for k, ln in enumerate(part):
            n = b0 + k + 1
            s = got.get(n) or {"line": n, "visual": ln.get("beat") or ln.get("narration", ""), "characters": [], "new_shot": True}
            s["line"] = n
            shots.append(s)
    shots[0]["new_shot"] = True
    names = {c["name"] for c in design["characters"]}
    for s in shots:
        s["characters"] = [c for c in s.get("characters", []) if c in names]
    post(pid, "dop", f"Shot list ready: {len(shots)} shots.")
    _update(pid, shots=shots)

    # 5. Script supervisor
    prod = load(pid)
    review = _call(pid, "supervisor", f"DESIGN BIBLE\n{bible}\n\nDIRECTOR\n{_j(director)}\n\nSTORYBOARD\n{_storyboard_text(prod)}",
                   "is checking continuity…")
    fixes = [i for i in review.get("issues", []) if isinstance(i.get("line"), int) and 1 <= i["line"] <= len(shots)]
    for i in fixes:
        if i.get("visual"):
            shots[i["line"] - 1]["visual"] = i["visual"]
        if i.get("characters") is not None:
            shots[i["line"] - 1]["characters"] = [c for c in i["characters"] if c in names]
    post(pid, "supervisor", (f"Fixed {len(fixes)} shot(s):\n" + "\n".join(f"• Line {i['line']}: {i['problem']}" for i in fixes))
         if fixes else f"Storyboard checks out. {review.get('note', '')}")
    _update(pid, shots=shots)

    # 6. Sound + 7. Editor
    voices = tts.all_voices()
    sound = _call(pid, "sound", f"BRIEF\n{_j(brief)}\n\nDIRECTOR\n{_j(director)}\n\nAVAILABLE VOICES\n{_j(voices)}",
                  "is choosing voice and music…")
    if sound.get("voice") not in voices or not narrated:
        sound["voice"] = _default_voice(brief) if narrated else ""
    post(pid, "sound", f"Narrator: {voices.get(sound['voice'], 'none')}. Music: "
                       + (f"{sound.get('music_prompt')} (vol {sound.get('music_volume')})" if sound.get("music") == "generate" else "none")
                       + (f"\n{sound['note']}" if sound.get("note") else ""))
    editor = _call(pid, "editor", f"BRIEF\n{_j(brief)}\n\nDIRECTOR\n{_j(director)}", "is planning the cut…")
    post(pid, "editor", f"Cut: {editor.get('chapter_transition')} between chapters, {editor.get('shot_transition')} between shots, "
                        f"filter {editor.get('filter')}, subtitles {'on' if editor.get('subtitles') else 'off'}"
                        + (f", title “{editor.get('title_text')}”" if editor.get("title_card") else "") + ".")
    _update(pid, sound=sound, editor=editor, stage="awaiting_storyboard")
    prod = load(pid)
    shot_secs = {"fast": 45, "balanced": 65, "max": 105}[brief.get("quality", "balanced")]
    n_shots = _estimate_shots(prod)
    post(pid, "producer", f"Storyboard is ready: {len(shots)} lines, about {n_shots} shots. Estimated render "
                          f"≈ {math.ceil((n_shots * shot_secs + 300) / 60)} min on {brief.get('quality')}. "
                          "Approve to start, or tell me what to change.",
         "storyboard", {"rows": _rows(prod)})


def _default_voice(brief: dict) -> str:
    voices = tts.all_voices()
    for v in (["parler:Divya:storyteller", "hf_alpha"] if brief.get("language") == "Hindi" else ["af_heart"]):
        if v in voices:
            return v
    return next(iter(voices))


def _estimate_shots(prod: dict) -> int:
    n = 0
    for ln in prod["script"]["lines"]:
        secs = len(ln.get("narration", "").split()) / 2.1 + 0.5 if ln.get("narration") else ln.get("seconds", 5)
        n += max(1, math.ceil(secs / 4.8))
    return n


def _rows(prod: dict) -> list[dict]:
    lines = prod["script"]["lines"]
    return [{"line": i + 1, "chapter": ln.get("chapter", 0), "narration": ln.get("narration") or "",
             "beat": ln.get("beat", ""), "visual": s["visual"], "characters": s.get("characters", []),
             "new_shot": s.get("new_shot", True)} for i, (ln, s) in enumerate(zip(lines, prod["shots"]))]


# --- render ---------------------------------------------------------------------------------

def _spec(prod: dict) -> dict:
    brief, design, sound, editor = prod["brief"], prod["design"], prod["sound"], prod["editor"]
    return {
        "mode": "studio", "crew": prod["id"],
        "title": editor.get("title_text") or brief.get("title") or prod["title"],
        "idea": brief.get("idea", "")[:2000], "language": brief.get("language", "English"),
        "voice": sound.get("voice") or None,
        "aspect": brief.get("aspect", "16:9"), "preset": brief.get("quality", "balanced"),
        "style": design.get("style") or brief.get("style") or "none",
        "seed": random.randint(1, 99999),
        "characters": design.get("characters", []), "setting": design.get("setting", ""), "look": design.get("look", ""),
        "lines": [{"narration": r["narration"], "visual": r["visual"], "characters": r["characters"],
                   "new_shot": r["new_shot"], "chapter": r["chapter"],
                   "seconds": next((ln.get("seconds") for ln in [prod["script"]["lines"][r["line"] - 1]]), None) or 5}
                  for r in _rows(prod)],
        "subtitles": bool(editor.get("subtitles", True)), "title_card": bool(editor.get("title_card", True)),
        "music": {"mode": sound.get("music", "none"), "prompt": sound.get("music_prompt", ""),
                  "volume": float(sound.get("music_volume") or 0.18)},
        "editor": editor,
    }


def _render(pid: str) -> None:
    prod = load(pid)
    job = _submit(_spec(prod))
    _update(pid, job_id=job["id"], stage="rendering", review_after=True)
    post(pid, "producer", "Cameras rolling. I'll post progress here; the QA Critic checks every keyframe "
                          "before it is animated.", "progress", {"job_id": job["id"]})
    threading.Thread(target=_watch_render, args=(pid,), daemon=True).start()


def _watch_render(pid: str) -> None:
    while True:
        prod = load(pid)
        job = _jobs.load(prod["job_id"]) if prod and prod.get("job_id") else None
        if not job:
            return
        if job["status"] == "done":
            break
        if job["status"] in ("failed", "cancelled", "interrupted"):
            _update(pid, stage="done")
            post(pid, "system", f"Render {job['status']}: {job.get('message', '')}. Say “resume” to continue.", "error")
            return
        time.sleep(5)
    if prod.get("stage") == "rendering" and prod.get("review_after", True):
        _update(pid, stage="reviewing")
        try:
            _final_review(pid)
        except Exception as e:
            traceback.print_exc()
            post(pid, "system", f"Final review skipped: {e}", "error")
    _update(pid, stage="done")
    job = _jobs.load(prod["job_id"])
    runner.add_history(f"{prod['title']} ({prod['id']}): {len(prod['script']['lines'])} lines, "
                       f"{round((job.get('timings') or {}).get('total', 0) / 60)} min render")
    post(pid, "producer", "Your film is ready. Watch it below; tell me anything you'd like changed, "
                          "or open it in the editor.", "video",
         {"job_id": prod["job_id"], "url": f"/outputs/{prod['job_id']}/final.mp4?{int(time.time())}"})


def _final_review(pid: str) -> None:
    """QA Critic watches one frame from the middle of every finished shot."""
    prod = load(pid)
    job_dir = _jobs.dir(prod["job_id"])
    spec = _jobs.load(prod["job_id"])["spec"]
    frames = job_dir / "qa_frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir()
    from .. import stitch

    first_of_line = {}
    for k, sc in enumerate(spec["scenes"]):
        first_of_line.setdefault(sc["line"], k)
    for line, k in first_of_line.items():
        clip = job_dir / f"scene_{k + 1:02d}.mp4"
        if clip.exists():
            t = stitch.probe_duration(clip) / 2
            stitch.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(clip), "-frames:v", "1",
                        "-vf", "scale=640:-2", str(frames / f"line_{line + 1:03d}.jpg")])
    results = critic_review(pid, spec, sorted(frames.glob("*.jpg")), "is watching the finished shots…")
    bad = [r for r in results if r["verdict"] == "redo"]
    if bad:
        post(pid, "critic", f"I found {len(bad)} shot(s) worth reshooting:\n"
             + "\n".join(f"• Line {_line_of(r['shot'])}: {r['problem']}" for r in bad)
             + "\nSay “reshoot them” and I'll fix exactly these.", "report", {"bad": bad})
    else:
        post(pid, "critic", "Every finished shot holds up. No reshoots needed.")


def _line_of(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def critic_review(pid: str, spec: dict, images: list[Path], activity: str) -> list[dict]:
    """Ask the QA Critic to look at images named line_NNN.*; returns pass/redo per image."""
    looks = _j({"characters": spec.get("characters"), "setting": spec.get("setting"), "look": spec.get("look")})
    results = []
    for b0 in range(0, len(images), 10):
        batch = images[b0:b0 + 10]
        listing = "\n".join(f"- {p.name}: line {_line_of(p.name)}: “{spec['lines'][_line_of(p.name) - 1].get('narration') or ''}”\n"
                            f"  intended shot: {spec['lines'][_line_of(p.name) - 1]['visual']}" for p in batch)
        out = _call(pid, "critic", f"DESIGN BIBLE\n{looks}\n\nOpen each image file below (in {batch[0].parent}) "
                    f"with the Read tool and judge it.\n{listing}", activity, read_dir=batch[0].parent)
        by_name = {r.get("shot"): r for r in out.get("results", [])}
        for p in batch:
            r = by_name.get(p.name) or {"verdict": "pass", "problem": "", "fix": ""}
            results.append({"shot": p.name, "verdict": r.get("verdict", "pass"), "problem": r.get("problem", ""), "fix": r.get("fix", "")})
    return results


# --- keyframe QA (called from inside the render) ----------------------------------------

def review_keyframes(pid: str, spec: dict, job_dir: Path, regenerate: Callable[[list[int]], None], rounds: int = 2) -> None:
    """Critic checks new keyframes; failing ones get the critic's fix and are redrawn."""
    checked: set[str] = set()
    for rnd in range(rounds):
        kfs = [p for p in sorted((job_dir / "keyframes").glob("line_*.png")) if p.name not in checked]
        if not kfs:
            return
        results = critic_review(pid, spec, kfs, f"is checking {len(kfs)} keyframes…")
        bad = [r for r in results if r["verdict"] == "redo"]
        checked |= {r["shot"] for r in results if r["verdict"] == "pass"}
        if not bad:
            post(pid, "critic", f"Checked {len(kfs)} keyframes: all good." if rnd == 0 else "Redrawn keyframes pass.")
            return
        if rnd == rounds - 1:
            post(pid, "critic", f"{len(bad)} keyframe(s) still imperfect after redraws; keeping the best attempt:\n"
                 + "\n".join(f"• Line {_line_of(r['shot'])}: {r['problem']}" for r in bad))
            return
        post(pid, "critic", f"Redrawing {len(bad)} keyframe(s):\n"
             + "\n".join(f"• Line {_line_of(r['shot'])}: {r['problem']}" for r in bad))
        lines = []
        for r in bad:
            n = _line_of(r["shot"])
            if r.get("fix"):
                spec["lines"][n - 1]["visual"] = r["fix"]
            lines.append(n - 1)
        regenerate(lines)


# --- changes ------------------------------------------------------------------------------------

def _apply_change(pid: str, change: dict) -> None:
    prod = load(pid)
    stage = prod["stage"]
    if stage in ("briefing", "awaiting_brief", "planning"):
        return
    if stage == "awaiting_storyboard":
        if change.get("script"):
            _plan(pid, notes=change["script"])
            return
        _reshoot_storyboard(pid, change.get("reshoot") or [])
        if change.get("sound") or change.get("edit"):
            _restyle(pid, change, rendered=False)
        prod = load(pid)
        post(pid, "producer", "Updated storyboard. Approve to render, or keep refining.", "storyboard", {"rows": _rows(prod)})
        return
    # Film exists: change it surgically, then re-render only what changed.
    if change.get("script"):
        post(pid, "producer", "Changing the narration words means re-planning the film. Start a new film in a "
                              "new chat (or tell me to rebuild this one from scratch).")
        return
    job = _jobs.load(prod["job_id"])
    if job["status"] in ("queued", "running"):
        post(pid, "producer", "The render is still running; I'll take changes once it finishes.")
        return
    reshoot = change.get("reshoot") or []
    if reshoot:
        _reshoot_storyboard(pid, reshoot)
    if change.get("sound") or change.get("edit"):
        _restyle(pid, change, rendered=True)
    if reshoot:
        _rerender_lines(pid, [r["line"] for r in reshoot if isinstance(r.get("line"), int)])
    elif change.get("sound") or change.get("edit"):
        _jobs.enqueue(prod["job_id"], task="export")
        post(pid, "editor", "Re-exporting with the changes (about a minute).", "progress", {"job_id": prod["job_id"]})
        _update(pid, stage="rendering", review_after=False)
        threading.Thread(target=_watch_render, args=(pid,), daemon=True).start()


def _reshoot_storyboard(pid: str, reshoot: list[dict]) -> None:
    prod = load(pid)
    todo = [r for r in reshoot if isinstance(r.get("line"), int) and 1 <= r["line"] <= len(prod["shots"])]
    if not todo:
        return
    bible = _j({k: prod["design"].get(k) for k in ("characters", "setting", "look")})
    listing = "\n".join(f"{r['line']}. line: {prod['script']['lines'][r['line'] - 1].get('narration') or ''}\n"
                        f"   current shot: {prod['shots'][r['line'] - 1]['visual']}\n   CHANGE: {r.get('instruction', '')}" for r in todo)
    out = _call(pid, "dop", f"DESIGN BIBLE\n{bible}\n\nRewrite only these shots to apply the requested change. "
                f"Keep the line numbers.\n{listing}", "is reworking the shots…")
    names = {c["name"] for c in prod["design"]["characters"]}
    for s in out.get("shots", []):
        if isinstance(s.get("line"), int) and 1 <= s["line"] <= len(prod["shots"]):
            prod["shots"][s["line"] - 1].update(visual=s["visual"], characters=[c for c in s.get("characters", []) if c in names])
    _update(pid, shots=prod["shots"])
    post(pid, "dop", "Reworked line(s) " + ", ".join(str(r["line"]) for r in todo) + ".")


def _restyle(pid: str, change: dict, rendered: bool) -> None:
    prod = load(pid)
    if change.get("sound"):
        voices = tts.all_voices()
        sound = _call(pid, "sound", f"CURRENT\n{_j(prod['sound'])}\n\nAVAILABLE VOICES\n{_j(voices)}\n\nCHANGE REQUESTED\n{change['sound']}",
                      "is reworking the sound…")
        if sound.get("voice") not in voices:
            sound["voice"] = prod["sound"].get("voice", "")
        _update(pid, sound=sound)
        post(pid, "sound", f"Narrator: {voices.get(sound['voice'], 'none')}; music: "
                           + (sound.get("music_prompt", "") if sound.get("music") == "generate" else "none") + ".")
    if change.get("edit"):
        editor = _call(pid, "editor", f"CURRENT\n{_j(prod['editor'])}\n\nCHANGE REQUESTED\n{change['edit']}", "is recutting…")
        _update(pid, editor=editor)
        post(pid, "editor", f"New cut: {editor.get('chapter_transition')}/{editor.get('shot_transition')}, "
                            f"filter {editor.get('filter')}, subtitles {'on' if editor.get('subtitles') else 'off'}.")
    if rendered:
        _sync_rendered(pid)


def _sync_rendered(pid: str) -> None:
    """Push the crew's sound/editor/shot decisions into the job's spec and edit.json."""
    from .. import editor as ed

    prod = load(pid)
    job = _jobs.load(prod["job_id"])
    spec = job["spec"]
    new = _spec(prod)
    for k in ("voice", "music", "editor", "subtitles", "title_card", "title"):
        spec[k] = new[k]
    for ln, row in zip(spec["lines"], new["lines"]):
        ln.update(visual=row["visual"], characters=row["characters"])
    _jobs.update(prod["job_id"], spec=spec)
    job_dir = _jobs.dir(prod["job_id"])
    if ed.edit_path(job_dir).exists():
        e = json.loads(ed.edit_path(job_dir).read_text())
        e["audio"]["voice"] = spec["voice"]
        e["audio"]["music_volume"] = spec["music"]["volume"]
        e["subtitles_on"] = spec["subtitles"]
        e["look"]["filter"] = spec["editor"].get("filter", "none")
        ed.edit_path(job_dir).write_text(json.dumps(e, indent=2))
    if spec["music"]["mode"] == "generate":
        (job_dir / "music.wav").unlink(missing_ok=True)


def _rerender_lines(pid: str, lines: list[int]) -> None:
    prod = load(pid)
    _sync_rendered(pid)
    job = _jobs.load(prod["job_id"])
    spec = job["spec"]
    d = _jobs.dir(prod["job_id"])
    hist = d / "history"
    hist.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    for k, sc in enumerate(spec["scenes"]):
        if sc["line"] + 1 in lines:
            clip = d / f"scene_{k + 1:02d}.mp4"
            if clip.exists():
                shutil.move(clip, hist / f"scene_{k + 1:02d}_{stamp}.mp4")
            shutil.rmtree(d / f"tail_{k + 1:02d}", ignore_errors=True)
            if sc.get("keyframe"):
                (d / sc["keyframe"]).unlink(missing_ok=True)
                sc["kf_seed"] = random.randint(1, 2**31 - 1)
            sc["prompt"] = spec["lines"][sc["line"]]["visual"]
    _jobs.update(prod["job_id"], spec=spec)
    _jobs.enqueue(prod["job_id"], task="redo")
    _update(pid, stage="rendering", review_after=True)
    post(pid, "producer", f"Reshooting line(s) {', '.join(map(str, lines))}; the rest of the film is kept.",
         "progress", {"job_id": prod["job_id"]})
    threading.Thread(target=_watch_render, args=(pid,), daemon=True).start()
