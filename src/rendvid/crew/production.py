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

from .. import config, library, tts
from ..studio import split_script
from . import runner
from .members import CREW

PROD_DIR = runner.CREW_DIR / "productions"
_locks: dict[str, threading.Lock] = {}
_busy: set[str] = set()
_stopping: set[str] = set()
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
            out.append({k: p.get(k) for k in ("id", "title", "stage", "updated", "created", "job_id", "busy", "activity", "active", "cost",
                                              "member_stats")})
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

UPLOADS = runner.CREW_DIR / "uploads"
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def save_attachment(name: str, data: bytes) -> dict:
    """Store a file sent in chat; returns its public info."""
    ext = Path(name).suffix.lower()
    if ext not in AUDIO_EXT | IMAGE_EXT:
        raise ValueError(f"Unsupported file type {ext or '(none)'}; send audio or images")
    if len(data) > 60 * 1024 * 1024:
        raise ValueError("File too large (max 60 MB)")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    aid = uuid.uuid4().hex[:10]
    path = UPLOADS / f"{aid}{ext}"
    path.write_bytes(data)
    info = {"id": aid, "name": Path(name).name[:80], "kind": "audio" if ext in AUDIO_EXT else "image",
            "url": f"/outputs/.crew/uploads/{aid}{ext}", "size": len(data), "file": path.name}
    if info["kind"] == "audio":
        from .. import stitch

        info["duration"] = round(stitch.probe_duration(path), 2)
        if info["duration"] <= 0:
            path.unlink(missing_ok=True)
            raise ValueError("Could not read that audio file")
    (UPLOADS / f"{aid}.json").write_text(json.dumps(info))
    return info


def attachment(aid: str) -> dict | None:
    p = UPLOADS / f"{Path(aid).name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def attachment_path(aid: str) -> Path | None:
    a = attachment(aid)
    return UPLOADS / a["file"] if a else None


def create(text: str, attachments: list[str] | None = None) -> dict:
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    title = text.strip().split("\n")[0][:60] or next((a["name"] for a in map(attachment, attachments or []) if a), "New film")
    prod = {"id": pid, "title": title, "created": time.time(),
            "stage": "briefing", "busy": False, "activity": "", "messages": [], "cost": 0.0}
    _save(prod)
    user_message(pid, text, attachments)
    return load(pid)


def user_message(pid: str, text: str, attachments: list[str] | None = None) -> None:
    files = [a for a in map(attachment, attachments or []) if a]
    post(pid, "user", text, data={"attachments": files} if files else None)
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


def stop(pid: str) -> str:
    """Interrupt whatever the crew is doing in this production."""
    prod = load(pid)
    stopped = []
    if pid in _busy:
        _stopping.add(pid)
        runner.stop(pid)
        stopped.append("the crew")
        # A half-made storyboard can't be approved: go back to the approved brief.
        if prod["stage"] == "planning":
            _update(pid, stage="awaiting_brief" if prod.get("brief") else "briefing")
    if prod.get("job_id"):
        job = _jobs.load(prod["job_id"]) or {}
        if job.get("status") in ("queued", "running"):
            _jobs.cancel(prod["job_id"])
            runner.stop(pid)  # a QA Critic check inside the render, if one is running
            stopped.append("the render (finished shots are kept)")
    if not stopped:
        raise ValueError("Nothing is running")
    post(pid, "system", f"Stopped {' and '.join(stopped)}. Reply with what to change, or resume.", "stopped",
         {"render": len(stopped) > 1 or stopped[0].startswith("the render")})
    return ", ".join(stopped)


def resume(pid: str) -> None:
    """Continue a stopped/failed render from the last finished shot."""
    prod = load(pid)
    job = _jobs.load(prod["job_id"]) if prod.get("job_id") else None
    if not job or job["status"] in ("queued", "running"):
        raise ValueError("No stopped render to resume")
    _jobs.enqueue(prod["job_id"])
    _update(pid, stage="rendering")
    post(pid, "user", "▶ Resume the render")
    post(pid, "producer", "Resuming from the last finished shot.", "progress", {"job_id": prod["job_id"]})
    threading.Thread(target=_watch_render, args=(pid,), daemon=True).start()


def _spawn(pid: str, fn, *args) -> None:
    if pid in _busy:
        raise ValueError("The crew is still working on the last step; wait a moment")

    def work():
        _busy.add(pid)
        _update(pid, busy=True)
        try:
            fn(pid, *args)
        except Exception as e:
            if pid in _stopping or isinstance(e, runner.Stopped):
                pass  # stop() already told the user
            else:
                traceback.print_exc()
                post(pid, "system", f"Something went wrong: {e}", "error")
        finally:
            _stopping.discard(pid)
            _busy.discard(pid)
            _active.pop(pid, None)
            _update(pid, busy=False, activity="", active=[])

    threading.Thread(target=work, daemon=True).start()


# --- member calls --------------------------------------------------------------------

_active: dict[str, dict[str, str]] = {}  # pid -> {member key: activity text}


def _set_active(pid: str) -> None:
    act = _active.get(pid, {})
    if len(act) == 1:
        text = next(iter(act.values()))
    elif act:
        roles = [CREW[k]["role"] for k in act]
        text = f"{', '.join(roles[:-1])} and {roles[-1]} are working…"
    else:
        text = ""
    _update(pid, activity=text, active=list(act))


def user_images(prod: dict) -> list[dict]:
    """Images the user attached anywhere in this chat."""
    return [a for m in prod["messages"] if m["from"] == "user"
            for a in (m.get("data") or {}).get("attachments", []) if a.get("kind") == "image"]


def _call(pid: str, key: str, prompt: str, activity: str, read_dir: Path | None = None) -> dict:
    m = CREW[key]
    if read_dir is None and (imgs := user_images(load(pid))):
        # Let the member actually look at the user's pictures (read-only Read tool).
        read_dir = UPLOADS
        prompt = ("THE USER ATTACHED THESE IMAGES. Open every one with the Read tool and look at it before you answer; "
                  "describe and use what you actually see:\n"
                  + "\n".join(f"- id {a['id']} ({a['name']}): {UPLOADS / a['file']}" for a in imgs) + "\n\n" + prompt)
    _active.setdefault(pid, {})[key] = f"{m['role']} {activity}"
    _set_active(pid)
    try:
        if pid in _stopping:
            raise runner.Stopped("stopped")
        out = runner.run(key, m, prompt, m["schema"], read_dir=read_dir, owner=pid)
    finally:
        _active.get(pid, {}).pop(key, None)
        _set_active(pid)
    runner.save_memories(key, m["role"], pid, out.pop("memories", None))
    with _lock(pid):
        prod = load(pid)
        prod["cost"] = round(prod.get("cost", 0) + (out.get("_cost") or 0), 4)
        stats = prod.setdefault("member_stats", {}).setdefault(key, {"cost": 0.0, "runs": 0})
        stats["cost"] = round(stats["cost"] + (out.get("_cost") or 0), 4)
        stats["runs"] += 1
        _save(prod)
    return out


def _parallel(*calls):
    """Run independent member calls at the same time; results in order."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(calls)) as ex:
        futures = [ex.submit(fn, *args) for fn, *args in calls]
        return [f.result() for f in futures]


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
    files = [a for m in prod["messages"] if m["from"] == "user" for a in (m.get("data") or {}).get("attachments", [])]
    if files:
        state.append("FILES THE USER ATTACHED IN THIS CHAT:\n" + "\n".join(
            f"- id {a['id']}: {a['name']} ({a['kind']}" + (f", {int(a['duration'] // 60)}:{int(a['duration'] % 60):02d}" if a.get("duration") else "")
            + ")" for a in files))
    state.append(f"USER'S LIBRARY (characters & resources they built; '@Name' in a message refers to these):\n{library.catalog()}")
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

def song_beats(duration: float, segments: list, target: float = 4.6, lo: float = 2.6, hi: float = 6.5) -> list[dict]:
    """Cut a song into shot-sized beats that start where lyric lines start."""
    from ..lyrics import in_window

    marks = sorted({round(t, 2) for s, e, _ in segments for t in (s, e) if 0.5 < t < duration - 0.5})
    beats, t, chapter, last_lyric = [], 0.0, 0, 0.0
    while duration - t > 0.3:
        options = [m for m in marks if lo <= m - t <= hi]
        end = min(options, key=lambda m: abs(m - t - target)) if options else min(t + target, duration)
        if duration - end < lo:  # don't leave a sliver at the end
            end = duration
        words = in_window(segments, t, end).strip()
        if words:
            if t - last_lyric > 4 and beats:  # a pause in the singing starts a new section
                chapter += 1
            last_lyric = end
        elif beats and beats[-1]["beat"] != "(instrumental)" and t - last_lyric > 3:
            chapter += 1
        beats.append({"narration": "", "beat": words or "(instrumental)", "seconds": round(end - t, 3),
                      "start": round(t, 2), "chapter": chapter})
        t = end
    return beats


def _plan(pid: str, notes: str = "") -> None:
    _update(pid, stage="planning")
    prod = load(pid)
    brief = prod["brief"]
    narrated = brief.get("format") == "narrated"
    music_video = brief.get("format") == "music_video"

    # 1. Script
    if music_video:
        song = attachment_path(brief.get("song", ""))
        if not song:
            raise ValueError("Attach the song (MP3) in the chat first")
        from .. import lyrics, stitch

        _active.setdefault(pid, {})["writer"] = "Writer is listening to the song and transcribing the lyrics…"
        _set_active(pid)
        try:
            duration = stitch.probe_duration(song)
            words = lyrics.transcribe(song) if brief.get("use_lyrics", True) else {"language": None, "segments": []}
        finally:
            _active.get(pid, {}).pop("writer", None)
            _set_active(pid)
        lines = song_beats(duration, words["segments"])
        script = {"title": brief.get("title", ""), "chapters": [], "lines": lines,
                  "song": {"path": str(song), "duration": duration, "lyrics": words["segments"], "language": words["language"]}}
        post(pid, "writer", f"Song is {int(duration // 60)}:{int(duration % 60):02d}. I found {len(words['segments'])} lyric lines "
                            f"({words['language'] or 'instrumental'}) and cut it into {len(lines)} beats that follow the singing.",
             "lyrics", {"lines": words["segments"], "language": words["language"], "duration": duration})
    elif narrated and brief.get("source") == "script" and brief.get("script", "").strip():
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
    if not script["lines"]:
        raise ValueError("The Writer returned an empty script, so there is nothing to shoot. "
                         "Reply with what should happen in the film and I'll try again.")
    _update(pid, script=script)
    script_text = "\n".join(
        f"{i}. [ch {ln.get('chapter', 0)}]" + (f" [{_mmss(ln['start'])}–{_mmss(ln['start'] + ln['seconds'])}] lyric: " if "start" in ln else " ")
        + (ln.get("narration") or ln.get("beat", "")) for i, ln in enumerate(script["lines"], 1))
    if music_video:
        script_text = ("MUSIC VIDEO. The song plays over the whole film; each numbered beat is one shot timed to the "
                       "lyric sung during it. No narration.\n" + script_text)

    # 2. Director
    director = _call(pid, "director", f"BRIEF\n{_j(brief)}\n\nSCRIPT\n{script_text}", "is shaping the vision…")
    director.pop("_cost", None)
    post(pid, "director", f"{director.get('vision', '')}\nTone: {director.get('tone', '')}")
    _update(pid, director=director)

    # 3. Production design, sound and the cut are independent of each other:
    #    run them at the same time.
    voices = tts.all_voices()
    design, sound, editor = _parallel(
        (_call, pid, "designer", f"BRIEF\n{_j(brief)}\n\nDIRECTOR\n{_j(director)}\n\nSCRIPT\n{script_text}\n\n"
         f"USER'S LIBRARY (reuse entries the user @mentions or that clearly fit; copy their description verbatim and set library_id)\n"
         f"{library.catalog()}", "is designing the look…"),
        (_call, pid, "sound", f"BRIEF\n{_j(brief)}\n\nDIRECTOR\n{_j(director)}\n\nAVAILABLE VOICES\n{_j(voices)}\n\n"
         f"MUSIC IN THE USER'S LIBRARY\n{library.catalog()}", "is choosing voice and music…"),
        (_call, pid, "editor", f"BRIEF\n{_j(brief)}\n\nDIRECTOR\n{_j(director)}", "is planning the cut…"),
    )
    known = {a["id"]: a for a in library.list_assets()}
    for c in design.get("characters", []):
        lib = known.get(c.get("library_id") or "")
        if lib:
            c["look"] = lib["description"] or c.get("look", "")
            library.mark_used(lib["id"], pid)
        else:
            c["library_id"] = ""
    design["characters"] = [c for c in design.get("characters", []) if c.get("name") and c.get("look")][:5]
    post(pid, "designer", "Design bible:\n" + "\n".join(
        f"• {c['name']}{' 📚 from your library' if c.get('library_id') else ''}: {c['look']}" for c in design["characters"])
         + f"\n• Setting: {design.get('setting', '')}\n• Art direction: {design.get('look', '')}")
    _update(pid, design=design)

    # 4. Cinematographer, in batches so long films stay sharp
    bible = _j({k: design.get(k) for k in ("characters", "setting", "look")})
    lines = script["lines"]

    def dop_batch(b0: int) -> list[dict]:
        part = lines[b0:b0 + 12]
        listing = "\n".join(
            f"{b0 + k + 1}. " + (f"[{_mmss(ln['start'])}, {ln['seconds']:.1f}s] lyric: {ln.get('beat')}" if "start" in ln
                                 else ln.get("narration") or "(silent) " + ln.get("beat", ""))
            for k, ln in enumerate(part))
        if music_video:
            listing = ("MUSIC VIDEO: the song plays throughout; the performer may be shown singing, but there is NO "
                       "lip-sync, so frame singing as side or three-quarter angles, medium or wide shots, eyes closed "
                       "in emotion, or singing away from camera, never a frontal mouth close-up.\n" + listing)
        prev = (f"\nTHE LINE BEFORE THIS BATCH (already shot by a colleague): "
                f"{lines[b0 - 1].get('narration') or lines[b0 - 1].get('beat', '')}\n") if b0 else ""
        out = _call(pid, "dop", f"BRIEF\n{_j(brief)}\n\nDIRECTOR\n{_j(director)}\n\nDESIGN BIBLE\n{bible}\n{prev}\n"
                    f"LINES (write exactly one shot per line, keep the line numbers)\n{listing}",
                    f"is designing shots {b0 + 1}–{b0 + len(part)}…")
        got = {s["line"]: s for s in out.get("shots", []) if isinstance(s.get("line"), int)}
        res = []
        for k, ln in enumerate(part):
            n = b0 + k + 1
            s = got.get(n) or {"line": n, "visual": ln.get("beat") or ln.get("narration", ""), "characters": [], "new_shot": True}
            s["line"] = n
            res.append(s)
        return res

    # Batches go out together (max 4 at once) so a long film is storyboarded in one pass.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        shots = [s for batch in ex.map(dop_batch, range(0, len(lines), 12)) for s in batch]
    shots[0]["new_shot"] = True
    names = {c["name"] for c in design["characters"]}
    image_ids = {a["id"] for a in user_images(load(pid))}
    for s in shots:
        s["characters"] = [c for c in s.get("characters", []) if c in names]
        s["from_image"] = s.get("from_image") if s.get("from_image") in image_ids else ""
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

    # Sound + cut were decided alongside the design; report them now.
    if music_video:  # the user's song is the soundtrack; beat cuts keep sync
        sound = {"voice": "", "music": "song", "music_prompt": "", "music_volume": 1.0,
                 "note": "Your song is the soundtrack, at full volume."}
        editor["shot_transition"] = "cut"
        if editor.get("chapter_transition") not in ("cut", "dissolve", "fade"):
            editor["chapter_transition"] = "dissolve"
        editor["end_text"] = ""
    if music_video:
        post(pid, "sound", "Soundtrack: your song at full volume, with every shot cut on the beat of the lyrics.")
    elif sound.get("voice") not in voices or not narrated:
        sound["voice"] = _default_voice(brief) if narrated else ""
    if not music_video:
        post(pid, "sound", f"Narrator: {voices.get(sound['voice'], 'none')}. Music: "
                           + (f"{sound.get('music_prompt')} (vol {sound.get('music_volume')})" if sound.get("music") == "generate" else "none")
                           + (f"\n{sound['note']}" if sound.get("note") else ""))
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


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


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
        n += max(1, math.ceil((secs - 0.05) / 5.04))
    return n


def _rows(prod: dict) -> list[dict]:
    lines = prod["script"]["lines"]
    return [{"line": i + 1, "chapter": ln.get("chapter", 0), "narration": ln.get("narration") or "",
             "beat": ln.get("beat", ""), "visual": s["visual"], "characters": s.get("characters", []),
             "new_shot": s.get("new_shot", True) or bool(s.get("from_image")), "from_image": s.get("from_image", "")}
            for i, (ln, s) in enumerate(zip(lines, prod["shots"]))]


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
        "style_asset": design.get("style_library_id") or "",
        "location_asset": design.get("location_library_id") or "",
        "lines": [{"narration": r["narration"], "visual": r["visual"], "characters": r["characters"],
                   "new_shot": r["new_shot"], "chapter": r["chapter"],
                   "from_image": str(attachment_path(r["from_image"]) or "") if r.get("from_image") else "",
                   "seconds": next((ln.get("seconds") for ln in [prod["script"]["lines"][r["line"] - 1]]), None) or 5}
                  for r in _rows(prod)],
        "subtitles": bool(editor.get("subtitles", True)), "title_card": bool(editor.get("title_card", True)),
        "music": {"mode": sound.get("music", "none"), "prompt": sound.get("music_prompt", ""),
                  "volume": float(sound.get("music_volume") or 0.18)},
        "editor": editor,
        **_song_spec(prod),
    }


def _song_spec(prod: dict) -> dict:
    song = (prod.get("script") or {}).get("song")
    if not song:
        return {}
    return {"song": True, "song_path": song["path"], "lyrics": song["lyrics"],
            "lyric_subtitles": bool(prod["editor"].get("subtitles", False)),
            "music": {"mode": "upload", "volume": 1.0}}


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
            if job["status"] != "cancelled":  # a user Stop already posted its own message
                post(pid, "system", f"Render {job['status']}: {job.get('message', '')}", "stopped", {"render": True})
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

    def batch_review(b0: int) -> list[dict]:
        batch = images[b0:b0 + 8]
        listing = "\n".join(f"- {p.name}: line {_line_of(p.name)}: “{spec['lines'][_line_of(p.name) - 1].get('narration') or ''}”\n"
                            f"  intended shot: {spec['lines'][_line_of(p.name) - 1]['visual']}" for p in batch)
        out = _call(pid, "critic", f"DESIGN BIBLE\n{looks}\n\nOpen each image file below (in {batch[0].parent}) "
                    f"with the Read tool and judge it.\n{listing}", activity, read_dir=batch[0].parent)
        by_name = {r.get("shot"): r for r in out.get("results", [])}
        res = []
        for p in batch:
            r = by_name.get(p.name) or {"verdict": "pass", "problem": "", "fix": ""}
            res.append({"shot": p.name, "verdict": r.get("verdict", "pass"), "problem": r.get("problem", ""), "fix": r.get("fix", "")})
        return res

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        return [r for batch in ex.map(batch_review, range(0, len(images), 8)) for r in batch]


# --- keyframe QA (called from inside the render) ----------------------------------------

def review_keyframes(pid: str, spec: dict, job_dir: Path, regenerate: Callable[[list[int]], None], rounds: int = 2) -> None:
    """Critic checks new keyframes; failing ones get the critic's fix and are redrawn."""
    checked: set[str] = set()
    for rnd in range(rounds):
        kfs = [p for p in sorted((job_dir / "keyframes").glob("line_*.png"))
               if p.name not in checked and not spec["lines"][_line_of(p.name) - 1].get("from_image")]
        if not kfs:
            return
        try:
            results = critic_review(pid, spec, kfs, f"is checking {len(kfs)} keyframes…")
        except runner.Stopped:
            from ..model import Cancelled

            raise Cancelled()
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
