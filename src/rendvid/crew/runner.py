"""Run one crew member: a headless ``claude --print`` call with its own persona,
craft skill, memory and a JSON schema for its answer (the cowork pattern).

Memory lives in outputs/.crew/ and is safe to read and edit by hand:
  team.md               facts every member should know
  members/<role>.md     each member's own craft notes
  history.md            one line per finished production
  skills/<role>.md      how each role works (copied from the shipped templates
                        once; your edits are kept across updates)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .. import config

CREW_DIR = config.OUTPUTS / ".crew"
SHIPPED_SKILLS = Path(__file__).with_name("skills")
MAX_MEMORY = 6000

MEMORIES_SCHEMA = {
    "type": "array", "maxItems": 4,
    "items": {"type": "object", "required": ["scope", "note"], "properties": {
        "scope": {"type": "string", "enum": ["team", "personal"]}, "note": {"type": "string"}}},
}


def init_memory(roles: dict[str, dict]) -> None:
    (CREW_DIR / "members").mkdir(parents=True, exist_ok=True)
    (CREW_DIR / "skills").mkdir(parents=True, exist_ok=True)
    # A shipped template replaces the workspace copy unless it was edited by hand.
    shipped_file = CREW_DIR / "skills" / ".shipped.json"
    shipped = json.loads(shipped_file.read_text()) if shipped_file.exists() else {}
    for f in SHIPPED_SKILLS.glob("*.md"):
        template = f.read_text()
        target = CREW_DIR / "skills" / f.name
        current = target.read_text() if target.exists() else None
        if current is None or current == shipped.get(f.name):
            target.write_text(template)
            shipped[f.name] = template
    shipped_file.write_text(json.dumps(shipped, indent=2))
    team = CREW_DIR / "team.md"
    if not team.exists():
        team.write_text("# Crew memory\n\nShared notes for the whole film crew.\n\n")
    for key, m in roles.items():
        p = CREW_DIR / "members" / f"{key}.md"
        if not p.exists():
            p.write_text(f"# {m['role']}'s notes\n\n")
    hist = CREW_DIR / "history.md"
    if not hist.exists():
        hist.write_text("# Production history\n\n")


def _tail(p: Path, limit: int = MAX_MEMORY) -> str:
    if not p.exists():
        return ""
    t = p.read_text()
    return t if len(t) <= limit else "…\n" + t[-limit:]


def skill(name: str) -> str:
    return _tail(CREW_DIR / "skills" / f"{name}.md", 20000)


def context(key: str) -> str:
    return "\n\n---\n\n".join([
        f"Crew memory:\n{_tail(CREW_DIR / 'team.md')}",
        f"Your notes:\n{_tail(CREW_DIR / 'members' / f'{key}.md')}",
        "Recent productions:\n" + "\n".join(_tail(CREW_DIR / "history.md").splitlines()[-12:]),
        "Memory: in `memories`, return only durable lessons worth knowing on future films "
        "(the user's preferences, what the models do well or badly, conventions). Scope \"team\" "
        "for things everyone needs, \"personal\" for your own craft. Skip anything already noted. "
        "An empty list is fine.",
    ])


def save_memories(key: str, role: str, production: str, memories: list | None) -> None:
    date = time.strftime("%Y-%m-%d")
    for m in memories or []:
        note = str(m.get("note", "")).strip().replace("\n", " ")
        if not note:
            continue
        target = CREW_DIR / ("team.md" if m.get("scope") == "team" else f"members/{key}.md")
        with target.open("a") as f:
            f.write(f"- {date} ({role}, {production}): {note}\n")


def add_history(line: str) -> None:
    with (CREW_DIR / "history.md").open("a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d')}: {line}\n")


class MemberError(RuntimeError):
    pass


def run(
    key: str, member: dict, prompt: str, schema: dict, *,
    read_dir: Path | None = None, timeout: int = 900,
) -> dict:
    """One member turn. Returns the structured output (memories saved separately).

    ``read_dir`` lets the member open files there (images for the QA critic)
    with the read-only Read tool; every other member runs with no tools."""
    full_schema = {**schema, "properties": {**schema["properties"], "memories": MEMORIES_SCHEMA}}
    system = f"{member['persona']}\n\n{skill(member['skill'])}\n\n---\n\n{context(key)}"
    args = [
        "claude", "--print", prompt,
        "--system-prompt", system,
        "--model", os.environ.get("RENDVID_CREW_MODEL", member.get("model", "opus")),
        "--output-format", "json",
        "--json-schema", json.dumps(full_schema),
        "--strict-mcp-config", "--no-session-persistence",
    ]
    if read_dir:
        args += ["--tools", "Read", "--allowedTools", "Read", "--add-dir", str(read_dir)]
    else:
        args += ["--tools", ""]
    p = subprocess.run(args, cwd=str(read_dir or tempfile.gettempdir()), capture_output=True, text=True, timeout=timeout)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise MemberError(f"{member['role']}: Claude CLI failed: {(p.stderr or p.stdout)[-400:]}")
    if out.get("is_error") or not out.get("structured_output"):
        raise MemberError(f"{member['role']}: {str(out.get('result') or 'no structured answer')[:400]}")
    data = out["structured_output"]
    data["_cost"] = out.get("total_cost_usd")
    data["_seconds"] = round((out.get("duration_ms") or 0) / 1000)
    return data
