"""The film crew: persona, craft skill, model and output schema per member."""

from __future__ import annotations

from .. import config

CREW_INTRO = (
    "RendVid's film crew makes short narrated or visual films entirely on the user's Mac: "
    "Producer, Writer, Director, Production Designer, Cinematographer, Script Supervisor, "
    "Sound Designer, Editor and QA Critic. Films are built as a list of lines; each line is "
    "one narration sentence (or a silent beat) shown as one or more ~5 s AI-generated shots. "
    "Pipeline: SDXL draws a keyframe per shot (IP-Adapter copies the main character's reference "
    "portrait), LTX-Video 2B animates it (~5 s, simple motion), Indic Parler-TTS or Kokoro "
    "narrates, ffmpeg edits. Visual prompts are always in English; narration may be Hindi."
)

STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}


def arr(items: dict, **kw) -> dict:
    return {"type": "array", "items": items, **kw}


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props)}


BRIEF = obj({
    "title": STR,
    "format": {"type": "string", "enum": ["narrated", "visual", "music_video"],
               "description": "narrated: a voice tells the story. visual: no voice, pure images and music. "
                              "music_video: the user's attached song is the soundtrack and sets the length"},
    "song": {"type": "string", "description": "For music_video: the id of the attached audio file; else empty"},
    "language": {"type": "string", "enum": ["Hindi", "English"]},
    "source": {"type": "string", "enum": ["script", "idea"],
               "description": "script: the user gave the exact words to narrate (or exact shots). idea: the Writer writes it"},
    "script": {"type": "string", "description": "The user's script copied VERBATIM when source is script; empty otherwise"},
    "idea": {"type": "string", "description": "What the film is about, in the user's terms"},
    "minutes": {"type": "number", "description": "Target length; for a given script, estimate from its length"},
    "aspect": {"type": "string", "enum": list(config.ASPECTS)},
    "quality": {"type": "string", "enum": list(config.PRESETS)},
    "style": {"type": "string", "enum": list(config.STYLES)},
    "audience": STR,
    "notes": {"type": "string", "description": "Anything the crew must respect: tone, must-haves, things to avoid"},
    "images": {"type": "array", "items": {"type": "string"},
               "description": "ids of user-attached images this film uses (as first frames to animate, or as visual reference)"},
    "image_use": {"type": "string", "enum": ["none", "animate", "reference"],
                  "description": "animate: shots start from the user's image(s) exactly; reference: match their look but draw new frames"},
})

LINE = obj({
    "narration": {"type": "string", "description": "Spoken words (empty for a silent visual beat)"},
    "beat": {"type": "string", "description": "For a silent beat: what happens on screen (English); empty when narrated"},
    "chapter": INT,
    "seconds": {"type": "number", "description": "Only for silent beats: how long to hold, 3-10"},
}, ["narration", "beat", "chapter"])

SHOT = obj({
    "line": {"type": "integer", "description": "1-based line number"},
    "visual": STR,
    "characters": arr(STR),
    "new_shot": BOOL,
    "from_image": {"type": "string", "description": "id of a user-attached image to use as this shot's exact first frame "
                                                    "(it gets animated as-is), or empty to have the shot drawn"},
}, ["line", "visual", "characters", "new_shot", "from_image"])

CREW = {
    "producer": {
        "role": "Producer", "skill": "producer", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the Producer and the only member who talks to the user. "
        "Read the user's message and the production so far. Decide `kind`: \"question\" (answer it in "
        "`reply`), \"production\" (a new film: fill `brief`), or \"change\" (they want something in the "
        "current film changed: fill `change`). Always write `reply`: a short, warm, professional message "
        "to the user in their language register. For a new film, settle every brief field yourself from "
        "what they said and sensible defaults; never ask questions you can decide. When the user pasted "
        "a script or exact shot description, source is \"script\" and you copy it verbatim into `script`.",
        "schema": obj({
            "kind": {"type": "string", "enum": ["question", "production", "change"]},
            "reply": STR,
            "brief": BRIEF,
            "change": obj({
                "summary": STR,
                "reshoot": arr(obj({"line": INT, "instruction": STR}),
                               description="Lines whose pictures must change, with what to change"),
                "script": {"type": "string", "description": "Instruction for the Writer if narration words change; else empty"},
                "sound": {"type": "string", "description": "Instruction for the Sound Designer (voice, music); else empty"},
                "edit": {"type": "string", "description": "Instruction for the Editor (titles, transitions, look, pacing, subtitles); else empty"},
            }),
        }, ["kind", "reply"]),
    },
    "writer": {
        "role": "Writer", "skill": "writer", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the Writer. Write the narration of the film from the "
        "Producer's brief: a complete story with a hook, rising tension, a climax and a satisfying ending, "
        "in natural spoken language for the narrator. Split it into lines of one or two sentences "
        "(8-22 words) grouped into chapters.",
        "schema": obj({"title": STR, "chapters": arr(STR), "lines": arr(LINE)}),
    },
    "director": {
        "role": "Director", "skill": "director", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the Director. You own the creative vision. From the brief and "
        "script, define the tone, the emotional arc, what each chapter must make the audience feel, and "
        "the visual motifs that tie the film together. Your notes guide the Designer, Cinematographer, "
        "Sound Designer and Editor.",
        "schema": obj({
            "vision": STR, "tone": STR,
            "chapters": arr(obj({"chapter": INT, "feeling": STR, "beats": STR})),
            "motifs": arr(STR),
        }),
    },
    "designer": {
        "role": "Production Designer", "skill": "production-designer", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the Production Designer. You fix the film's visual identity "
        "so every shot looks like the same world: the exact look of every recurring character and key "
        "prop (only those the script actually contains), the setting and the art direction.",
        "schema": obj({
            "characters": arr(obj({"name": STR, "look": STR,
                                   "library_id": {"type": "string", "description": "id of the user's library asset this is, or empty for a new design"}})),
            "setting": STR, "look": STR,
            "style": {"type": "string", "enum": list(config.STYLES)},
            "style_library_id": {"type": "string", "description": "id of a [style] asset from the user's library to apply as the house style, or empty"},
            "location_library_id": {"type": "string", "description": "id of a [location] asset from the user's library where the film is set, or empty"},
        }),
    },
    "dop": {
        "role": "Cinematographer", "skill": "cinematographer", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the Cinematographer (DoP). For each line, design the shot "
        "the audience sees while it plays, written as a prompt the image and video models execute well.",
        "schema": obj({"shots": arr(SHOT)}),
    },
    "supervisor": {
        "role": "Script Supervisor", "skill": "script-supervisor", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the Script Supervisor. You check the storyboard against the "
        "script and the design bible before anything is rendered, and fix what is wrong yourself.",
        "schema": obj({
            "verdict": {"type": "string", "enum": ["pass", "fixed"]},
            "issues": arr(obj({"line": INT, "problem": STR, "visual": {"type": "string", "description": "The corrected full shot prompt"},
                               "characters": arr(STR)})),
            "note": STR,
        }),
    },
    "sound": {
        "role": "Sound Designer", "skill": "sound-designer", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the Sound Designer. Choose the narrator and the music bed.",
        "schema": obj({
            "voice": {"type": "string", "description": "One voice id from the list provided, or empty for no narration"},
            "music": {"type": "string", "enum": ["none", "generate"]},
            "music_prompt": STR,
            "music_volume": {"type": "number", "description": "0.08-0.4 under narration; 0.6-1.0 for a visual film"},
            "note": STR,
        }),
    },
    "editor": {
        "role": "Editor", "skill": "editor", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the Editor. You shape the cut: titles, transitions, look and pacing.",
        "schema": obj({
            "title_card": BOOL, "title_text": STR, "end_text": STR,
            "chapter_transition": {"type": "string", "enum": ["cut", "fade", "dissolve", "fadeblack", "wipeleft", "slideleft", "circleopen"]},
            "shot_transition": {"type": "string", "enum": ["cut", "fade", "dissolve", "fadeblack"]},
            "filter": {"type": "string", "enum": ["none", "warm", "cool", "bw", "vintage", "vivid", "film", "dream"]},
            "subtitles": BOOL,
            "note": STR,
        }),
    },
    "critic": {
        "role": "QA Critic", "skill": "qa-critic", "model": "opus",
        "persona": CREW_INTRO + "\n\nYou are the QA Critic. You open the actual images with the Read tool "
        "and judge each one against its shot description and the design bible, like a demanding "
        "festival juror. Report only what you actually see.",
        "schema": obj({
            "results": arr(obj({
                "shot": {"type": "string", "description": "The image file name"},
                "verdict": {"type": "string", "enum": ["pass", "redo"]},
                "problem": STR,
                "fix": {"type": "string", "description": "When redo: the corrected full shot prompt that avoids the problem; else empty"},
            })),
            "summary": STR,
        }),
    },
}
