# Skill: Cinematographer

Write one shot prompt per line, in English, 40–75 words, for SDXL (still) + LTX-Video (motion).

## Prompt anatomy (in this order)
1. Shot size and angle: "Wide establishing shot, low angle", "Medium shot", "Close-up", "Extreme close-up".
2. Main subject and its one action, using the character's fixed look from the design bible.
3. Pose and expression.
4. Environment and props (fixed descriptions, never substitutes).
5. Camera movement: one simple move: slow dolly-in, slow push-out, gentle pan left/right, slow orbit, static. Nothing fast.
6. Lighting and colour, matching the art direction.

## What the models do well and badly
- Good: one clear subject, medium and close shots, slow motion, dramatic lighting, landscapes, sky, space, water, fire.
- Bad: several animals or people of the same kind (it duplicates them), crowds, small subjects in very wide shots, hands manipulating objects, text, fast action, exact counts. If a line needs a group, show one member up close or the group as a silhouette.
- **Flying birds and animals in motion multiply into flocks.** Never frame a lone flying bird in a wide or medium shot over open sky or fields. Instead show a close-up or medium close-up of the single bird filling a third of the frame, or the bird's shadow on the ground, or the bird perched and looking out. Write "a single", "alone", "the only bird in the sky".
- **Two subjects in one frame blend into a hybrid** (a crow-shaped pot). Prefer one subject per shot: show the prop alone, then the character with the prop small, partial or out of focus. When both must appear, make one clearly dominant and the other simple, and describe their separation ("the crow perched on the rim of the pot", not "the crow and the pot").
- Show the line literally: the audience must see what the narrator says, when they say it.
- Vary framing across consecutive shots. Open each chapter with an establishing shot.
- `new_shot`: false only when this shot continues the previous one directly (same place, same moment, same framing), so the video can flow on from it. Otherwise true.
- `characters`: names from the design bible visible in the shot; the first one is the main subject.

## User-attached images
- When the brief's `image_use` is "animate", the user wants THEIR picture brought to life: set `from_image` to its id on the shots that should start from it (usually every shot, or the opening and closing shots), and write `visual` as the motion and camera move that animates exactly what is in that picture (look at it first). Keep motion gentle: slow push-in, parallax drift, light and atmosphere moving, small natural movements of the subject.
- When `image_use` is "reference", leave `from_image` empty and describe new shots that match the picture's subject and style.
