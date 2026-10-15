"""RendVid: local text-to-video on Apple Silicon."""

import argparse
import json
import os

# Let PyTorch fall back to CPU for the few ops MPS lacks, and don't cap MPS
# memory below what the 16 GB machine can actually give.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def _load_dotenv() -> None:
    """KEY=VALUE lines from the project's .env (e.g. HF_TOKEN for gated models).
    Real environment variables win. Child processes (the Parler worker) inherit them."""
    from pathlib import Path

    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_dotenv()


def _ensure_text_shaping() -> None:
    """Pillow shapes Devanagari (Hindi matras, conjuncts) via raqm, which
    dlopens Homebrew's fribidi by leaf name. dyld only reads its search path at
    launch, so re-exec once with /opt/homebrew/lib on the fallback path."""
    import sys

    brew = "/opt/homebrew/lib"
    paths = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if os.path.exists(f"{brew}/libfribidi.dylib") and brew not in paths.split(":"):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(p for p in (paths, brew, "/usr/local/lib", "/usr/lib") if p)
        os.execv(sys.executable, [sys.executable, "-c", "import rendvid; rendvid.main()", *sys.argv[1:]])


def main() -> None:
    _ensure_text_shaping()
    ap = argparse.ArgumentParser(prog="rendvid")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web UI (default)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    b = sub.add_parser("bench", help="render one scene and report time")
    b.add_argument("--preset", default="balanced", choices=["fast", "balanced", "max"])
    b.add_argument("--aspect", default="16:9")
    b.add_argument("--prompt", default="A golden retriever runs along a sunny beach, waves crashing behind it, "
                   "its ears flapping in the wind. The camera tracks alongside at dog height. Warm afternoon light.")
    t = sub.add_parser("train", help="train a character or style LoRA locally")
    t.add_argument("--kind", required=True, choices=["character", "style"])
    t.add_argument("--name", required=True)
    t.add_argument("--images", help="folder of training images (optional <image>.txt captions); "
                   "omit to generate a set from --describe")
    t.add_argument("--describe", default="", help="description to generate a training set from")
    t.add_argument("--style", default="", help="style words for generated training images")
    t.add_argument("--class-word", default="", help="class word for character captions, e.g. crow")
    t.add_argument("--count", type=int, default=16, help="images to generate when --images is omitted")
    t.add_argument("--steps", type=int)
    t.add_argument("--rank", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--resolution", type=int)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "bench":
        _bench(args)
        return
    if args.cmd == "train":
        _train(args)
        return
    import uvicorn

    uvicorn.run("rendvid.server:app", host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8000))


def _bench(args) -> None:
    import time

    import torch

    from . import config, stitch
    from .model import VideoEngine, encode_prompts

    size = config.ASPECTS[args.aspect][args.preset]
    t = time.time()
    emb = encode_prompts([args.prompt], config.DEFAULT_NEGATIVE)[0]
    print(f"encode: {time.time() - t:.1f}s")
    t = time.time()
    eng = VideoEngine(multiscale=config.PRESETS[args.preset]["multiscale"])
    print(f"load: {time.time() - t:.1f}s")
    t = time.time()
    frames = eng.render(emb, size, seed=42, on_step=lambda d, n: print(f"  step {d}/{n} {time.time() - t:.0f}s", flush=True))
    print(f"render {size} x {len(frames)} frames: {time.time() - t:.1f}s")
    print(f"peak mps memory: {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")
    config.OUTPUTS.mkdir(exist_ok=True)
    out = config.OUTPUTS / f"bench_{args.preset}.mp4"
    stitch.write_clip(frames, out)
    print(f"wrote {out}")


def _train(args) -> None:
    from pathlib import Path

    from . import config, keyframes, training

    trigger = training.default_trigger(args.kind, args.name)
    slug = trigger.removeprefix("rvd_style_").removeprefix("rvd_")
    if args.images:
        images = sorted(p for p in Path(args.images).iterdir()
                        if p.suffix.lower() in training.IMAGE_EXTS and not p.name.startswith("_"))
    else:
        if not args.describe:
            raise SystemExit("give --images <dir> or --describe \"<what it looks like>\"")
        data = config.OUTPUTS / ".loras" / "datasets" / args.kind / slug
        images = sorted(p for p in data.glob("[0-9]*.png")) if data.exists() else []
        if len(images) < args.count:
            print(f"generating {args.count} training images in {data} ...", flush=True)
            images = training.make_training_set(args.describe, args.style, data, count=args.count, seed=args.seed,
                                                kind=args.kind,
                                                on_progress=lambda d, n: print(f"  image {d}/{n}", flush=True))
    out = keyframes.LORA_DIR / args.kind / f"{slug}.safetensors"
    print(f"training {args.kind} LoRA {trigger!r} on {len(images)} images -> {out}", flush=True)
    info = training.train_lora(args.kind, args.name, images, out, trigger=trigger,
                               class_word=args.class_word if args.kind != "style" else "",
                               steps=args.steps, rank=args.rank, lr=args.lr, resolution=args.resolution,
                               seed=args.seed, log=lambda m: print(m, flush=True))
    print(json.dumps(info, indent=2))
