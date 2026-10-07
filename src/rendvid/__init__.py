"""RendVid: local text-to-video on Apple Silicon."""

import argparse
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
    args = ap.parse_args()

    if args.cmd == "bench":
        _bench(args)
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
