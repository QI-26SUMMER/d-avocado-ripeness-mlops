"""Inference on real user photos (deployment scenario).

Usage:
  python -m src.predict --image path/to/photo.jpg
  python -m src.predict --dir  path/to/folder
  python -m src.predict --image photo.jpg --checkpoint P3_t10_resnet18_paper_aug_oversample
  python -m src.predict --image photo.jpg --storage-group T20   # also computes shelf-life

Default model = P1 general ResNet-18 (for generalization when storage condition is
unknown). Input is a single RGB image only (§2.3).

Warning: domain gap (CLAUDE.md §3): training = light box (white background, controlled
   lighting), real-world = kitchen lighting → prediction confidence may be lower.
   Vulnerable to background/lighting differences, so always check the probability
   distribution alongside the prediction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from .data import LABELS
    from .models import build_model
    from .shelf_life import ALPHA_5STAGE, estimate_days_left
    from .transforms import evaluation_transform
    from .utils import OUTPUT_DIR, get_device
except ImportError:
    from data import LABELS
    from models import build_model
    from shelf_life import ALPHA_5STAGE, estimate_days_left
    from transforms import evaluation_transform
    from utils import OUTPUT_DIR, get_device

DEFAULT_ID = "P1_general_resnet18_paper_aug_oversample"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Service-facing hints (§1 label meaning). Recommended range 3-4.
STAGE_HINT = {1: "Unripe (firm)", 2: "Breaking", 3: "Ready to eat", 4: "Peak (end of shelf life)", 5: "Overripe (too late)"}


def resolve_checkpoint(arg: str | None) -> Path:
    """Checkpoint argument: a path or an experiment id. Defaults to P1 general."""
    if arg is None:
        return OUTPUT_DIR / "checkpoints" / DEFAULT_ID / "best.pt"
    p = Path(arg)
    if p.exists():
        return p
    cand = OUTPUT_DIR / "checkpoints" / arg / "best.pt"
    if cand.exists():
        return cand
    raise SystemExit(f"Checkpoint not found: {arg}")


def try_register_heif() -> None:
    """iPhone HEIC support (if installed)."""
    try:
        import pillow_heif  # noqa

        pillow_heif.register_heif_opener()
        EXTS.update({".heic", ".heif"})
    except Exception:
        pass


def collect_images(image, directory) -> list[Path]:
    if image:
        return [Path(image)]
    files = [p for p in sorted(Path(directory).iterdir()) if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f"No supported images found in {directory} (supported: {sorted(EXTS)})")
    return files


def bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def predict(model, tf, path: Path, device: str):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].cpu().numpy()
    return int(probs.argmax()) + 1, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--dir")
    ap.add_argument("--checkpoint", default=None, help="Path or experiment id (defaults to P1 general resnet18)")
    ap.add_argument("--storage-group", choices=["T10", "T20", "Tam", "Tamb"], default=None,
                    help="If given, also estimates shelf-life (days remaining). Not fed into the model (§2.3)")
    args = ap.parse_args()
    if not args.image and not args.dir:
        raise SystemExit("Specify either --image or --dir.")

    try_register_heif()
    device = get_device()
    ckpt_path = resolve_checkpoint(args.checkpoint)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg["model"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tf = evaluation_transform(cfg.get("img_size", 224))
    print(f"Model: {cfg['experiment_id']} ({cfg['model']}, {cfg['dataset']})  device={device}")
    print("Warning: trained on light-box photos — kitchen-lighting photos may yield unstable predictions (check the probability distribution too).\n")

    for path in collect_images(args.image, args.dir):
        try:
            stage, probs = predict(model, tf, path, device)
        except Exception as e:
            print(f"[failed to open] {path.name}: {e}")
            continue
        conf = probs[stage - 1]
        print(f"■ {path.name}")
        print(f"  → Predicted stage: {stage} ({LABELS[stage]}, {STAGE_HINT[stage]})  confidence {conf:.0%}")
        for k in range(1, 6):
            mark = "◀" if k == stage else " "
            print(f"    {k} {STAGE_HINT[k]:<12} {bar(probs[k-1])} {probs[k-1]:5.1%} {mark}")
        if args.storage_group:
            days = estimate_days_left(stage, args.storage_group)
            print(f"  → shelf-life({args.storage_group}, α={ALPHA_5STAGE[args.storage_group]}): "
                  f"~{days:.1f} days remaining (based on predicted stage)")
        print()


if __name__ == "__main__":
    main()
