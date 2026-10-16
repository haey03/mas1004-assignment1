"""Controlled background-swap experiment for Problem 7.

Use an image with transparency (PNG with alpha) so the foreground pixels stay
identical and only the background changes.

Example:
    uv run python src/background_experiment.py data/new_images/cartoon_animal/animal.png

Writes original_white.png, changed_blue.png, and predictions.json to
results/background_experiment/. It measures the chosen exported PyTorch model.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import prepare_image
from train import load
import torch

MODEL = ROOT / "results" / "frozen_model.pt"
OUT = ROOT / "results" / "background_experiment"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python src/background_experiment.py path/to/transparent_animal.png")
    source = Path(sys.argv[1])
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise SystemExit(f"Image not found: {source}")
    if not MODEL.exists():
        raise SystemExit(f"Model not found: {MODEL}")
    rgba = Image.open(source).convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    if alpha.min() == 255:
        raise SystemExit("This image has no transparent pixels. Use a PNG with a genuine alpha channel.")
    OUT.mkdir(parents=True, exist_ok=True)
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    blue = Image.new("RGBA", rgba.size, (70, 130, 220, 255))
    original = Image.alpha_composite(white, rgba).convert("RGB")
    changed = Image.alpha_composite(blue, rgba).convert("RGB")
    p1, p2 = OUT / "original_white.png", OUT / "changed_blue.png"
    original.save(p1)
    changed.save(p2)
    model = load(MODEL)
    class_names = sorted(d.name for d in (ROOT / "data" / "clean").iterdir() if d.is_dir())
    results = {}
    for name, im in [("white_background", original), ("blue_background", changed)]:
        x = torch.from_numpy(prepare_image(im)).unsqueeze(0)
        with torch.no_grad():
            logits = model(x).numpy()[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        order = np.argsort(probs)[::-1]
        results[name] = {
            "prediction": class_names[int(order[0])],
            "confidence": float(probs[order[0]]),
            "scores": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        }
    payload = {
        "source_image": str(source),
        "experiment": "same RGBA foreground composited onto white vs blue background",
        "note": "Inspect the two output images to ensure only the background changed.",
        "results": results,
        "confidence_difference": results["blue_background"]["confidence"] - results["white_background"]["confidence"],
    }
    (OUT / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Open these images and verify the foreground is unchanged: {p1} and {p2}")


if __name__ == "__main__":
    main()
