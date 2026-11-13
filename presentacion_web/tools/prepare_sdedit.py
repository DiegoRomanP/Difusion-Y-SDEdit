"""Genera las tiras de SDEdit (original -> ruido -> imagen nueva) para la diapositiva 8.

    ../.venv/bin/python tools/prepare_sdedit.py <best.safetensors> <foto> [--strengths 0.3 0.6 0.9]

Escribe public/evidencia/sdedit/sNN.png (8 cuadros de 64x64 por intensidad). Cambia <foto> por
una foto real propia y vuelve a ejecutarlo para actualizar la diapositiva.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pc3.data import canonical_image  # noqa: E402
from pc3.inference import sdedit, to_tensor  # noqa: E402
from pc3.model import load_weights  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("weights", type=Path)
parser.add_argument("photo", type=Path)
parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.6, 0.9])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = load_weights(args.weights)
model.to(device)
photo = canonical_image(args.photo)
out = Path(__file__).resolve().parents[1] / "public/evidencia/sdedit"
out.mkdir(parents=True, exist_ok=True)
for strength in args.strengths:
    _, frames = sdedit(model, to_tensor(photo).unsqueeze(0), strength=strength, device=device, capture=True)
    sequence = [photo, *frames]
    picks = np.linspace(0, len(sequence) - 1, 8).round().astype(int)
    strip = Image.new("RGB", (64 * 8, 64))
    for i, k in enumerate(picks):
        strip.paste(sequence[k], (i * 64, 0))
    strip.save(out / f"s{round(strength * 100):02d}.png")
    print("ok", strength)
