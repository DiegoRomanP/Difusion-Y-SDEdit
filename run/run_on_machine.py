# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "diffusers==0.42.1", "safetensors", "numpy", "pillow", "tqdm"]
# ///
"""Genera caras anime con unos pesos entrenados, en esta máquina (GPU si hay, si no CPU).

    uv run run_on_machine.py ruta/a/best.safetensors 16

Escribe en run/output/: images/000000.png … (una por semilla), grid.png (las 16 primeras)
y el tiempo total. Reutiliza src/pc3, así que las semillas dan las mismas caras que el repo.
`uv run` instala las dependencias de este script solo (no usa las extras cpu/cu128 del
proyecto); en Linux el torch de PyPI ya trae CUDA.
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from pc3.inference import sample_tensor, save_grid, to_pil  # noqa: E402
from pc3.model import load_weights  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("weights", type=Path, help="archivo best.safetensors")
    parser.add_argument("count", type=int, help="cuántas imágenes generar (entero > 0)")
    parser.add_argument("--steps", type=int, default=50, help="pasos DDIM (def.: 50, el del protocolo)")
    parser.add_argument("--seed", type=int, default=12345, help="semilla de la primera imagen")
    parser.add_argument("--batch-size", type=int, help="def.: 32 con GPU, 4 con CPU")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()

    if args.count < 1 or args.steps < 1:
        parser.error("count y --steps deben ser enteros positivos")
    if not args.weights.is_file():
        parser.error(f"no existe el archivo de pesos: {args.weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = args.batch_size or (32 if device.type == "cuda" else 4)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Dispositivo: {name} ({'fp16' if device.type == 'cuda' else 'fp32'}) · lote {batch} · {args.steps} pasos DDIM")

    model, meta = load_weights(args.weights)
    model.to(device)
    print(f"Pesos: paso {meta.get('step')}, {meta.get('weights')}, {sum(p.numel() for p in model.parameters()):,} parámetros")

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    sample_tensor(model, seeds=[args.seed], steps=1, device=device)  # calentamiento (arranque de CUDA)

    seconds = 0.0
    for start in tqdm(range(0, args.count, batch), desc="generando"):
        stop = min(args.count, start + batch)
        begin = time.perf_counter()
        images, _ = sample_tensor(
            model, seeds=list(range(args.seed + start, args.seed + stop)), steps=args.steps, device=device
        )
        seconds += time.perf_counter() - begin  # .cpu() al final del muestreo ya sincroniza
        for index, tensor in zip(range(start, stop), images, strict=True):
            to_pil(tensor).save(image_dir / f"{index:06}.png")

    from PIL import Image

    save_grid([Image.open(image_dir / f"{i:06}.png") for i in range(min(16, args.count))], args.output / "grid.png")
    print(f"{args.count} imágenes en {seconds:.1f} s ({seconds / args.count:.2f} s por imagen) → {args.output}")


if __name__ == "__main__":
    main()
