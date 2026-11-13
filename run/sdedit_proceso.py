# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "diffusers==0.42.1", "safetensors", "numpy", "pillow", "tqdm"]
# ///
"""SDEdit mostrando el proceso: foto → ruido → imagen nueva (GPU si hay, si no CPU).

    uv run sdedit_proceso.py ruta/a/best.safetensors mi_foto.jpg
    uv run sdedit_proceso.py ruta/a/best.safetensors carpeta_de_fotos --strengths 0.5 0.8

Por cada foto y cada --strength escribe en run/output_sdedit/<foto>/:
  s0.50_proceso.png   tira: original · hacia el ruido (4 cuadros) · limpieza (hasta 13 cuadros)
  s0.50.gif           la misma secuencia animada
  s0.50.png           resultado final
La foto se recorta al centro y se reduce a 64×64, igual que los datos de entrenamiento.
Una intensidad baja conserva la foto; una alta se acerca a una cara anime libre.
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pc3.data import canonical_image  # noqa: E402
from pc3.inference import sdedit, to_pil, to_tensor  # noqa: E402
from pc3.model import load_weights  # noqa: E402

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("weights", type=Path, help="archivo best.safetensors")
    parser.add_argument("source", type=Path, help="una foto o una carpeta de fotos")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=HERE / "output_sdedit")
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"no existe el archivo de pesos: {args.weights}")
    if args.source.is_file():
        sources = [args.source]
    elif args.source.is_dir():
        sources = sorted(p for p in args.source.iterdir() if p.suffix.lower() in EXTENSIONS)
    else:
        parser.error(f"no existe: {args.source}")
    if not sources:
        parser.error(f"no hay imágenes en {args.source}")
    if args.steps < 1 or not all(0 < s <= 1 for s in args.strengths):
        parser.error("--steps debe ser positivo y cada --strengths estar en (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Dispositivo: {name} ({'fp16' if device.type == 'cuda' else 'fp32'}) · {args.steps} pasos DDIM")
    model, _ = load_weights(args.weights)
    model.to(device)

    for path in sources:
        photo = canonical_image(path)
        folder = args.output / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        for strength in args.strengths:
            begin = time.perf_counter()
            result, frames = sdedit(
                model, to_tensor(photo).unsqueeze(0), strength=strength, steps=args.steps,
                seed=args.seed, device=device, capture=True,
            )
            sequence = [photo, *frames]
            strip = Image.new("RGB", (64 * len(sequence), 64))
            for i, frame in enumerate(sequence):
                strip.paste(frame, (i * 64, 0))
            tag = f"s{strength:.2f}"
            strip.save(folder / f"{tag}_proceso.png")
            sequence[0].save(folder / f"{tag}.gif", save_all=True, append_images=sequence[1:], duration=200, loop=0)
            to_pil(result[0]).save(folder / f"{tag}.png")
            print(f"{path.name} · fuerza {strength:.2f}: {time.perf_counter() - begin:.1f} s")
    print("Resultados en", args.output)


if __name__ == "__main__":
    main()
