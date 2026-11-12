"""SDEdit: parte de una foto real en vez de ruido y deja que el modelo la lleve hacia anime.

    uv run python scripts/sdedit.py --weights runs/anime64_4070ti/best.safetensors \
        --source data/real_faces_demo --strengths 0.3 0.5 0.7 0.9

La foto se recorta a 64x64, se le añade ruido hasta un paso intermedio y desde ahí se
sigue el muestreo normal. --strength controla cuánto: 0.3 conserva casi toda la foto,
0.9 la acerca mucho a las caras anime del entrenamiento. No es traducción controlada:
el modelo nunca vio fotos ni pares foto-anime, y no conserva la identidad.

Escribe en --output, por cada foto: <nombre>_s0.50.png por intensidad y
<nombre>_comparacion.png (original + todas las intensidades en fila).
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from pc3.data import canonical_image
from pc3.inference import sdedit, to_pil, to_tensor
from pc3.model import get_device, load_weights

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LABEL_HEIGHT = 16


def labeled(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (64, 64 + LABEL_HEIGHT), "white")
    canvas.paste(image, (0, LABEL_HEIGHT))
    ImageDraw.Draw(canvas).text((2, 2), text, fill="black")
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Foto o carpeta de fotos")
    parser.add_argument("--output", type=Path, default=Path("reports/sdedit"))
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    model, _ = load_weights(args.weights)
    model.to(device)
    if args.source.is_file():
        sources = [args.source]
    else:
        sources = sorted(p for p in args.source.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not sources:
        raise SystemExit(f"No hay imágenes en {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)

    for path in sources:
        photo = canonical_image(path)
        columns = [labeled(photo, "original")]
        for strength in args.strengths:
            result = sdedit(
                model,
                to_tensor(photo).unsqueeze(0),
                strength=strength,
                steps=args.steps,
                sampler=args.sampler,
                seed=args.seed,
                device=device,
                amp=not args.no_amp,
            )
            image = to_pil(result[0])
            image.save(args.output / f"{path.stem}_s{strength:.2f}.png")
            columns.append(labeled(image, f"s={strength:.2f}"))
        row = Image.new("RGB", (64 * len(columns), 64 + LABEL_HEIGHT), "white")
        for index, column in enumerate(columns):
            row.paste(column, (index * 64, 0))
        row.save(args.output / f"{path.stem}_comparacion.png")
        print("listo:", path.name)
    print("Resultados en", args.output)


if __name__ == "__main__":
    main()
