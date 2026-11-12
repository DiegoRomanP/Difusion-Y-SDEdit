"""Mide la calidad de las caras generadas por scripts/generate_images.py.

    uv run python scripts/evaluate_quality.py --samples reports/samples/anime64_4070ti

Un solo informe (--samples/quality.json) con tres partes:
  1. FID y KID: distancia entre generadas y reales de test (o val), con Inception.
     Menor es mejor. Inception se entrenó con fotos naturales: en anime es orientativo.
  2. Diversidad: SSIM entre pares de generadas. Muy alto = caras demasiado parecidas.
  3. Copia del dataset: para cada generada, la imagen de train más parecida por dHash.
     nearest_training.png las muestra lado a lado para revisar a ojo.
La primera ejecución descarga los pesos de Inception.
Fuentes: https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm.auto import tqdm

from pc3.config import load_project, read_json, write_json
from pc3.data import canonical_image, load_splits, mirror_dhash, split_records
from pc3.model import get_device


def as_uint8(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert("RGB")).transpose(2, 0, 1).copy())


def fid_kid(project, real_paths, fake_paths, *, device, batch_size):
    subset = min(project.kid_subset_size, len(real_paths))
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        feature=2048, subsets=project.kid_subsets, subset_size=subset, normalize=False
    ).to(device)
    with torch.inference_mode():
        for is_real, paths, load in (
            (True, real_paths, canonical_image),
            (False, fake_paths, Image.open),
        ):
            for start in tqdm(
                range(0, len(paths), batch_size), desc="reales" if is_real else "generadas"
            ):
                images = torch.stack([as_uint8(load(p)) for p in paths[start : start + batch_size]])
                fid.update(images.to(device), real=is_real)
                kid.update(images.to(device), real=is_real)
        torch.manual_seed(project.seed)  # Mismos subconjuntos de KID en cada evaluación.
        kid_mean, kid_std = (float(value) for value in kid.compute())
        return {
            "fid": float(fid.compute()),
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "kid_subset_size": subset,
        }


def diversity(project, fake_paths, limit):
    """SSIM entre pares de generadas (nunca contra reales: aquí no mide fidelidad)."""
    count = min(limit, len(fake_paths))
    chosen = random.Random(project.seed + 3).sample(fake_paths, count)
    pixels = torch.stack([as_uint8(Image.open(p)) for p in chosen]).float() / 255
    rng = random.Random(project.seed + 4)
    pairs = [rng.sample(range(count), 2) for _ in range(count)]
    similarities = structural_similarity_index_measure(
        pixels[[a for a, _ in pairs]],
        pixels[[b for _, b in pairs]],
        data_range=1.0,
        reduction="none",
    ).tolist()
    return chosen, {
        "pairwise_ssim_mean": float(np.mean(similarities)),
        "pairwise_ssim_std": float(np.std(similarities)),
        "pairwise_ssim": similarities,
    }


def nearest_training(project, splits, generated_paths, sheet_path):
    """Vecino de train con menor distancia dHash. Un vecino cercano no prueba copia."""
    representatives = {}
    for row in split_records(splits, "train"):
        representatives.setdefault(int(row["group"], 16), row["path"])
    neighbors = []
    for path in generated_paths:
        code = mirror_dhash(Image.open(path).convert("RGB"))
        nearest = min(representatives, key=lambda candidate: (candidate ^ code).bit_count())
        neighbors.append(
            {
                "generated": Path(path).name,
                "training": representatives[nearest],
                "hamming": (nearest ^ code).bit_count(),
            }
        )
    shown = list(zip(generated_paths, neighbors, strict=True))[:8]
    sheet = Image.new("RGB", (240, 20 + len(shown) * 72), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((0, 2), "Generada | Train | dHash", fill="black")
    for index, (path, row) in enumerate(shown):
        y = 20 + index * 72
        sheet.paste(Image.open(path).convert("RGB"), (0, y))
        sheet.paste(canonical_image(Path(project.data_dir) / row["training"]), (64, y))
        draw.text((132, y + 12), f"Hamming: {row['hamming']}", fill="black")
    sheet.save(sheet_path)
    return neighbors


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path, required=True, help="Salida de generate_images")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--num-images", type=int, help="Def.: sample_count o lo disponible")
    parser.add_argument("--diversity-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project = load_project()
    device = get_device(args.device)
    splits = load_splits(project.splits_file)
    generation = read_json(args.samples / "generation.json")
    fake_paths = sorted((args.samples / "images").glob("*.png"))
    real = split_records(splits, args.split)  # Nunca train: sería medir contra lo memorizado.
    random.Random(project.seed + 2_000_000).shuffle(real)
    count = args.num_images or min(project.sample_count, len(real), len(fake_paths))
    real_paths = [Path(project.data_dir) / row["path"] for row in real[:count]]
    fake_paths = fake_paths[:count]

    print(f"FID/KID con {count} imágenes de {args.split} y {count} generadas.")
    metrics = fid_kid(project, real_paths, fake_paths, device=device, batch_size=args.batch_size)
    chosen, similarity = diversity(project, fake_paths, args.diversity_images)
    neighbors = nearest_training(project, splits, chosen, args.samples / "nearest_training.png")

    warnings = [
        "FID/KID usan Inception entrenada con fotos naturales: en anime son orientativos.",
        "KID std describe la variación entre subconjuntos; no es un intervalo de confianza.",
        "SSIM bajo entre generadas puede ser diversidad o ruido; no es calidad.",
        "dHash mide estructura gruesa: un vecino cercano no prueba memorización.",
    ]
    if count < 5000:
        warnings.append(f"N={count}: con menos de 5000 imágenes el FID está sesgado al alza.")
    result = {
        "label": generation["label"],
        "split": args.split,
        "num_images": count,
        "sampler": generation["sampler"],
        "sampling_steps": generation["steps"],
        **metrics,
        **similarity,
        "nearest_training_dhash": neighbors,
        "seconds_per_image": generation["seconds_per_image"],
        "images_per_second": generation["images_per_second"],
        "peak_vram_gib": generation["peak_vram_gib"],
        "warnings": warnings,
    }
    write_json(args.samples / "quality.json", result)
    print(
        f"FID {metrics['fid']:.3f} | KID {metrics['kid_mean']:.6f} ± {metrics['kid_std']:.6f} | "
        f"SSIM medio {similarity['pairwise_ssim_mean']:.3f} → {args.samples / 'quality.json'}"
    )


if __name__ == "__main__":
    main()
