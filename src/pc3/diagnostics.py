"""Diagnósticos auxiliares; ni diversidad ni dHash prueban calidad/memorización."""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from pc3.config import Config
from pc3.data import _dhash, canonical_image, load_manifest, resolve_image
from pc3.io import file_sha256, read_json, write_json


def diagnose(config: Config, *, samples: Path, limit=64) -> dict:
    import numpy as np
    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure

    from pc3.tensor_data import GeneratedDataset

    samples = Path(samples)
    generation = read_json(samples / "generation.json")
    if not generation.get("complete") or generation["manifest_sha256"] != file_sha256(
        config.manifest
    ):
        raise ValueError("Generación incompleta o asociada a otro dataset.")
    manifest = load_manifest(config.manifest)
    count = min(limit, len(generation["images"]))
    if count < 2:
        raise ValueError("El diagnóstico necesita al menos dos imágenes.")
    indices = random.Random(config.seed + 3).sample(range(len(generation["images"])), count)
    records = [generation["images"][index] for index in indices]
    dataset = GeneratedDataset(samples, records)
    pixels = torch.stack([dataset[index] for index in range(count)]).float() / 255
    rng = random.Random(config.seed + 4)
    pairs = [rng.sample(range(count), 2) for _ in range(count)]
    with torch.inference_mode():
        # SSIM se aplica ENTRE GENERADAS como proxy estructural de diversidad,
        # nunca a una generada y una real arbitraria como supuesta fidelidad.
        # https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
        similarities = structural_similarity_index_measure(
            pixels[[pair[0] for pair in pairs]],
            pixels[[pair[1] for pair in pairs]],
            data_range=1.0,
            reduction="none",
        ).tolist()
    representatives = {}
    for row in manifest["records"]:
        if row["split"] == "train":
            representatives.setdefault(int(row["group"], 16), row["path"])
    neighbors = []
    for row in records:
        image = canonical_image(resolve_image(samples, row["path"]))
        code = int(min(_dhash(image), _dhash(ImageOps.mirror(image))), 16)
        nearest = min(representatives, key=lambda candidate: (candidate ^ code).bit_count())
        neighbors.append(
            {
                "generated": row["path"],
                "training": representatives[nearest],
                "hamming": (nearest ^ code).bit_count(),
            }
        )
    shown = neighbors[:8]
    sheet = Image.new("RGB", (240, 20 + len(shown) * 72), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((0, 2), "Generada | Train | dHash (proxy)", fill="black")
    for index, row in enumerate(shown):
        y = 20 + index * 72
        sheet.paste(canonical_image(resolve_image(samples, row["generated"])), (0, y))
        sheet.paste(canonical_image(resolve_image(Path(config.data_dir), row["training"])), (64, y))
        draw.text(
            (132, y + 12),
            f"Hamming: {row['hamming']}\n{Path(row['training']).name[:12]}",
            fill="black",
        )
    sheet.save(samples / "nearest_training.png")
    result = {
        "label": generation["label"],
        "manifest_sha256": generation["manifest_sha256"],
        "source_hash": generation["source_hash"],
        "count": count,
        "pairwise_ssim_mean": float(np.mean(similarities)),
        "pairwise_ssim_std": float(np.std(similarities)),
        "pairwise_ssim": similarities,
        "nearest_training_dhash": neighbors,
        "warnings": [
            "Menor SSIM entre generadas puede indicar diversidad o ruido; no es calidad.",
            "dHash mide estructura gruesa: un vecino no prueba memorización ni la descarta.",
            "La inspección se hace contra train, nunca se usa test para ajustar el entrenamiento.",
        ],
    }
    write_json(samples / "diagnostics.json", result)
    return result


def plot_diagnostics(paths, *, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = [read_json(path) for path in paths]
    if len({result["manifest_sha256"] for result in results}) != 1:
        raise ValueError("Los diagnósticos corresponden a distintos datasets.")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for result in results:
        axes[0].hist(
            result["pairwise_ssim"],
            bins=np.linspace(-1, 1, 21),
            histtype="step",
            density=True,
            label=result["label"],
        )
        axes[1].hist(
            [row["hamming"] for row in result["nearest_training_dhash"]],
            bins=np.arange(0, 69, 4),
            histtype="step",
            density=True,
            label=result["label"],
        )
    axes[0].set(title="Similitud entre generadas (proxy)", xlabel="SSIM", ylabel="Densidad")
    axes[1].set(
        title="Vecino más cercano en train (proxy)", xlabel="Distancia dHash", ylabel="Densidad"
    )
    for axis in axes:
        axis.legend()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
