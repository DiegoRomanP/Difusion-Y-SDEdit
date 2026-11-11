"""FID/KID con test reservado y contrato de comparación explícito.

Los pesos de Inception se descargan únicamente al invocar evaluate, no al importar.
Fuentes: https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html
"""

import importlib.metadata
import math
import random
from pathlib import Path

from pc3.config import Config, training_contract
from pc3.data import load_manifest
from pc3.io import file_sha256, fingerprint, read_json, source_fingerprint, write_json


def metric_protocol(digest, split, real_paths, count, subset_size, subsets, seed, generation):
    return {
        "manifest_sha256": digest,
        "split": split,
        "num_images": count,
        "real_subset_sha256": fingerprint(real_paths),
        "image_size": 64,
        "extractor": "torch-fidelity/inception-v3-compat/2048",
        "input": "RGB uint8 [0,255]",
        "resize": "TorchMetrics Inception interno 299x299",
        "kid_subset_size": subset_size,
        "kid_subsets": subsets,
        "kid_seed": seed,
        "sampler": generation["sampler"],
        "sampling_steps": generation["steps"],
        "generation_seed": generation["seed"],
        "generation_batch_size": generation["batch_size"],
        "generation_amp": generation["amp"],
        "pc3_source_sha256": generation["pc3_source_sha256"],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "torchmetrics", "torch-fidelity", "pillow")
        },
    }


# Lo que no describe comparabilidad sino contaminación entre train y test: si
# esto difiere, el número no es peor, es inválido. Sigue siendo error duro.
HARD_PROTOCOL_KEYS = ("manifest_sha256", "real_subset_sha256", "split")

BUDGET_KINDS = frozenset({"presupuesto_incompleto", "imagenes_consumidas"})


def _differing_keys(left: dict, right: dict) -> list[str]:
    return sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))


def divergences(results: list[dict]) -> list[dict]:
    """Diferencias que el informe debe declarar en vez de ocultar rechazándose.

    Rechazar la salida escondía la limitación; anotarla la pone en la figura.
    ADR 002 lo decide así, con la excepción de HARD_PROTOCOL_KEYS.
    """
    if len(results) < 2 or len({result["label"] for result in results}) != len(results):
        raise ValueError("Se requieren al menos dos modelos con etiquetas únicas.")
    reference, found = results[0], []
    for result in results[1:]:
        label = result["label"]
        for key in HARD_PROTOCOL_KEYS:
            if result["protocol"].get(key) != reference["protocol"].get(key):
                raise ValueError(
                    f"{key} difiere entre {reference['label']!r} y {label!r}: no es una "
                    "divergencia de perfil sino contaminación entre train y test."
                )
        soft = _differing_keys(
            {k: v for k, v in result["protocol"].items() if k not in HARD_PROTOCOL_KEYS},
            {k: v for k, v in reference["protocol"].items() if k not in HARD_PROTOCOL_KEYS},
        )
        if soft:
            found.append({"kind": "protocolo", "label": label, "detail": soft})
        contract, ref_contract = (
            result["provenance"]["contract"],
            reference["provenance"]["contract"],
        )
        if contract != ref_contract:
            found.append(
                {
                    "kind": "contrato_de_entrenamiento",
                    "label": label,
                    "detail": _differing_keys(contract, ref_contract),
                }
            )
        if result["generation"]["hardware"] != reference["generation"]["hardware"]:
            found.append(
                {
                    "kind": "hardware_de_inferencia",
                    "label": label,
                    "detail": {
                        "referencia": reference["generation"]["hardware"].get("name"),
                        "este": result["generation"]["hardware"].get("name"),
                    },
                    "note": "El coste no se compara entre equipos distintos.",
                }
            )
        if result["generation"]["count"] != reference["generation"]["count"]:
            found.append(
                {
                    "kind": "n_generado",
                    "label": label,
                    "detail": {
                        "referencia": reference["generation"]["count"],
                        "este": result["generation"]["count"],
                    },
                }
            )
    for result in results:
        run = result.get("run_summary") or {}
        contract = result["provenance"]["contract"]
        budget = contract["max_steps"]
        if run.get("status") != "completed" or run.get("completed_steps") != budget:
            found.append(
                {
                    "kind": "presupuesto_incompleto",
                    "label": result["label"],
                    "detail": {
                        "status": run.get("status"),
                        "completed_steps": run.get("completed_steps"),
                        "max_steps": budget,
                    },
                }
            )
            continue
        expected = budget * contract.get("batch_size", 1) * contract.get("accumulation_steps", 1)
        if run.get("consumed_images") != expected:
            found.append(
                {
                    "kind": "imagenes_consumidas",
                    "label": result["label"],
                    "detail": {"esperado": expected, "registrado": run.get("consumed_images")},
                }
            )
    return found


def assert_comparable(results: list[dict], *, allow_partial=False) -> None:
    """Rechazo duro: lo usan la ablación y `report --strict`."""
    found = divergences(results)
    if allow_partial:
        found = [item for item in found if item["kind"] not in BUDGET_KINDS]
    if found:
        raise ValueError(f"Resultados no comparables: {found}")


def evaluate(
    config: Config,
    *,
    samples: Path,
    output: Path | None = None,
    split="test",
    num_images: int | None = None,
    device_name="cuda",
    batch_size=4,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from tqdm.auto import tqdm

    from pc3.sampling import require_device
    from pc3.tensor_data import FaceDataset, GeneratedDataset

    config.validate()
    if split not in {"val", "test"} or batch_size < 1:
        raise ValueError("Evaluar en val/test y batch_size >=1.")
    device = require_device(device_name)
    manifest = load_manifest(config.manifest)
    samples = Path(samples)
    generation = read_json(samples / "generation.json")
    if generation["pc3_source_sha256"] != source_fingerprint():
        raise ValueError(
            "El código cambió desde sample; mantener la versión original del experimento."
        )
    if not generation.get("complete") or generation["count"] != len(generation["images"]):
        raise ValueError("La generación está incompleta; reanudar sample antes de evaluar.")
    digest = file_sha256(config.manifest)
    if generation["manifest_sha256"] != digest:
        raise ValueError("Las muestras se generaron para otro manifiesto.")
    if generation["provenance"]["contract"] != training_contract(config):
        raise ValueError("El perfil de evaluación no corresponde al experimento.")
    real = [row for row in manifest["records"] if row["split"] == split]
    random.Random(config.seed + 2000000).shuffle(real)
    available = min(len(real), generation["count"])
    count = min(config.sample_count, available) if num_images is None else num_images
    if not 2 <= count <= available:
        raise ValueError(f"Se requieren 2..{available} imágenes reales y generadas.")
    real = real[:count]
    subset = min(config.kid_subset_size, count)
    protocol = metric_protocol(
        digest,
        split,
        [row["path"] for row in real],
        count,
        subset,
        config.kid_subsets,
        config.seed,
        generation,
    )
    print(
        "Evaluando FID/KID; la primera ejecución puede descargar los pesos oficiales de Inception."
    )
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        feature=2048, subsets=config.kid_subsets, subset_size=subset, normalize=False
    ).to(device)
    real_data = FaceDataset(Path(config.data_dir), real, config.seed)
    fake_data = GeneratedDataset(samples, generation["images"][:count])
    with torch.inference_mode():
        for is_real, dataset in ((True, real_data), (False, fake_data)):
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=config.num_workers)
            for images in tqdm(loader, desc="reales" if is_real else generation["label"]):
                if is_real:
                    images = ((images + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
                images = images.to(device)
                fid.update(images, real=is_real)
                kid.update(images, real=is_real)
        # Mismas selecciones de subconjuntos para todos los modelos.
        torch.manual_seed(config.seed)
        fid_value = float(fid.compute())
        kid_mean, kid_std = (float(value) for value in kid.compute())
    if not all(math.isfinite(value) for value in (fid_value, kid_mean, kid_std)):
        raise FloatingPointError("Métricas no finitas; no se escribirá un resultado válido.")
    warnings = [
        "FID/KID usan Inception entrenada en imágenes naturales: su validez en anime es limitada.",
        "KID std describe subconjuntos; no es un IC ni variabilidad entre entrenamientos.",
        "Menor MSE de ruido no garantiza menor FID ni mejor calidad visual.",
    ]
    if count < 5000:
        warnings.append(
            f"Solo N={count}: comparación exploratoria; FID es sensible al tamaño muestral."
        )
    result = {
        "schema": 1,
        "label": generation["label"],
        "protocol": protocol,
        "fid": fid_value,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "seconds_per_image": generation["seconds_per_image"],
        "images_per_second": generation["images_per_second"],
        "peak_vram_gib": generation["peak_vram_gib"],
        "provenance": generation["provenance"],
        "run_summary": generation.get("run_summary"),
        "generation": {key: generation[key] for key in ("hardware", "count", "source_hash")},
        "warnings": warnings,
    }
    write_json(output or samples / "metrics.json", result)
    return result
