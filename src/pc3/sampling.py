"""Muestreo DDPM/DDIM, semillas por imagen y evidencias de inferencia reanudables."""

import math
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from tqdm.auto import tqdm

from pc3.config import Config, training_contract
from pc3.io import file_sha256, read_json, source_fingerprint, write_json
from pc3.model import create_model, load_weights, noise_scheduler


def device_info(device: torch.device) -> dict:
    info = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(name=properties.name, total_vram_gib=properties.total_memory / 1024**3)
    else:
        info["name"] = "CPU"
    return info


def require_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Este proyecto admite cpu o cuda[:índice].")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Instalar con uv sync --extra cu128 y revisar el driver; "
            "--device cpu solo es una alternativa explícita, mucho más lenta."
        )
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def sample_tensor(model, *, seeds, steps=50, sampler="ddim", device, amp=True, capture=False):
    """DDIM eta=0: mismo ruido inicial por imagen, incluso cambiando el batch.

    DDPM usa además un generador independiente por imagen para el ruido de cada
    transición. Fuente: https://huggingface.co/docs/diffusers/api/schedulers/ddim
    """
    if sampler not in {"ddim", "ddpm"} or not 1 <= steps <= 1000 or not seeds:
        raise ValueError("Usar ddim/ddpm, 1..1000 pasos y al menos una semilla.")
    scheduler = noise_scheduler()
    if sampler == "ddim":
        scheduler = DDIMScheduler.from_config(scheduler.config)
    scheduler.set_timesteps(steps, device=device)
    image = torch.stack(
        [torch.randn((3, 64, 64), generator=torch.Generator().manual_seed(seed)) for seed in seeds]
    ).to(device)
    generators = [torch.Generator().manual_seed(seed + 10000000) for seed in seeds]
    frames = [to_pil(image[0])] if capture else []
    checkpoints = set(np.linspace(0, steps - 1, min(12, steps), dtype=int).tolist())
    was_training = model.training
    model.eval()
    try:
        for index, timestep in enumerate(scheduler.timesteps):
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
            ):
                prediction = model(image, timestep).sample
            kwargs = {"eta": 0.0} if sampler == "ddim" else {"generator": generators}
            image = scheduler.step(prediction.float(), timestep, image, **kwargs).prev_sample
            if capture and index in checkpoints:
                frames.append(to_pil(image[0]))
    finally:
        model.train(was_training)
    return image.detach().cpu(), frames


def to_pil(tensor) -> Image.Image:
    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round()
    return Image.fromarray(array.to(torch.uint8).permute(1, 2, 0).numpy())


def save_grid(images, path: Path, columns=4):
    images = list(images)
    grid = Image.new("RGB", (columns * 64, math.ceil(len(images) / columns) * 64))
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * 64, (index // columns) * 64))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def generate(
    config: Config,
    *,
    output: Path,
    weights: Path | None,
    device_name="cuda",
    sampler="ddim",
    label="modelo",
) -> dict:
    """La referencia sin adaptar se selecciona con weights=None; descarga solo al invocarla."""
    config.validate()
    device = require_device(device_name)
    output = Path(output)
    source_hash = file_sha256(weights) if weights else config.pretrained_revision
    request = {
        "schema": 1,
        "label": label,
        "sampler": sampler,
        "steps": config.sampling_steps,
        "count": config.sample_count,
        "batch_size": config.sample_batch_size,
        "seed": config.sample_seed,
        "amp": config.amp,
        "image_size": 64,
        "source_hash": source_hash,
        "hardware": device_info(device),
        "manifest_sha256": file_sha256(config.manifest),
        "pc3_source_sha256": source_fingerprint(),
    }
    if output.exists() and any(output.iterdir()):
        if not (output / "request.json").exists() or read_json(output / "request.json") != request:
            raise FileExistsError("La salida pertenece a otra generación; elegir otro directorio.")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("images", "grid.png", "denoising.gif"):
        if (output / name).is_symlink():
            raise ValueError("No escribir resultados a través de enlaces simbólicos.")
    write_json(output / "request.json", request)
    run_summary = None
    if weights:
        run_file = Path(weights).parent / "run.json"
        if run_file.is_file():
            run = read_json(run_file)
            if run["versions"]["pc3_source_sha256"] != request["pc3_source_sha256"]:
                raise ValueError(
                    "El código cambió desde el entrenamiento; conservar el código original."
                )
            if run["status"] in {"training", "initializing"}:
                raise ValueError(
                    "Esperar a que termine/se pause train; ya guarda previews periódicos."
                )
            run_summary = {
                key: run[key]
                for key in (
                    "run_id",
                    "status",
                    "completed_steps",
                    "consumed_images",
                    "wall_seconds",
                    "parameter_count",
                    "hardware",
                    "versions",
                )
            }
        model, provenance = load_weights(weights)
        if file_sha256(weights) != source_hash:
            raise ValueError("Los pesos cambiaron mientras se cargaban; repetir sin train activo.")
        if run_summary and run_summary["run_id"] != provenance["run_id"]:
            raise ValueError("Los pesos y run.json pertenecen a ejecuciones distintas.")
        if provenance["manifest_sha256"] != request["manifest_sha256"]:
            raise ValueError("Los pesos se entrenaron con otro manifiesto.")
        if provenance["contract"] != training_contract(config):
            raise ValueError("La configuración no corresponde a los pesos entrenados.")
    else:
        model = create_model(config, pretrained=True)
        provenance = {
            "initialization": "pretrained_reference",
            "step": 0,
            "weights": "pretrained",
            "contract": training_contract(config),
            "manifest_sha256": request["manifest_sha256"],
        }
    model.to(device)
    progress_path = output / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"images": {}, "batches": {}}
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    # Warmup excluido de las mediciones; no guardar esta muestra como resultado.
    sample_tensor(model, seeds=[config.sample_seed], steps=1, device=device, amp=config.amp)
    synchronize(device)
    for start in tqdm(range(0, config.sample_count, config.sample_batch_size), desc=label):
        stop = min(config.sample_count, start + config.sample_batch_size)
        names = [f"{index:06}.png" for index in range(start, stop)]
        if any((image_dir / name).is_symlink() for name in names):
            raise ValueError("Una muestra de salida es un enlace simbólico; no se sobrescribirá.")
        intact = all(
            name in progress["images"]
            and (image_dir / name).is_file()
            and file_sha256(image_dir / name) == progress["images"][name]
            for name in names
        )
        if intact and (start != 0 or (output / "denoising.gif").exists()):
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        begin = time.perf_counter()
        images, _ = sample_tensor(
            model,
            seeds=list(range(config.sample_seed + start, config.sample_seed + stop)),
            steps=config.sampling_steps,
            sampler=sampler,
            device=device,
            amp=config.amp,
        )
        synchronize(device)
        seconds = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
        for name, tensor in zip(names, images, strict=True):
            to_pil(tensor).save(image_dir / name)
            progress["images"][name] = file_sha256(image_dir / name)
        progress["batches"][str(start)] = {"seconds": seconds, "peak_vram_gib": peak}
        if start == 0:
            # Capturar cuadros implica copias GPU→CPU; se mide en otra pasada para
            # no falsear la comparación de tiempos entre modelos.
            _, frames = sample_tensor(
                model,
                seeds=[config.sample_seed],
                steps=config.sampling_steps,
                sampler=sampler,
                device=device,
                amp=config.amp,
                capture=True,
            )
            imageio.mimsave(
                output / "denoising.gif",
                [np.array(frame) for frame in frames],
                duration=0.15,
                loop=0,
            )
        write_json(progress_path, progress)
    seconds = sum(batch["seconds"] for batch in progress["batches"].values())
    result = {
        **request,
        "complete": True,
        "provenance": provenance,
        "run_summary": run_summary,
        "inference_seconds": seconds,
        "seconds_per_image": seconds / config.sample_count,
        "images_per_second": config.sample_count / seconds,
        "peak_vram_gib": max(batch["peak_vram_gib"] for batch in progress["batches"].values()),
        "images": [
            {"path": f"images/{name}", "sha256": digest}
            for name, digest in sorted(progress["images"].items())
        ],
        "configuration": asdict(config),
    }
    save_grid(
        [Image.open(image_dir / f"{i:06}.png").copy() for i in range(min(16, config.sample_count))],
        output / "grid.png",
    )
    write_json(output / "generation.json", result)
    return result
