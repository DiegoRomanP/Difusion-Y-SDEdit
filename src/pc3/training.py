"""Bucle PyTorch propio: acumulación, FP16, EMA y reanudación a límites de update.

No se importa desde prepare/--help y nunca arranca al importar el notebook.
Fuentes: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
https://huggingface.co/docs/diffusers/training/unconditional_training
"""

import csv
import importlib.metadata
import platform
import random
import signal
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pc3.config import Config, training_contract
from pc3.data import load_manifest
from pc3.io import file_sha256, fingerprint, read_json, source_fingerprint, write_json
from pc3.model import (
    ExponentialMovingAverage,
    create_model,
    export_weights,
    noise_scheduler,
    q_sample,
)
from pc3.sampling import device_info, require_device, sample_tensor, save_grid, synchronize, to_pil
from pc3.stream import ReplayBatchSampler, learning_rate_at
from pc3.tensor_data import FaceDataset


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "pc3_source_sha256": source_fingerprint(),
        **{
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "diffusers", "safetensors", "pillow", "numpy")
        },
    }


def _loader(dataset, config, device, *, sampler=None):
    kwargs = {"batch_sampler": sampler} if sampler else {"batch_size": config.batch_size}
    return DataLoader(
        dataset,
        # DataLoader sortea base_seed incluso con shuffle=False. Separarlo evita
        # consumir el RNG de la red cada vez que se reconstruye al reanudar.
        # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        generator=torch.Generator().manual_seed(config.seed + 5),
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        **kwargs,
    )


@torch.inference_mode()
def validation_loss(model, loader, config, device, alphas) -> float:
    """Mismos x, epsilon y t por imagen en cada validación; no consume RNG de train."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for clean in loader:
            noises, times = [], []
            for index in range(count, count + len(clean)):
                rng = torch.Generator().manual_seed(config.seed + 1000000 + index)
                times.append(torch.randint(1000, (1,), generator=rng))
                noises.append(torch.randn((3, 64, 64), generator=rng))
            clean = clean.to(device, non_blocking=True)
            noise, times = torch.stack(noises).to(device), torch.cat(times).to(device)
            noisy = q_sample(clean, times, noise, alphas)
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=config.amp and device.type == "cuda"
            ):
                prediction = model(noisy, times).sample
            loss = F.mse_loss(prediction.float(), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("La pérdida de validación no es finita.")
            total += float(loss) * len(clean)
            count += len(clean)
    finally:
        model.train(was_training)
    return total / count


def _save_checkpoint(path, state):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _write_history(path, rows):
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(
    config: Config, *, initialization: str, run_dir: Path, resume=False, device_name="cuda"
) -> dict:
    config.validate()
    if initialization not in {"random", "pretrained"}:
        raise ValueError("Inicialización random o pretrained requerida.")
    device = require_device(device_name)  # Fallar antes de cualquier descarga de pesos.
    manifest = load_manifest(config.manifest)
    if manifest["seed"] != config.seed:
        raise ValueError("La semilla de prepare y train debe ser la misma.")
    contract = training_contract(config)
    digest = file_sha256(config.manifest)
    run_dir = Path(run_dir)
    metadata_path, checkpoint_path = run_dir / "run.json", run_dir / "latest.pt"
    if resume:
        if not metadata_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("No hay run.json/latest.pt reanudables en ese directorio.")
        metadata = read_json(metadata_path)
        if (
            metadata["contract"] != contract
            or metadata["manifest_sha256"] != digest
            or metadata["initialization"] != initialization
        ):
            raise ValueError("No cambiar datos, inicialización o contrato al reanudar.")
        if metadata["versions"] != runtime_versions() or metadata["hardware"] != device_info(
            device
        ):
            raise ValueError(
                "Para reanudar reproduciblemente se requiere el mismo software/hardware."
            )
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("La ejecución ya existe; usar --resume o un directorio nuevo.")
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": 1,
            "run_id": str(uuid.uuid4()),
            "initialization": initialization,
            "created_utc": datetime.now(UTC).isoformat(),
            "configuration": asdict(config),
            "contract": contract,
            "manifest_sha256": digest,
            "hardware": device_info(device),
            "versions": runtime_versions(),
            "completed_steps": 0,
            "status": "initializing",
        }
        write_json(metadata_path, metadata)
    lock_path = run_dir / ".training.lock"
    lock_id = str(uuid.uuid4())
    try:
        with lock_path.open("x", encoding="utf-8") as file:
            file.write(lock_id)
    except FileExistsError as error:
        raise RuntimeError(
            "La ejecución está bloqueada. Si hubo SIGKILL, comprobar que ningún proceso la usa "
            "antes de retirar manualmente .training.lock."
        ) from error
    try:
        return _train_locked(config, initialization, run_dir, resume, device, manifest, metadata)
    finally:
        # Solo retirar el archivo que acabamos de crear, no un bloqueo ajeno.
        if lock_path.is_file() and lock_path.read_text() == lock_id:
            lock_path.unlink()


def _train_locked(config, initialization, run_dir, resume, device, manifest, metadata):
    # Reduce fuentes de variabilidad; no promete identidad bit a bit entre GPUs.
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    model = create_model(config, pretrained=initialization == "pretrained" and not resume).to(
        device
    )
    model.train().requires_grad_(True)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    ema = ExponentialMovingAverage(model, config.ema_decay, config.ema_device)
    rng = torch.Generator(device=device).manual_seed(config.seed)
    step, microstep, best, previous_wall, skipped = 0, 0, float("inf"), 0.0, 0
    history = []
    # Independizar el azar del forward del usado al inicializar/cargar los pesos.
    torch.manual_seed(config.seed + 7)
    if resume:
        # Nuestros checkpoints contienen únicamente tensores/tipos simples.
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
        if state["run_id"] != metadata["run_id"]:
            raise ValueError("El checkpoint no pertenece a esta ejecución.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        ema.shadow = {k: v.to(config.ema_device) for k, v in state["ema"].items()}
        step, microstep, best = state["step"], state["microstep"], state["best_val"]
        previous_wall, skipped, history = state["wall_seconds"], state["skipped"], state["history"]
        rng.set_state(state["noise_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    records = [row for row in manifest["records"] if row["split"] == "train"]
    validation = [row for row in manifest["records"] if row["split"] == "val"]
    random.Random(config.seed).shuffle(validation)
    validation = validation[: config.validation_images]
    train_loader = _loader(
        FaceDataset(Path(config.data_dir), records, config.seed, augment=True),
        config,
        device,
        sampler=ReplayBatchSampler(len(records), config.batch_size, config.seed, start=microstep),
    )
    val_loader = _loader(
        FaceDataset(Path(config.data_dir), validation, config.seed), config, device
    )
    iterator = iter(train_loader)
    alphas = noise_scheduler().alphas_cumprod.to(device)
    started, stop_requested = time.perf_counter(), False
    old_handlers = {}

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("\nParada solicitada: guardando al terminar la actualización actual.", flush=True)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, request_stop)

    def checkpoint():
        _save_checkpoint(
            run_dir / "latest.pt",
            {
                "schema": 1,
                "run_id": metadata["run_id"],
                "step": step,
                "microstep": microstep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "noise_rng": rng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "best_val": best,
                "wall_seconds": previous_wall + time.perf_counter() - started,
                "skipped": skipped,
                "history": history,
            },
        )
        _write_history(run_dir / "history.csv", history)
        metadata.update(
            completed_steps=step,
            consumed_images=microstep * config.batch_size,
            skipped_fp16_attempts=skipped,
            best_validation_mse=best,
            wall_seconds=previous_wall + time.perf_counter() - started,
        )
        write_json(run_dir / "run.json", metadata)

    def evaluate_and_preview(row):
        nonlocal best
        with ema.apply(model):
            row["val_mse"] = validation_loss(model, val_loader, config, device, alphas)
            provenance = {
                "run_id": metadata["run_id"],
                "initialization": initialization,
                "contract": metadata["contract"],
                "manifest_sha256": metadata["manifest_sha256"],
                "step": step,
                "weights": "ema",
                "validation_mse": row["val_mse"],
            }
            if row["val_mse"] < best:
                best = row["val_mse"]
                export_weights(model, run_dir / "best.safetensors", provenance)
            images = []
            for start in range(2026, 2042, config.sample_batch_size):
                batch, _ = sample_tensor(
                    model,
                    seeds=list(range(start, min(2042, start + config.sample_batch_size))),
                    steps=20,
                    device=device,
                    amp=config.amp,
                )
                images.extend(to_pil(image) for image in batch)
            save_grid(images, run_dir / "previews" / f"step_{step:06}.png")
            if step == config.max_steps or stop_requested:
                export_weights(model, run_dir / "last.safetensors", provenance)

    metadata.update(
        status="training",
        parameter_count=sum(p.numel() for p in model.parameters()),
        validation_subset_sha256=fingerprint([row["path"] for row in validation]),
        preview_protocol={"sampler": "ddim", "steps": 20, "seeds": list(range(2026, 2042))},
    )
    write_json(run_dir / "run.json", metadata)
    try:
        if not resume:
            initial = {"step": 0, "kind": "initial"}
            evaluate_and_preview(initial)
            history.append(initial)
            checkpoint()
        progress = tqdm(total=config.max_steps, initial=step, desc=initialization)
        while step < config.max_steps and not stop_requested:
            synchronize(device)
            begin = time.perf_counter()
            # Se conservan microbatches CPU para repetirlos si FP16 desborda. Así
            # no se saltan ejemplos ni cambia el orden respecto al otro modelo.
            batches = [next(iterator) for _ in range(config.accumulation_steps)]
            noise_state, torch_state = rng.get_state(), torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            failures = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for clean in batches:
                    clean = clean.to(device, non_blocking=True)
                    noise = torch.randn(clean.shape, generator=rng, device=device)
                    times = torch.randint(1000, (len(clean),), generator=rng, device=device)
                    noisy = q_sample(clean, times, noise, alphas)
                    with torch.autocast(
                        device.type,
                        dtype=torch.float16,
                        enabled=config.amp and device.type == "cuda",
                    ):
                        prediction = model(noisy, times).sample
                    loss = F.mse_loss(prediction.float(), noise.float())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("MSE no finita: revisar datos/configuración.")
                    total_loss += float(loss.detach()) / config.accumulation_steps
                    scaler.scale(loss / config.accumulation_steps).backward()
                rate = learning_rate_at(
                    step, config.max_steps, config.warmup_steps, config.learning_rate
                )
                for group in optimizer.param_groups:
                    group["lr"] = rate
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    break
                skipped, failures = skipped + 1, failures + 1
                if failures >= 16:
                    raise FloatingPointError(
                        "FP16 sigue desbordando tras 16 intentos; no se avanzó el dataset."
                    )
                rng.set_state(noise_state)
                torch.set_rng_state(torch_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, device)
            ema.update(model)
            step += 1
            microstep += config.accumulation_steps
            synchronize(device)
            update_seconds = time.perf_counter() - begin
            row = {
                "step": step,
                "kind": "train",
                "train_mse": total_loss,
                "learning_rate": rate,
                "update_seconds": update_seconds,
                "consumed_images": microstep * config.batch_size,
                "epoch_equivalent": microstep * config.batch_size / len(records),
                "wall_seconds": previous_wall + time.perf_counter() - started,
                "peak_vram_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            validate = (
                step % config.validate_every == 0 or step == config.max_steps or stop_requested
            )
            if validate:
                optimizer.zero_grad(set_to_none=True)
                evaluate_and_preview(row)
            if step % config.log_every == 0 or validate:
                history.append(row)
                _write_history(run_dir / "history.csv", history)
                progress.set_postfix(mse=f"{total_loss:.4f}")
            if step % config.checkpoint_every == 0 or validate:
                checkpoint()
            progress.update(1)
        progress.close()
        metadata["status"] = "completed" if step == config.max_steps else "paused"
        checkpoint()
        return metadata
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def calibrate(config: Config, *, device_name="cuda", updates=3) -> dict:
    """Medición sintética explícita del compañero; no usa dataset ni pesos abiertos."""
    config.validate()
    if updates < 1:
        raise ValueError("Se requiere al menos una actualización medida.")
    device = require_device(device_name)
    torch.manual_seed(config.seed)
    model = create_model(config).to(device).train()
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    ema = ExponentialMovingAverage(model, config.ema_decay, config.ema_device)
    clean = torch.randn(
        config.batch_size, 3, config.image_size, config.image_size, device=device
    )
    times = torch.zeros(config.batch_size, dtype=torch.long, device=device)
    # Una sola iteración de warmup no basta: el autotune de cuDNN y el crecimiento
    # del asignador siguen pagándose en las primeras. Con `updates=3` eso infló una
    # medición real de la 3060 por 8.2x, así que el warmup escala con la petición.
    warmup = max(3, updates // 5)
    durations, skipped = [], 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while len(durations) < updates + warmup:
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.accumulation_steps):
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=config.amp and device.type == "cuda"
            ):
                predicted = model(clean, times).sample
            loss = predicted.float().square().mean() / config.accumulation_steps
            scaler.scale(loss).backward()
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale:
            skipped += 1
            if skipped >= 16:
                raise FloatingPointError("La calibración FP16 no logró una actualización finita.")
            continue
        ema.update(model)
        synchronize(device)
        durations.append(time.perf_counter() - start)
    seconds = sum(durations[warmup:]) / updates
    return {
        "kind": "synthetic_calibration_not_training_evidence",
        "hardware": device_info(device),
        "seconds_per_update": seconds,
        "estimated_hours_per_model_without_io_validation_checkpoints": seconds
        * config.max_steps
        / 3600,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0,
        "effective_batch": config.batch_size * config.accumulation_steps,
        "measured_updates": updates,
        "discarded_warmup_updates": warmup,
        "arch": config.arch,
        "skipped_fp16_attempts": skipped,
    }
