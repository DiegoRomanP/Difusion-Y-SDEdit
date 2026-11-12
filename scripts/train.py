"""Entrena el DDPM de rostros anime.

    P=configs/entrenamiento/rtx4070ti_12gb.toml
    uv run python scripts/train.py --profile $P --run mi_corrida
    uv run python scripts/train.py --profile $P --run mi_corrida --resume

--profile trae lo que depende de la GPU (lote, updates, memoria); datos y semilla salen
de configs/proyecto.toml.

Escribe en runs/<run>/:
  latest.pt          estado completo para reanudar (modelo, optimizador, EMA, azar)
  best.safetensors   pesos EMA con la menor MSE de validación
  last.safetensors   pesos EMA al terminar o pausar
  history.csv        MSE de train/val, LR y tiempos
  previews/          16 caras fijas (mismas semillas) en cada validación
  run.json           resumen: perfil, estado, updates, mejor MSE, horas

Ctrl+C termina el update en curso, guarda y sale; --resume sigue con el mismo orden de
datos y el mismo azar. Fuentes: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
https://huggingface.co/docs/diffusers/training/unconditional_training
"""

import argparse
import csv
import math
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from pc3.config import load_project, load_training, write_json
from pc3.data import canonical_image, load_splits, split_records
from pc3.inference import sample_tensor, save_grid, to_pil, to_tensor
from pc3.model import (
    ExponentialMovingAverage,
    create_model,
    device_info,
    get_device,
    noise_scheduler,
    q_sample,
    save_weights,
)

PREVIEW_SEEDS = list(range(2026, 2042))
PREVIEW_STEPS = 20


class FaceDataset(Dataset):
    """Recibe (índice, posición); el volteo horizontal depende de la posición, no de un RNG."""

    def __init__(self, root, records, seed, augment=False):
        self.root, self.records, self.seed, self.augment = Path(root), records, seed, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        index, position = item if isinstance(item, tuple) else (item, item)
        image = canonical_image(self.root / self.records[index]["path"])
        if self.augment and random.Random(self.seed ^ position).randrange(2):
            image = ImageOps.mirror(image)
        return to_tensor(image)


class ReplayBatchSampler:
    """Orden de datos reanudable: cada época se baraja con seed + época.

    `start` = microlotes ya consumidos, así que reanudar repite exactamente el orden.
    """

    def __init__(self, size, batch_size, seed, start=0):
        self.size, self.batch_size, self.seed, self.start = size, batch_size, seed, start

    def __iter__(self):
        position = self.start * self.batch_size
        cached_epoch, permutation = -1, []
        while True:
            batch = []
            for _ in range(self.batch_size):
                epoch, offset = divmod(position, self.size)
                if epoch != cached_epoch:
                    permutation = list(range(self.size))
                    random.Random(self.seed + epoch).shuffle(permutation)
                    cached_epoch = epoch
                batch.append((permutation[offset], position))
                position += 1
            yield batch


def learning_rate_at(step, cfg) -> float:
    """Warmup lineal y luego coseno hasta 0."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps))
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def validation_loss(model, loader, seed, device, alphas, use_amp) -> float:
    """Mismo ruido y mismo t por imagen en cada validación, para que las curvas sean comparables."""
    model.eval()
    total, count = 0.0, 0
    for clean in loader:
        noises, times = [], []
        for index in range(count, count + len(clean)):
            rng = torch.Generator().manual_seed(seed + 1_000_000 + index)
            times.append(torch.randint(1000, (1,), generator=rng))
            noises.append(torch.randn((3, 64, 64), generator=rng))
        clean = clean.to(device)
        noise, times = torch.stack(noises).to(device), torch.cat(times).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(q_sample(clean, times, noise, alphas), times).sample
        total += F.mse_loss(prediction.float(), noise).item() * len(clean)
        count += len(clean)
    model.train()
    return total / count


def write_history(path, rows):
    columns = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile", type=Path, required=True, help="configs/entrenamiento/<gpu>.toml"
    )
    parser.add_argument("--run", required=True, help="Nombre de la carpeta dentro de runs_dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project = load_project()
    cfg = load_training(args.profile)
    device = get_device(args.device)
    use_amp = cfg.amp and device.type == "cuda"
    run_dir = Path(project.runs_dir) / args.run
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} ya existe: usar --resume u otro --run.")
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(project.splits_file)
    train_records = split_records(splits, "train")
    val_records = split_records(splits, "val")
    random.Random(project.seed).shuffle(val_records)
    val_records = val_records[: cfg.validation_images]

    # Reduce variabilidad; no promete resultados idénticos bit a bit entre GPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(project.seed)
    model = create_model().to(device).train()
    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ExponentialMovingAverage(model, cfg.ema_decay, cfg.ema_device)
    noise_rng = torch.Generator(device=device).manual_seed(project.seed)
    torch.manual_seed(project.seed + 7)
    step, microstep, best, wall_before, skipped, history = 0, 0, math.inf, 0.0, 0, []

    if args.resume:
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        ema.shadow = {k: v.to(cfg.ema_device) for k, v in state["ema"].items()}
        step, microstep, best = state["step"], state["microstep"], state["best_val"]
        wall_before, skipped, history = state["wall_seconds"], state["skipped"], state["history"]
        noise_rng.set_state(state["noise_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        print(f"Reanudando {run_dir} desde el update {step}.")

    loader_options = {
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
        # Generador propio: reconstruir el DataLoader al reanudar no consume el azar global.
        "generator": torch.Generator().manual_seed(project.seed + 5),
    }
    sampler = ReplayBatchSampler(len(train_records), cfg.batch_size, project.seed, start=microstep)
    batches = iter(
        DataLoader(
            FaceDataset(project.data_dir, train_records, project.seed, augment=True),
            batch_sampler=sampler,
            **loader_options,
        )
    )
    val_loader = DataLoader(
        FaceDataset(project.data_dir, val_records, project.seed),
        batch_size=cfg.batch_size,
        **loader_options,
    )
    alphas = noise_scheduler().alphas_cumprod.to(device)
    started = time.perf_counter()
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        tqdm.write("Parada solicitada: se guarda al terminar el update actual.")

    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}

    def wall_seconds():
        return wall_before + time.perf_counter() - started

    def weights_metadata(val_mse):
        return {"run": args.run, "step": step, "weights": "ema", "validation_mse": val_mse}

    def validate_and_preview(row):
        nonlocal best
        with ema.apply(model):
            row["val_mse"] = validation_loss(
                model, val_loader, project.seed, device, alphas, use_amp
            )
            if row["val_mse"] < best:
                best = row["val_mse"]
                save_weights(model, run_dir / "best.safetensors", weights_metadata(best))
            images = []
            for start in range(0, len(PREVIEW_SEEDS), project.sample_batch_size):
                seeds = PREVIEW_SEEDS[start : start + project.sample_batch_size]
                batch, _ = sample_tensor(
                    model, seeds=seeds, steps=PREVIEW_STEPS, device=device, amp=cfg.amp
                )
                images.extend(to_pil(image) for image in batch)
            save_grid(images, run_dir / "previews" / f"step_{step:06}.png")

    def save_checkpoint(status):
        torch.save(
            {
                "step": step,
                "microstep": microstep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "noise_rng": noise_rng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "best_val": best,
                "wall_seconds": wall_seconds(),
                "skipped": skipped,
                "history": history,
            },
            run_dir / "latest.pt",
        )
        write_history(run_dir / "history.csv", history)
        write_json(
            run_dir / "run.json",
            {
                "status": status,
                "profile": args.profile.as_posix(),
                "training": asdict(cfg),
                "project": asdict(project),
                "hardware": device_info(device),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "completed_steps": step,
                "consumed_images": microstep * cfg.batch_size,
                "best_validation_mse": best,
                "skipped_fp16_updates": skipped,
                "wall_seconds": wall_seconds(),
            },
        )

    try:
        if not args.resume:
            initial = {"step": 0}
            validate_and_preview(initial)
            history.append(initial)
            save_checkpoint("training")
        progress = tqdm(total=cfg.max_steps, initial=step, desc=args.run)
        while step < cfg.max_steps and not stop_requested:
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            train_mse = 0.0
            for _ in range(cfg.accumulation_steps):
                clean = next(batches).to(device, non_blocking=True)
                noise = torch.randn(clean.shape, generator=noise_rng, device=device)
                times = torch.randint(1000, (len(clean),), generator=noise_rng, device=device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    prediction = model(q_sample(clean, times, noise, alphas), times).sample
                loss = F.mse_loss(prediction.float(), noise) / cfg.accumulation_steps
                scaler.scale(loss).backward()
                train_mse += loss.item()
            rate = learning_rate_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = rate
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:  # FP16 desbordó: GradScaler omitió este update.
                skipped += 1
            ema.update(model)
            step += 1
            microstep += cfg.accumulation_steps
            progress.update(1)

            finished = step == cfg.max_steps or stop_requested
            validate = step % cfg.validate_every == 0 or finished
            row = {
                "step": step,
                "train_mse": train_mse,
                "learning_rate": rate,
                "update_seconds": time.perf_counter() - begin,
                "epoch": microstep * cfg.batch_size / len(train_records),
                "wall_seconds": wall_seconds(),
            }
            if validate:
                validate_and_preview(row)
            if step % cfg.log_every == 0 or validate:
                history.append(row)
                progress.set_postfix(mse=f"{train_mse:.4f}")
                val = f" | val {row['val_mse']:.5f}" if "val_mse" in row else ""
                tqdm.write(
                    f"update {step}/{cfg.max_steps} | mse {train_mse:.4f}{val} | "
                    f"lr {rate:.2e} | época {row['epoch']:.2f}"
                )
            if step % cfg.checkpoint_every == 0 or validate:
                save_checkpoint("training")
        progress.close()
        status = "completed" if step == cfg.max_steps else "paused"
        with ema.apply(model):
            save_weights(model, run_dir / "last.safetensors", weights_metadata(None))
        save_checkpoint(status)
        print(f"{status}: {step} updates, mejor MSE de validación {best:.5f} → {run_dir}")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
