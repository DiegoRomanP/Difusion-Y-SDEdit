"""Busca la configuración de entrenamiento más rápida para esta GPU y escribe el perfil.

    uv run python scripts/calibrate.py --hours 6

Prueba combinaciones que solo cambian velocidad y memoria, nunca lo que aprende el modelo:
  - lote físico 8/16/32, con acumulación para mantener el lote efectivo del perfil base;
  - EMA en GPU o en CPU.
Gradient checkpointing (recalcular activaciones en vez de guardarlas: ~50 % menos VRAM,
~30 % más lento) va apagado. Solo se prueba encendido si nada cabe sin él.
En cada una hace updates reales sobre imágenes aleatorias (no toca el dataset) y mide
segundos por update y VRAM pico. Se queda con la más rápida que deja margen de VRAM,
calcula `max_steps` para que el entrenamiento quepa en --hours y escribe
configs/entrenamiento/<gpu>.toml. LR, warmup y EMA decay se copian de --base: buscarlos
exigiría entrenamientos reales con datos, no unos pasos sintéticos.
Las mediciones quedan en reports/calibration/<gpu>.json.
"""

import argparse
import itertools
import math
import re
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import torch

from pc3.config import ROOT, load_project, load_training, read_json, write_json
from pc3.model import ExponentialMovingAverage, create_model, device_info, get_device

# La corrida anime64_4070ti tardó 5.77 h frente a 5.58 h estimadas: validaciones,
# previews, checkpoints y lectura de datos suman ~3 % sobre los updates puros.
OVERHEAD = 1.03
# No usar más de esta fracción de la VRAM: en Windows, pasarse desborda a RAM
# compartida y el rendimiento se hunde en vez de dar error.
VRAM_LIMIT = 0.85


def measure(project, cfg, *, updates, device):
    """Segundos por update y VRAM pico (GiB) de `cfg`, o None si no cabe."""
    use_amp = cfg.amp and device.type == "cuda"
    torch.manual_seed(project.seed)
    try:
        model = create_model().to(device).train()
        if cfg.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        ema = ExponentialMovingAverage(model, cfg.ema_decay, cfg.ema_device)
        size = project.image_size
        clean = torch.randn(cfg.batch_size, 3, size, size, device=device)
        times = torch.zeros(cfg.batch_size, dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # Los primeros updates pagan el arranque de cuDNN y del asignador.
        warmup, durations = 3, []
        for _ in range(warmup + updates):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.accumulation_steps):
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    predicted = model(clean, times).sample
                loss = predicted.float().square().mean() / cfg.accumulation_steps
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
        return sum(durations[warmup:]) / updates, peak
    except torch.OutOfMemoryError:
        return None
    finally:
        model = optimizer = ema = None
        if device.type == "cuda":
            torch.cuda.empty_cache()


def to_toml(values: dict) -> str:
    def literal(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        return f'"{value}"' if isinstance(value, str) else repr(value)

    return "".join(f"{key} = {literal(value)}\n" for key, value in values.items())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hours", type=float, required=True, help="Horas de entrenamiento")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "configs/entrenamiento/rtx4070ti_12gb.toml",
        help="Perfil del que se copian LR, warmup, EMA y frecuencias",
    )
    parser.add_argument("--output", type=Path, help="Def.: configs/entrenamiento/<gpu>.toml")
    parser.add_argument("--updates", type=int, default=5, help="Updates medidos por combinación")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project = load_project()
    base = load_training(args.base)
    device = get_device(args.device)
    hardware = device_info(device)
    total_vram = hardware.get("total_vram_gib", math.inf)
    effective = base.batch_size * base.accumulation_steps

    trials = []
    batches = [b for b in (8, 16, 32) if effective % b == 0]
    ema_devices = ("cuda", "cpu") if device.type == "cuda" else ("cpu",)
    candidates = []
    for checkpointing in (False, True):
        if checkpointing:
            print("Nada cabe sin gradient checkpointing; probando con él encendido.")
        for batch, ema_device in itertools.product(batches, ema_devices):
            cfg = replace(
                base,
                batch_size=batch,
                accumulation_steps=effective // batch,
                gradient_checkpointing=checkpointing,
                ema_device=ema_device,
            )
            result = measure(project, cfg, updates=args.updates, device=device)
            seconds, peak = result if result else (None, None)
            fits = result is not None and peak <= VRAM_LIMIT * total_vram
            trial = {"cfg": cfg, "seconds_per_update": seconds, "peak_vram_gib": peak}
            trials.append(trial)
            if fits:
                candidates.append(trial)
            shown = f"{seconds:.3f} s/update, {peak:.2f} GiB" if result else "no cabe"
            print(
                f"lote {batch:>2} x{effective // batch:<2} | EMA {ema_device:<4} → "
                f"{shown}{'' if fits or not result else ' (sin margen)'}"
            )
        if candidates:
            break
    if not candidates:
        raise SystemExit("Ninguna combinación cabe en la VRAM de esta GPU.")
    best = min(candidates, key=lambda trial: trial["seconds_per_update"])

    seconds = best["seconds_per_update"]
    max_steps = int(args.hours * 3600 / (seconds * OVERHEAD))
    if max_steps < 10:
        raise SystemExit(f"En {args.hours} h solo caben {max_steps} updates: pedir más horas.")
    chosen = replace(
        best["cfg"], max_steps=max_steps, warmup_steps=min(base.warmup_steps, max_steps // 10)
    )
    train_images = read_json(project.splits_file)["counts"]["train"]
    epochs = max_steps * effective / train_images

    name = re.sub(r"[^a-z0-9]+", "-", hardware["name"].lower())
    name = re.sub(r"^(nvidia-|geforce-)+", "", name).strip("-")
    slug = f"{name}-{round(hardware.get('total_vram_gib', 0))}gb"
    output = args.output or ROOT / "configs/entrenamiento" / f"{slug}.toml"
    header = (
        f"# Generado por scripts/calibrate.py el {date.today()} en {hardware['name']}.\n"
        f"# Objetivo: {args.hours} h → {max_steps} updates (~{epochs:.1f} épocas), "
        f"{seconds:.3f} s/update medidos, {best['peak_vram_gib']:.2f} GiB pico.\n"
        f"# LR, warmup, EMA y frecuencias copiados de {args.base.name}.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + to_toml(asdict(chosen)), encoding="utf-8")
    write_json(
        ROOT / "reports/calibration" / f"{slug}.json",
        {
            "note": "Medición sintética: no es entrenamiento ni evidencia de calidad.",
            "hardware": hardware,
            "hours": args.hours,
            "chosen": asdict(chosen),
            "estimated_epochs": epochs,
            "trials": [
                {
                    **asdict(t["cfg"]),
                    "seconds_per_update": t["seconds_per_update"],
                    "peak_vram_gib": t["peak_vram_gib"],
                }
                for t in trials
            ],
        },
    )
    print(
        f"\nElegido: lote {chosen.batch_size} x{chosen.accumulation_steps}, "
        f"EMA en {chosen.ema_device}, gradient checkpointing "
        f"{'encendido' if chosen.gradient_checkpointing else 'apagado'}.\n"
        f"{max_steps} updates (~{epochs:.1f} épocas) en {args.hours} h → {output}"
    )


if __name__ == "__main__":
    main()
