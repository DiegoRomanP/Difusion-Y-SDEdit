"""Mide cuánto tarda un update de entrenamiento y cuánta VRAM usa, sin tocar el dataset.

    uv run python scripts/calibrate.py --config configs/rtx4070ti_12gb.toml --updates 20

Hace updates reales (modelo, optimizador, EMA y AMP del perfil) sobre imágenes aleatorias
y proyecta las horas para `max_steps`. La proyección no incluye lectura de datos,
validación ni checkpoints, así que hay que reservar margen. Escribe
reports/calibration/<gpu>.json.
"""

import argparse
import re
import time
from pathlib import Path

import torch

from pc3.config import load_config, write_json
from pc3.model import ExponentialMovingAverage, create_model, device_info, get_device


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20, help="Updates medidos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    use_amp = cfg.amp and device.type == "cuda"
    torch.manual_seed(cfg.seed)
    model = create_model().to(device).train()
    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ExponentialMovingAverage(model, cfg.ema_decay, cfg.ema_device)
    clean = torch.randn(cfg.batch_size, 3, cfg.image_size, cfg.image_size, device=device)
    times = torch.zeros(cfg.batch_size, dtype=torch.long, device=device)

    # Los primeros updates pagan el arranque de CUDA/cuDNN y el crecimiento del asignador.
    # Con un solo update de calentamiento una medición real salió 8.2x más lenta.
    warmup = max(3, args.updates // 5)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(warmup + args.updates):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.accumulation_steps):
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                predicted = model(clean, times).sample
            scaler.scale(predicted.float().square().mean() / cfg.accumulation_steps).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)

    seconds = sum(durations[warmup:]) / args.updates
    hardware = device_info(device)
    result = {
        "note": "Medición sintética: no es entrenamiento ni evidencia de calidad.",
        "hardware": hardware,
        "seconds_per_update": seconds,
        "estimated_hours_for_max_steps": seconds * cfg.max_steps / 3600,
        "max_steps": cfg.max_steps,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
        ),
        "effective_batch": cfg.batch_size * cfg.accumulation_steps,
        "measured_updates": args.updates,
        "discarded_warmup_updates": warmup,
    }
    if args.output is None:
        name = re.sub(r"[^a-z0-9]+", "-", hardware["name"].lower())
        name = re.sub(r"^(nvidia-|geforce-)+", "", name).strip("-")
        vram = round(hardware.get("total_vram_gib", 0))
        args.output = Path("reports/calibration") / f"{name}-{vram}gb.json"
    write_json(args.output, result)
    print(
        f"{seconds:.3f} s/update, {result['peak_vram_gib']:.2f} GiB pico, "
        f"~{result['estimated_hours_for_max_steps']:.1f} h para {cfg.max_steps} updates "
        f"→ {args.output}"
    )


if __name__ == "__main__":
    main()
