"""Genera caras anime desde ruido con unos pesos entrenados y mide la inferencia.

    uv run python scripts/generate_images.py --config configs/rtx4070ti_12gb.toml \
        --weights runs/anime64_4070ti/best.safetensors --output reports/samples/anime64_4070ti

Escribe en --output:
  images/000000.png …   una imagen por semilla (sample_seed + índice)
  grid.png              las 16 primeras
  denoising.gif         cómo el ruido se convierte en cara (primera semilla)
  generation.json       muestreador, pasos, segundos por imagen y VRAM de inferencia

Por defecto genera `sample_count` imágenes con `sampling_steps` pasos DDIM; --count y
--steps lo cambian sin editar el perfil.
"""

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from pc3.config import load_config, write_json
from pc3.inference import sample_tensor, save_grid, to_pil
from pc3.model import device_info, get_device, load_weights


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", help="Nombre del modelo en los informes (def.: carpeta del run)")
    parser.add_argument("--count", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    count = args.count or cfg.sample_count
    steps = args.steps or cfg.sampling_steps
    device = get_device(args.device)
    model, weights_metadata = load_weights(args.weights)
    model.to(device)
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    def generate(seeds, capture=False):
        return sample_tensor(
            model,
            seeds=seeds,
            steps=steps,
            sampler=args.sampler,
            device=device,
            amp=cfg.amp,
            capture=capture,
        )

    # Calentamiento fuera de la medición (arranque de CUDA).
    sample_tensor(model, seeds=[cfg.sample_seed], steps=1, device=device, amp=cfg.amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seconds = 0.0  # Solo el muestreo; guardar los PNG no cuenta.
    for start in tqdm(range(0, count, cfg.sample_batch_size), desc="generando"):
        stop = min(count, start + cfg.sample_batch_size)
        begin = time.perf_counter()
        images, _ = generate(list(range(cfg.sample_seed + start, cfg.sample_seed + stop)))
        seconds += time.perf_counter() - begin  # .cpu() al final del muestreo ya sincroniza.
        for index, tensor in zip(range(start, stop), images, strict=True):
            to_pil(tensor).save(image_dir / f"{index:06}.png")
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0

    # El GIF va en una pasada aparte: capturar cuadros copia a CPU y falsearía el tiempo.
    _, frames = generate([cfg.sample_seed], capture=True)
    imageio.mimsave(
        args.output / "denoising.gif", [np.array(f) for f in frames], duration=0.15, loop=0
    )
    save_grid(
        [Image.open(image_dir / f"{i:06}.png").copy() for i in range(min(16, count))],
        args.output / "grid.png",
    )
    result = {
        "label": args.label or args.weights.parent.name,
        "weights": args.weights.as_posix(),
        "weights_metadata": {k: v for k, v in weights_metadata.items() if k != "model_config"},
        "sampler": args.sampler,
        "steps": steps,
        "count": count,
        "seed": cfg.sample_seed,
        "batch_size": cfg.sample_batch_size,
        "amp": cfg.amp,
        "hardware": device_info(device),
        "inference_seconds": seconds,
        "seconds_per_image": seconds / count,
        "images_per_second": count / seconds,
        "peak_vram_gib": peak,
    }
    write_json(args.output / "generation.json", result)
    print(f"{count} imágenes en {seconds:.1f} s ({count / seconds:.1f} img/s) → {args.output}")


if __name__ == "__main__":
    main()
