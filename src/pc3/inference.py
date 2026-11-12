"""Inferencia: generar caras desde ruido y SDEdit (partir de una foto en vez de ruido).

Ambas recorren el mismo bucle de eliminación de ruido; solo cambia el punto de partida.
Fuente DDIM: https://huggingface.co/docs/diffusers/api/schedulers/ddim
"""

import math
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from pc3.model import noise_scheduler


def _scheduler(sampler: str, steps: int, device):
    scheduler = noise_scheduler()
    if sampler == "ddim":
        scheduler = DDIMScheduler.from_config(scheduler.config)
    scheduler.set_timesteps(steps, device=device)
    return scheduler


@torch.inference_mode()
def _denoise(model, image, scheduler, timesteps, *, sampler, seeds, amp, capture=False):
    """Bucle común. DDIM (eta=0) es determinista; DDPM usa un generador por imagen."""
    generators = [torch.Generator().manual_seed(seed + 10_000_000) for seed in seeds]
    marks = set(np.linspace(0, len(timesteps) - 1, min(12, len(timesteps)), dtype=int).tolist())
    frames = [to_pil(image[0])] if capture else []
    was_training = model.training
    model.eval()
    try:
        for index, timestep in enumerate(timesteps):
            with torch.autocast(
                image.device.type, dtype=torch.float16, enabled=amp and image.device.type == "cuda"
            ):
                prediction = model(image, timestep).sample
            kwargs = {"eta": 0.0} if sampler == "ddim" else {"generator": generators}
            image = scheduler.step(prediction.float(), timestep, image, **kwargs).prev_sample
            if capture and index in marks:
                frames.append(to_pil(image[0]))
    finally:
        model.train(was_training)
    return image.detach().cpu(), frames


def sample_tensor(model, *, seeds, steps=50, sampler="ddim", device, amp=True, capture=False):
    """Genera una cara por semilla. La misma semilla da la misma cara aunque cambie el lote.

    Con capture=True devuelve además ~12 cuadros de la primera imagen (para el GIF).
    """
    scheduler = _scheduler(sampler, steps, device)
    noise = [torch.randn((3, 64, 64), generator=torch.Generator().manual_seed(s)) for s in seeds]
    image = torch.stack(noise).to(device)
    return _denoise(
        model,
        image,
        scheduler,
        scheduler.timesteps,
        sampler=sampler,
        seeds=seeds,
        amp=amp,
        capture=capture,
    )


def sdedit(model, photo, *, strength, steps=50, sampler="ddim", seed=12345, device, amp=True):
    """Difumina `photo` (1,3,64,64 en [-1,1]) hasta un paso intermedio y sigue desde ahí.

    strength=0 casi no cambia la foto; strength=1 recorre todo el programa de ruido, como
    una generación libre. No es traducción controlada: el modelo nunca vio fotos reales.
    """
    scheduler = _scheduler(sampler, steps, device)
    first = steps - max(1, min(steps, round(steps * strength)))
    timesteps = scheduler.timesteps[first:]
    noise = torch.randn(photo.shape, generator=torch.Generator().manual_seed(seed)).to(device)
    image = scheduler.add_noise(photo.to(device), noise, timesteps[:1])
    result, _ = _denoise(model, image, scheduler, timesteps, sampler=sampler, seeds=[seed], amp=amp)
    return result


def to_pil(tensor) -> Image.Image:
    """Tensor (3,64,64) en [-1,1] → imagen RGB."""
    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round()
    return Image.fromarray(array.to(torch.uint8).permute(1, 2, 0).numpy())


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Imagen RGB → tensor (3,H,W) en [-1,1]."""
    pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1)
    return torch.from_numpy(pixels) / 127.5 - 1.0


def save_grid(images, path: Path, columns=4):
    images = list(images)
    grid = Image.new("RGB", (columns * 64, math.ceil(len(images) / columns) * 64))
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * 64, (index // columns) * 64))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
