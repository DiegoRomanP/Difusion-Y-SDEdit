"""U-Net compartida, proceso directo, EMA y pesos portables con procedencia."""

import json
import os
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from pc3.config import Config
from pc3.io import fingerprint


def unet_config() -> dict:
    return json.loads(files("pc3").joinpath("assets/unet_config.json").read_text())


def noise_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        variance_type="fixed_small",
        prediction_type="epsilon",
        clip_sample=True,
    )


def q_sample(clean, times, noise, alphas_cumprod):
    """x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon.

    Fuente: https://arxiv.org/abs/2006.11239, ecuación 4.
    La prueba contrasta esta implementación educativa con DDPMScheduler.add_noise.
    """
    if clean.shape != noise.shape or times.shape != (clean.shape[0],):
        raise ValueError("Formas incompatibles de imagen, ruido o tiempos.")
    alphas = alphas_cumprod.to(clean.device)[times].reshape(-1, 1, 1, 1)
    return alphas.sqrt() * clean + (1 - alphas).sqrt() * noise


def create_model(config: Config, pretrained: bool = False) -> UNet2DModel:
    if pretrained:
        # Solo state_dict del SHA fijado, sin código remoto. Diffusers 0.42.1 usa
        # torch.load(weights_only=True) para el .bin antiguo; no activar pickle libre.
        # https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/model_loading_utils.py
        model, loading = UNet2DModel.from_pretrained(
            config.pretrained_id,
            subfolder="unet",
            revision=config.pretrained_revision,
            use_safetensors=False,
            low_cpu_mem_usage=False,
            output_loading_info=True,
        )
        if any(
            loading.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(
                "La conversión del checkpoint antiguo no fue estricta; "
                "no se aceptarán parámetros inicializados silenciosamente al azar."
            )
    else:
        model = UNet2DModel.from_config(unet_config())
    for key, expected in unet_config().items():
        if not key.startswith("_") and fingerprint(model.config[key]) != fingerprint(expected):
            raise ValueError(f"El checkpoint no coincide con la arquitectura compartida: {key}")
    return model


class ExponentialMovingAverage:
    """EMA en CPU por defecto: ahorra VRAM a cambio de transferencias por update."""

    def __init__(self, model, decay: float, device: str):
        self.decay = decay
        self.shadow = {k: v.detach().to(device).clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            value = value.detach().to(self.shadow[key].device)
            if value.is_floating_point():
                self.shadow[key].lerp_(value, 1 - self.decay)
            else:
                self.shadow[key].copy_(value)

    @contextmanager
    def apply(self, model):
        original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(original)


def export_weights(model, path: Path, provenance: dict) -> None:
    """Pesos y metadatos en un único safetensors reemplazado atómicamente."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    metadata = {**provenance, "model_config": dict(model.config)}
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(temporary),
        metadata={"pc3": json.dumps(metadata, allow_nan=False)},
    )
    temporary.replace(path)


def weights_metadata(path: Path) -> dict:
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        if "pc3" not in metadata:
            raise ValueError("El archivo no contiene procedencia PC3.")
        return json.loads(metadata["pc3"])


def load_weights(path: Path):
    metadata = weights_metadata(path)
    for key, expected in unet_config().items():
        if not key.startswith("_") and fingerprint(
            metadata["model_config"].get(key)
        ) != fingerprint(expected):
            raise ValueError(f"Arquitectura PC3 incompatible: {key}")
    model = UNet2DModel.from_config(metadata["model_config"])
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model, metadata
