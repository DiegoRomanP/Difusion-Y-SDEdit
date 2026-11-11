"""U-Net compartida, proceso directo, EMA y pesos portables con procedencia."""

import json
import uuid
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from pc3.config import Config
from pc3.io import fingerprint

DEFAULT_ARCH = "anime64"

# Mínimo estructural para aceptar unos pesos sin exigir una arquitectura concreta.
REQUIRED_ARCH_KEYS = frozenset(
    {
        "block_out_channels",
        "down_block_types",
        "up_block_types",
        "in_channels",
        "out_channels",
        "sample_size",
    }
)


def available_architectures() -> list[str]:
    return sorted(
        path.name.removesuffix(".json")
        for path in files("pc3").joinpath("assets/arch").iterdir()
        if path.name.endswith(".json")
    )


def unet_config(name: str = DEFAULT_ARCH) -> dict:
    """Arquitectura con nombre: la elige el perfil, no la fija el código."""
    if name not in available_architectures():
        raise ValueError(
            f"Arquitectura desconocida: {name!r}. Disponibles: {available_architectures()}"
        )
    return json.loads(files("pc3").joinpath(f"assets/arch/{name}.json").read_text())


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


def create_model(config: Config) -> UNet2DModel:
    return UNet2DModel.from_config(unet_config(config.arch))


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
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
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


def load_weights(path: Path, expected_arch: str | None = None):
    """Los pesos traen su arquitectura; `strict=True` ya garantiza las formas.

    `expected_arch` se comprueba solo si el llamante exige una arquitectura
    concreta: así un checkpoint histórico sigue cargándose tras cambiarla.
    """
    metadata = weights_metadata(path)
    stored = metadata.get("model_config")
    if not isinstance(stored, dict) or stored.get("_class_name") != "UNet2DModel":
        raise ValueError("Arquitectura ausente o ajena en los pesos: se esperaba UNet2DModel.")
    missing = REQUIRED_ARCH_KEYS - stored.keys()
    if missing:
        raise ValueError(f"Arquitectura incompleta en los pesos: falta {sorted(missing)}")
    if expected_arch is not None:
        for key, expected in unet_config(expected_arch).items():
            if not key.startswith("_") and fingerprint(stored.get(key)) != fingerprint(expected):
                raise ValueError(f"Los pesos no son de la arquitectura {expected_arch!r}: {key}")
    model = UNet2DModel.from_config(stored)
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model, metadata
