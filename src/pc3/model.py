"""U-Net, programa de ruido, EMA, dispositivo y guardado/carga de pesos."""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# 4 bloques, autoatención a 16x16 y 8x8, ~88 M de parámetros.
UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "block_out_channels": (128, 256, 384, 512),
    "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    "attention_head_dim": 64,
    "norm_num_groups": 32,
}


def create_model() -> UNet2DModel:
    return UNet2DModel(**UNET_CONFIG)


def noise_scheduler() -> DDPMScheduler:
    """1000 pasos, beta lineal 0.0001..0.02, la red predice el ruido."""
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
    """Proceso directo: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon.

    Fuente: https://arxiv.org/abs/2006.11239, ecuación 4.
    """
    alphas = alphas_cumprod.to(clean.device)[times].reshape(-1, 1, 1, 1)
    return alphas.sqrt() * clean + (1 - alphas).sqrt() * noise


class ExponentialMovingAverage:
    """Copia suavizada de los pesos; es la que se guarda y se usa para generar."""

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
        """Usar los pesos EMA dentro del bloque y restaurar los de entrenamiento al salir."""
        original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(original)


def get_device(name: str = "cuda") -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible (¿uv sync --extra cu128?). Usar --device cpu.")
    return device


def device_info(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"device": str(device), "name": "CPU"}
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": properties.name,
        "total_vram_gib": properties.total_memory / 1024**3,
    }


def save_weights(model, path: Path, metadata: dict) -> None:
    """Un .safetensors con los pesos y, en sus metadatos, la config de la U-Net."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, "model_config": dict(model.config)}
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(path),
        metadata={"pc3": json.dumps(metadata)},
    )


def load_weights(path: Path):
    """Reconstruye la U-Net desde la config guardada en el propio archivo."""
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = json.loads(file.metadata()["pc3"])
    model = UNet2DModel.from_config(metadata["model_config"])
    model.load_state_dict(load_file(str(path), device="cpu"))
    return model, metadata
