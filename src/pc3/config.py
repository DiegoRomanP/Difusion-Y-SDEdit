"""Configuración única para notebook/CLI; validación sin descargar ni cargar modelos."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    image_size: int = 64
    arch: str = "anime64"
    dataset_handle: str = "splcher/animefacedataset/versions/3"
    data_dir: str = "data/raw"
    manifest: str = "data/manifest.json"
    runs_dir: str = "runs"
    batch_size: int = 1
    accumulation_steps: int = 16
    max_steps: int = 30000
    learning_rate: float = 0.0001
    weight_decay: float = 0.000001
    warmup_steps: int = 500
    ema_decay: float = 0.999
    ema_device: str = "cpu"
    amp: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 2
    log_every: int = 50
    checkpoint_every: int = 500
    validate_every: int = 500
    validation_images: int = 256
    sampling_steps: int = 50
    sample_batch_size: int = 4
    sample_count: int = 5000
    sample_seed: int = 12345
    kid_subset_size: int = 1000
    kid_subsets: int = 50

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            expected = type(field.default)
            if expected is float:
                valid = type(value) in (int, float) and math.isfinite(value)
            else:
                valid = type(value) is expected
            if not valid:
                raise ValueError(f"Tipo/valor inválido para {field.name}: {value!r}")
            if expected is int and (
                value < 0
                or (
                    value == 0
                    and field.name not in {"seed", "num_workers", "warmup_steps", "sample_seed"}
                )
            ):
                raise ValueError(f"{field.name} debe ser positivo (o cero si corresponde).")
        if not files("pc3").joinpath(f"assets/arch/{self.arch}.json").is_file():
            raise ValueError(f"Arquitectura desconocida: {self.arch!r}")
        if self.image_size < 32 or self.image_size % 8:
            raise ValueError("image_size debe ser >=32 y múltiplo de 8.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate debe ser >0 y weight_decay >=0.")
        if not 0 < self.ema_decay < 1 or self.ema_device not in {"cpu", "cuda"}:
            raise ValueError("EMA requiere 0<ema_decay<1 y ema_device cpu/cuda.")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps debe ser menor que max_steps.")
        if self.sampling_steps > 1000 or self.sample_count < 2:
            raise ValueError("sampling_steps <=1000 y sample_count >=2.")
        if self.kid_subset_size < 2:
            raise ValueError("kid_subset_size debe ser >=2.")
        for name in ("data_dir", "manifest", "runs_dir", "dataset_handle"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} no puede estar vacío.")


def load_config(path: str | Path) -> Config:
    """Rechaza errores tipográficos en vez de ignorar parámetros del experimento."""
    with Path(path).open("rb") as file:
        values = tomllib.load(file)
    unknown = values.keys() - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Parámetros desconocidos: {sorted(unknown)}")
    config = Config(**values)
    config.validate()
    return config


def training_contract(config: Config) -> dict:
    """Contrato comparable: se excluyen ubicación, salida y opciones solo de inferencia."""
    excluded = {
        "data_dir",
        "manifest",
        "runs_dir",
        "sampling_steps",
        "sample_batch_size",
        "sample_count",
        "sample_seed",
        "kid_subset_size",
        "kid_subsets",
    }
    return {key: value for key, value in asdict(config).items() if key not in excluded}
