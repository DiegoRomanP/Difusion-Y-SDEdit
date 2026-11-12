"""Perfil del experimento (TOML en configs/) y lectura/escritura de JSON."""

import json
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    image_size: int = 64
    dataset_handle: str = "splcher/animefacedataset/versions/3"
    data_dir: str = "data/raw"
    splits_file: str = "data/dataset_splits.json"
    runs_dir: str = "runs"
    # Entrenamiento
    batch_size: int = 32
    accumulation_steps: int = 1
    max_steps: int = 60000
    learning_rate: float = 0.0002
    weight_decay: float = 0.000001
    warmup_steps: int = 1000
    ema_decay: float = 0.999
    ema_device: str = "cuda"
    amp: bool = True
    gradient_checkpointing: bool = False
    num_workers: int = 2
    log_every: int = 50
    checkpoint_every: int = 1000
    validate_every: int = 1000
    validation_images: int = 256
    # Generación y evaluación
    sampling_steps: int = 50
    sample_batch_size: int = 32
    sample_count: int = 5000
    sample_seed: int = 12345
    kid_subset_size: int = 1000
    kid_subsets: int = 50


def load_config(path: str | Path) -> Config:
    """Una clave desconocida en el TOML casi siempre es un error de tipeo: se rechaza."""
    with Path(path).open("rb") as file:
        values = tomllib.load(file)
    unknown = values.keys() - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Parámetros desconocidos en {path}: {sorted(unknown)}")
    return Config(**values)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
