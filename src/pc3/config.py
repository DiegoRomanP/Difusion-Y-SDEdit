"""Configuración en dos niveles y lectura/escritura de JSON.

- configs/proyecto.toml: datos, generación y evaluación. Un solo archivo fijo; todos los
  scripts lo leen solos.
- configs/entrenamiento/<gpu>.toml: cómo entrenar en una GPU concreta (lote, updates,
  memoria). Solo lo piden train.py y calibrate.py, con --profile.
"""

import json
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_FILE = ROOT / "configs" / "proyecto.toml"
PATH_FIELDS = {"data_dir", "splits_file", "runs_dir"}


@dataclass(frozen=True)
class Project:
    seed: int
    image_size: int
    dataset_handle: str
    data_dir: Path
    splits_file: Path
    runs_dir: Path
    sampling_steps: int
    sample_batch_size: int
    sample_count: int
    sample_seed: int
    kid_subset_size: int
    kid_subsets: int


@dataclass(frozen=True)
class Training:
    batch_size: int
    accumulation_steps: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    ema_decay: float
    ema_device: str
    amp: bool
    gradient_checkpointing: bool
    num_workers: int
    log_every: int
    checkpoint_every: int
    validate_every: int
    validation_images: int


def _load(cls, path: Path):
    """Una clave desconocida o ausente casi siempre es un error de tipeo: se rechaza."""
    with Path(path).open("rb") as file:
        values = tomllib.load(file)
    expected = {field.name for field in fields(cls)}
    if values.keys() != expected:
        raise ValueError(
            f"{path}: sobran {sorted(values.keys() - expected)}, "
            f"faltan {sorted(expected - values.keys())}"
        )
    return cls(**{k: ROOT / v if k in PATH_FIELDS else v for k, v in values.items()})


def load_project() -> Project:
    return _load(Project, PROJECT_FILE)


def load_training(path: str | Path) -> Training:
    return _load(Training, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    path.write_text(text + "\n", encoding="utf-8")
