"""Adaptador PyTorch. Importarlo solo para entrenar/evaluar, no para preparar datos."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from pc3.data import canonical_image, pixel_hash, resolve_image
from pc3.io import file_sha256


class FaceDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], seed: int, augment: bool = False):
        self.root, self.records, self.seed, self.augment = root, records, seed, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        index, position = item if isinstance(item, tuple) else (item, item)
        record = self.records[index]
        image = canonical_image(resolve_image(self.root, record["path"]))
        if pixel_hash(image) != record["sha256"]:
            raise ValueError(f"La imagen cambió desde prepare: {record['path']}")
        if self.augment and random.Random(self.seed ^ position).randrange(2):
            image = ImageOps.mirror(image)
        pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1)
        return torch.from_numpy(pixels) / 127.5 - 1.0


class GeneratedDataset(Dataset):
    def __init__(self, root: Path, records: list[dict]):
        self.root, self.records = root, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = resolve_image(self.root, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Muestra alterada después de sample: {record['path']}")
        with Image.open(path) as image:
            if image.size != (64, 64) or image.mode != "RGB":
                raise ValueError("La muestra debe ser RGB64 sin preprocesamiento adicional.")
            return torch.from_numpy(np.array(image).transpose(2, 0, 1).copy())
