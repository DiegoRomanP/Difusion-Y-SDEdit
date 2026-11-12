"""Imágenes del dataset: recorte común, huella de casi duplicados y particiones.

El archivo de particiones (`splits_file`, por defecto data/dataset_splits.json) lo crea
scripts/prepare_dataset.py. Asigna cada imagen a train, val o test. Está versionado:
regenerarlo podría pasar a train imágenes que antes estaban en test.
"""

from pathlib import Path

from PIL import Image, ImageOps

from pc3.config import read_json


def canonical_image(path: Path, image_size: int = 64) -> Image.Image:
    """Mismo recorte central Lanczos al preparar, entrenar, evaluar y en SDEdit."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS)


def mirror_dhash(image: Image.Image) -> int:
    """dHash de 64 bits, igual para una imagen y su reflejo horizontal.

    Dos imágenes con pocos bits distintos son probables casi duplicados.
    """

    def dhash(picture):
        small = picture.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | (small.getpixel((x, y)) > small.getpixel((x + 1, y)))
        return bits

    return min(dhash(image), dhash(ImageOps.mirror(image)))


def load_splits(path: str | Path) -> dict:
    """Lee las particiones y comprueba que ningún grupo ni imagen esté en dos a la vez."""
    splits = read_json(path)
    seen = {}
    for row in splits["records"]:
        for key in (row["group"], row["sha256"]):
            if seen.setdefault(key, row["split"]) != row["split"]:
                raise ValueError(f"Fuga de datos en {path}: {row['path']} cruza particiones.")
    return splits


def split_records(splits: dict, name: str) -> list[dict]:
    return [row for row in splits["records"] if row["split"] == name]
