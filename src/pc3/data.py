"""Datos portables, particiones por grupos y control de corrupción/duplicados.

SHA-256 identifica píxeles idénticos tras el preprocesamiento. Un dHash de 64 bits,
invariante a reflexión horizontal, agrupa posibles casi duplicados (no es una prueba
exhaustiva de similitud). Las imágenes de un grupo nunca cruzan particiones.
"""

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from pc3.io import read_json, write_json

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_image(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("La imagen debe tener una ruta relativa dentro del dataset.")
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("La ruta o enlace de imagen escapa del dataset.")
    return path


def canonical_image(path: Path, image_size: int = 64) -> Image.Image:
    """Mismo recorte central Lanczos en preparación, entrenamiento y evaluación."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS)


def pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _dhash(image: Image.Image) -> str:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (small.getpixel((x, y)) > small.getpixel((x + 1, y)))
    return f"{bits:016x}"


def split_groups(groups: dict[str, list[dict]], seed: int) -> dict[str, str]:
    if len(groups) < 3:
        raise ValueError("Se necesitan al menos tres grupos distintos para train/val/test.")
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    size = sum(len(group) for group in groups.values())
    target = {"train": 0.8 * size, "val": 0.1 * size, "test": 0.1 * size}
    counts = dict.fromkeys(target, 0)
    result = {}
    for name in names:
        split = max(target, key=lambda key: target[key] - counts[key])
        for row in groups[name]:
            result[row["path"]] = split
        counts[split] += len(groups[name])
    if not all(counts.values()):
        raise ValueError("Un grupo domina el dataset: no se pudieron formar tres particiones.")
    return result


def create_manifest(root: Path, output: Path, *, image_size: int, seed: int) -> dict:
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(
            f"El manifiesto ya existe: {output}. Usar otro archivo para otra división."
        )
    if not root.is_dir():
        raise ValueError(f"No existe el directorio de imágenes: {root}")
    records, seen = [], set()
    excluded = Counter(corrupt=0, exact_duplicate=0, unsafe_path=0)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            path = resolve_image(root, relative)
        except ValueError:
            excluded["unsafe_path"] += 1
            continue
        try:
            image = canonical_image(path, image_size)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            excluded["corrupt"] += 1
            continue
        digest = pixel_hash(image)
        if digest in seen:
            excluded["exact_duplicate"] += 1
            continue
        seen.add(digest)
        records.append(
            {
                "path": relative,
                "sha256": digest,
                "group": min(_dhash(image), _dhash(ImageOps.mirror(image))),
            }
        )
    groups = defaultdict(list)
    for row in records:
        groups[row["group"]].append(row)
    assignment = split_groups(dict(groups), seed)
    for row in records:
        row["split"] = assignment[row["path"]]
    manifest = {
        "schema": 1,
        "image_size": image_size,
        "seed": seed,
        "split_method": "grouped_dhash_mirror_invariant_80_10_10",
        "excluded": dict(excluded),
        "counts": dict(Counter(row["split"] for row in records)),
        "records": records,
    }
    write_json(output, manifest)
    return manifest


def load_manifest(path: str | Path) -> dict:
    manifest = read_json(path)
    if manifest.get("schema") != 1 or manifest.get("image_size") != 64:
        raise ValueError("Manifiesto incompatible: schema=1 e image_size=64 requeridos.")
    group_splits, digest_splits, paths, splits = {}, {}, set(), set()
    for row in manifest["records"]:
        split = row["split"]
        if split not in {"train", "val", "test"} or row["path"] in paths:
            raise ValueError("Partición inválida o ruta repetida en manifiesto.")
        paths.add(row["path"])
        splits.add(split)
        for key, previous in (("group", group_splits), ("sha256", digest_splits)):
            if row[key] in previous and previous[row[key]] != split:
                raise ValueError("Fuga de datos: grupo o imagen compartido entre particiones.")
            previous[row[key]] = split
    if splits != {"train", "val", "test"}:
        raise ValueError("Se requieren train, val y test no vacíos.")
    return manifest


def download_dataset(handle: str, output: Path) -> str:
    """Descarga explícita, versión fijada y credenciales gestionadas por KaggleHub.

    Fuente: https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/datasets.py
    Nunca copiar, imprimir ni guardar el contenido de kaggle.json.
    """
    import kagglehub

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("El destino no está vacío; usar prepare para los datos existentes.")
    return kagglehub.dataset_download(handle, output_dir=str(output.resolve()))
