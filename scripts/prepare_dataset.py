"""Descarga Anime Face Dataset y lo reparte en train/val/test (80/10/10).

    uv run python scripts/prepare_dataset.py --download
    uv run python scripts/prepare_dataset.py

Rutas y semilla salen de configs/proyecto.toml. --download baja el dataset a `data_dir`
con KaggleHub (credenciales de Kaggle del equipo); sin él usa lo que ya haya allí.

Reparto: cada imagen se recorta a 64x64; las ilegibles y los duplicados exactos se
descartan; las casi duplicadas (dHash igual, también en espejo) forman un grupo que va
entero a una sola partición, para que test no contenga copias de train. El resultado se
escribe en `splits_file` (data/dataset_splits.json), que está versionado: no se
sobrescribe, porque regenerarlo cambiaría qué imágenes son de test.
"""

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from pc3.config import load_project, write_json
from pc3.data import canonical_image, mirror_dhash

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def create_splits(root: Path, *, image_size: int, seed: int) -> dict:
    root = Path(root)
    records, seen, excluded = [], set(), Counter(corrupt=0, exact_duplicate=0)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        try:
            image = canonical_image(path, image_size)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            excluded["corrupt"] += 1
            continue
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in seen:
            excluded["exact_duplicate"] += 1
            continue
        seen.add(digest)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "group": f"{mirror_dhash(image):016x}",
            }
        )

    groups = defaultdict(list)
    for row in records:
        groups[row["group"]].append(row)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    target = {"train": 0.8 * len(records), "val": 0.1 * len(records), "test": 0.1 * len(records)}
    counts = dict.fromkeys(target, 0)
    for name in names:  # Cada grupo va a la partición más lejos de su objetivo.
        split = max(target, key=lambda key: target[key] - counts[key])
        for row in groups[name]:
            row["split"] = split
        counts[split] += len(groups[name])
    return {
        "schema": 1,
        "image_size": image_size,
        "seed": seed,
        "split_method": "grouped_dhash_mirror_invariant_80_10_10",
        "excluded": dict(excluded),
        "counts": counts,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--download", action="store_true", help="Bajar el dataset con KaggleHub")
    args = parser.parse_args()
    project = load_project()

    if args.download:
        import kagglehub

        data_dir = Path(project.data_dir).resolve()
        if data_dir.exists() and any(data_dir.iterdir()):
            raise SystemExit(f"{data_dir} no está vacío; quitar --download para usarlo.")
        print(
            "Descargado en",
            kagglehub.dataset_download(project.dataset_handle, output_dir=str(data_dir)),
        )

    output = Path(project.splits_file)
    if output.exists():
        raise SystemExit(
            f"{output} ya existe. Borrarlo a mano solo si de verdad hay que rehacer test."
        )
    splits = create_splits(Path(project.data_dir), image_size=project.image_size, seed=project.seed)
    write_json(output, splits)
    print(f"{splits['counts']} | excluidas {splits['excluded']} → {output}")


if __name__ == "__main__":
    main()
