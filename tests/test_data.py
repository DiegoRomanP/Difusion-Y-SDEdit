import random

import pytest
from PIL import Image, ImageOps

from pc3.data import create_manifest, load_manifest, resolve_image, split_groups


def test_split_keeps_groups_together_and_is_reproducible():
    groups = {f"g{i}": [{"path": f"{i}-{j}.png"} for j in range(3)] for i in range(100)}
    first = split_groups(groups, seed=42)
    assert first == split_groups(groups, seed=42)
    assert first != split_groups(groups, seed=43)
    for group in groups.values():
        assert len({first[row["path"]] for row in group}) == 1
    assert set(first.values()) == {"train", "val", "test"}


def test_image_paths_cannot_escape_dataset_root(tmp_path):
    with pytest.raises(ValueError):
        resolve_image(tmp_path, "../secret.png")
    outside = tmp_path.parent / "outside.png"
    outside.touch()
    (tmp_path / "link.png").symlink_to(outside)
    with pytest.raises(ValueError):
        resolve_image(tmp_path, "link.png")


def test_manifest_excludes_corruption_and_groups_mirrors(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for i in range(30):
        rng = random.Random(i)
        image = Image.frombytes("RGB", (16, 16), rng.randbytes(16 * 16 * 3))
        image.save(root / f"{i:02}.png")
    original = Image.open(root / "00.png")
    original.save(root / "duplicate.png")
    ImageOps.mirror(original).save(root / "mirror.png")
    (root / "broken.png").write_bytes(b"not an image")
    path = tmp_path / "manifest.json"
    manifest = create_manifest(root, path, image_size=64, seed=42)
    assert manifest["excluded"]["corrupt"] == 1
    assert manifest["excluded"]["exact_duplicate"] == 1
    loaded = load_manifest(path)
    assert loaded == manifest
    rows = {row["path"]: row for row in loaded["records"]}
    assert rows["00.png"]["split"] == rows["mirror.png"]["split"]
    with pytest.raises(FileExistsError):
        create_manifest(root, path, image_size=64, seed=42)


def test_manifest_rejects_split_leakage(tmp_path):
    from pc3.io import write_json

    path = tmp_path / "bad.json"
    write_json(
        path,
        {
            "schema": 1,
            "image_size": 64,
            "records": [
                {"path": "a.png", "split": "train", "group": "same", "sha256": "a"},
                {"path": "b.png", "split": "test", "group": "same", "sha256": "b"},
                {"path": "c.png", "split": "val", "group": "other", "sha256": "c"},
            ],
        },
    )
    with pytest.raises(ValueError, match="grupo"):
        load_manifest(path)
