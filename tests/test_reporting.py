import random
from dataclasses import replace

import pytest
from PIL import Image

from pc3.config import Config, training_contract
from pc3.data import create_manifest
from pc3.diagnostics import diagnose
from pc3.io import file_sha256, write_json
from pc3.reporting import report, report_ablation


def synthetic_metrics(label, initialization):
    config = Config()
    return {
        "label": label,
        "protocol": {"num_images": 100, "split": "test", "manifest_sha256": "fixture"},
        "fid": 12.0,
        "kid_mean": 0.02,
        "kid_std": 0.001,
        "seconds_per_image": 0.1,
        "images_per_second": 10.0,
        "peak_vram_gib": 1.0,
        "generation": {
            "hardware": {"name": "synthetic-fixture"},
            "count": 100,
            "source_hash": "same",
        },
        "provenance": {
            "contract": training_contract(config),
            "initialization": initialization,
            "step": 7,
        },
        "run_summary": {
            "status": "completed",
            "completed_steps": config.max_steps,
            "consumed_images": config.max_steps * 16,
            "wall_seconds": 60.0,
        },
        "warnings": ["Datos sintéticos exclusivos de pruebas, no evidencias del modelo."],
    }


def test_render_report_from_explicit_synthetic_fixtures_only(tmp_path):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    write_json(paths[0], synthetic_metrics("synthetic-A", "random"))
    write_json(paths[1], synthetic_metrics("synthetic-C", "pretrained"))
    output = tmp_path / "figures"
    report(paths, output=output)
    assert (output / "comparison_quality.png").is_file()
    assert (output / "comparison_cost.png").is_file()
    assert (output / "comparison.csv").is_file()
    assert "no establece un ganador universal" in (output / "results.md").read_text()


def test_ablation_rejects_test_even_if_other_settings_match(tmp_path):
    path = tmp_path / "test_metrics.json"
    write_json(path, synthetic_metrics("fixture", "random"))
    with pytest.raises(ValueError, match="val"):
        report_ablation([path, path], output=tmp_path / "not_created")


def test_diagnostic_detects_exact_training_neighbor_in_synthetic_fixture(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    for index in range(20):
        pixels = random.Random(index).randbytes(64 * 64 * 3)
        Image.frombytes("RGB", (64, 64), pixels).save(root / f"{index:02}.png")
    manifest_path = tmp_path / "manifest.json"
    manifest = create_manifest(root, manifest_path, image_size=64, seed=Config().seed)
    samples = tmp_path / "samples"
    samples.mkdir()
    records = []
    train = [row for row in manifest["records"] if row["split"] == "train"][:8]
    for index, row in enumerate(train):
        path = samples / f"{index}.png"
        path.write_bytes((root / row["path"]).read_bytes())
        records.append({"path": path.name, "sha256": file_sha256(path)})
    write_json(
        samples / "generation.json",
        {
            "complete": True,
            "label": "synthetic-unit-test",
            "count": len(records),
            "manifest_sha256": file_sha256(manifest_path),
            "source_hash": "synthetic",
            "images": records,
        },
    )
    config = replace(Config(), data_dir=str(root), manifest=str(manifest_path))
    result = diagnose(config, samples=samples, limit=4)
    assert all(row["hamming"] == 0 for row in result["nearest_training_dhash"])
    assert len(result["pairwise_ssim"]) == 4
    assert (samples / "nearest_training.png").is_file()
