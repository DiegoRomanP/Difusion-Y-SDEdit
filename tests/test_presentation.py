import json
import runpy
from pathlib import Path

import pytest
from pptx import Presentation
from pptx.util import Inches

from pc3.config import Config, training_contract
from pc3.io import file_sha256, source_fingerprint


@pytest.fixture
def slides():
    return runpy.run_path("tools/build_presentation.py")


def test_talk_has_seven_slides_and_thirty_seconds_of_safety(slides):
    plan = slides["talk_plan"]([])
    assert len(plan) == 7
    assert sum(item["seconds"] for item in plan) == 330
    assert all(item["speech"].strip() for item in plan)


def test_no_results_means_no_numbers_or_claimed_winner(slides, tmp_path):
    assert slides["load_evidence"](tmp_path, Config()) == []
    output = tmp_path / "deliverable"
    with pytest.raises(FileNotFoundError, match="resultados"):
        slides["build"](Path.cwd(), output, require_results=True)
    assert not output.exists()


def test_malformed_results_fail_instead_of_becoming_a_pending_deck(slides, tmp_path):
    report = tmp_path / "reports/comparison/comparison.json"
    report.parent.mkdir(parents=True)
    report.write_text('[{"fid": 12}]')
    with pytest.raises(ValueError, match="evidencias"):
        slides["load_evidence"](tmp_path, Config())


def test_editable_deck_has_notes_hidden_appendices_and_no_off_canvas_shapes(slides, tmp_path):
    slides["build"](Path.cwd(), tmp_path)
    deck = Presentation(tmp_path / "pc3_difusion_anime.pptx")
    assert len(deck.slides) == 9
    assert [slide._element.get("show", "1") for slide in deck.slides] == ["1"] * 7 + ["0"] * 2
    all_text = []
    for slide in deck.slides:
        assert slide.notes_slide.notes_text_frame.text.strip()
        assert slide.shapes.title is not None
        assert slide.shapes.title.left >= Inches(0.5)
        for shape in slide.shapes:
            assert 0 <= shape.left <= deck.slide_width - shape.width
            assert 0 <= shape.top <= deck.slide_height - shape.height
            if shape.has_text_frame:
                all_text.append(shape.text)
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        assert run.font.size is None or run.font.size.pt >= 10
    assert "pendiente" in " ".join(all_text).lower()
    assert "Sin muestras" in " ".join(all_text)
    manifest = json.loads((tmp_path / "estado.json").read_text())
    assert manifest["status"] == "pending_results"
    assert manifest["planned_seconds"] == 330
    assert "0:25" in (tmp_path / "guion.md").read_text()


def synthetic_evidence(root):
    from PIL import Image

    protocol = {
        "split": "test",
        "num_images": 4,
        "sampler": "ddim",
        "sampling_steps": 50,
        "manifest_sha256": "fixture",
        "pc3_source_sha256": source_fingerprint(),
    }
    rows = []
    for initialization, folder in [
        ("random", "from_scratch"),
        ("pretrained_reference", "pretrained"),
        ("pretrained", "finetuned"),
    ]:
        directory = root / "reports/samples" / folder
        directory.mkdir(parents=True)
        records = []
        for index in range(4):
            path = directory / f"{index:06}.png"
            Image.new("RGB", (64, 64), (index * 32, 0, 0)).save(path)
            records.append({"path": path.name, "sha256": file_sha256(path)})
        provenance = {
            "initialization": initialization,
            "contract": training_contract(Config()),
            "step": 0,
            "weights": "ema",
            "manifest_sha256": "fixture",
        }
        generation = {
            "complete": True,
            "source_hash": folder,
            "count": 4,
            "images": records,
            "manifest_sha256": "fixture",
            "provenance": provenance,
            "pc3_source_sha256": protocol["pc3_source_sha256"],
        }
        (directory / "generation.json").write_text(json.dumps(generation))
        rows.append(
            {
                "label": folder,
                "protocol": protocol,
                "fid": 12.0,
                "kid_mean": 0.01,
                "kid_std": 0.001,
                "seconds_per_image": 0.1,
                "provenance": provenance,
                "generation": {
                    "hardware": {"name": "synthetic-fixture"},
                    "count": 4,
                    "source_hash": folder,
                },
                "run_summary": {
                    "status": "completed",
                    "completed_steps": 30000,
                    "consumed_images": 480000,
                },
            }
        )
    report = root / "reports/comparison/comparison.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(rows))
    return report


def test_only_consistent_results_and_verified_images_are_accepted(slides, tmp_path):
    report = synthetic_evidence(tmp_path)
    evidence = slides["load_evidence"](tmp_path, Config())
    assert len(evidence) == 3
    assert all(len(item["images"]) == 4 for item in evidence)
    image = tmp_path / "reports/samples/finetuned/000000.png"
    image.write_bytes(b"altered")
    with pytest.raises(ValueError, match="muestra"):
        slides["load_evidence"](tmp_path, Config())
    assert report.is_file()


def test_validation_metrics_cannot_be_presented_as_final_test_results(slides, tmp_path):
    report = synthetic_evidence(tmp_path)
    rows = json.loads(report.read_text())
    for row in rows:
        row["protocol"]["split"] = "val"
    report.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="test"):
        slides["load_evidence"](tmp_path, Config())


def test_generation_and_metric_data_fingerprints_must_match(slides, tmp_path):
    synthetic_evidence(tmp_path)
    path = tmp_path / "reports/samples/finetuned/generation.json"
    generation = json.loads(path.read_text())
    generation["manifest_sha256"] = "another-dataset"
    path.write_text(json.dumps(generation))
    with pytest.raises(ValueError, match="evidencias"):
        slides["load_evidence"](tmp_path, Config())


def test_results_can_be_embedded_only_in_a_temporary_synthetic_fixture(slides, tmp_path):
    synthetic_evidence(tmp_path)
    (tmp_path / "configs").mkdir()
    profile = Path("configs/rtx3060_8gb.toml")
    (tmp_path / profile).write_bytes(profile.read_bytes())
    output = tmp_path / "synthetic-deck"
    slides["build"](tmp_path, output, require_results=True)
    deck = Presentation(output / "pc3_difusion_anime.pptx")
    result = deck.slides[5]
    assert sum(shape.shape_type == 13 for shape in result.shapes) == 12
    assert "Pendiente" not in " ".join(
        shape.text for shape in result.shapes if shape.has_text_frame
    )
    state = json.loads((output / "estado.json").read_text())
    assert state["status"] == "executed_results"
