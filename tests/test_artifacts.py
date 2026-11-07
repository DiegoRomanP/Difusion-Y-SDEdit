from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pc3.config import Config
from pc3.model import export_weights, load_weights, weights_metadata
from pc3.reporting import report
from pc3.training import _loader, _save_checkpoint


def test_local_checkpoint_uses_weights_only_compatible_types(tmp_path):
    path = tmp_path / "latest.pt"
    state = {"step": 17, "model": {"w": torch.ones(2)}, "history": [{"loss": 0.5}]}
    _save_checkpoint(path, state)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored["step"] == 17
    assert torch.equal(restored["model"]["w"], state["model"]["w"])
    assert not list(tmp_path.glob("*.tmp"))


def test_safetensors_embeds_provenance_without_a_sidecar(tmp_path):
    model = torch.nn.Linear(2, 1)
    model.config = {"test_model": True}
    path = tmp_path / "best.safetensors"
    export_weights(model, path, {"run_id": "synthetic-unit-test", "step": 7})
    assert weights_metadata(path)["step"] == 7
    assert weights_metadata(path)["model_config"] == {"test_model": True}
    assert [file.name for file in tmp_path.iterdir()] == ["best.safetensors"]
    with pytest.raises(ValueError, match="Arquitectura"):
        load_weights(path)  # Rechazar configuración arbitraria antes de reservar una U-Net.


def test_report_never_manufactures_results_when_metrics_are_missing(tmp_path):
    output = tmp_path / "report"
    with pytest.raises(FileNotFoundError):
        report([Path("does-not-exist.json")], output=output)
    assert not output.exists()


def test_dataloader_creation_does_not_advance_training_rng():
    loader = _loader([torch.ones(3, 8, 8)], replace(Config(), num_workers=0), torch.device("cpu"))
    before = torch.get_rng_state().clone()
    next(iter(loader))
    assert torch.equal(torch.get_rng_state(), before)
