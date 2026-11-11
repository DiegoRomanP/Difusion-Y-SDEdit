from copy import deepcopy

import pytest

from pc3.metrics import assert_comparable, divergences, metric_protocol


def example(label="A"):
    return {
        "label": label,
        "protocol": {"num_images": 1000, "split": "test"},
        "generation": {"hardware": {"name": "RTX"}, "count": 1000},
        "provenance": {"contract": {"max_steps": 10}},
        "run_summary": {"status": "completed", "completed_steps": 10, "consumed_images": 10},
    }


def test_different_sample_counts_are_declared_not_hidden():
    a, b = example(), example("B")
    b["protocol"]["num_images"] = 999
    assert [item["kind"] for item in divergences([a, b])] == ["protocolo"]
    with pytest.raises(ValueError, match="no comparables"):
        assert_comparable([a, b])


def test_different_training_contracts_are_declared_not_hidden():
    a, b = example(), example("B")
    b["provenance"]["contract"]["max_steps"] = 20
    b["run_summary"] = {"status": "completed", "completed_steps": 20, "consumed_images": 20}
    declared = divergences([a, b])
    assert [item["kind"] for item in declared] == ["contrato_de_entrenamiento"]
    assert declared[0]["detail"] == ["max_steps"]


def test_a_different_test_split_is_contamination_not_a_divergence():
    """Cambiar la partición o el manifiesto invalida el número, no lo empeora."""
    a, b = example(), example("B")
    b["protocol"]["split"] = "val"
    with pytest.raises(ValueError, match="contaminación"):
        divergences([a, b])


def test_different_inference_hardware_is_recorded_as_a_divergence():
    a, b = example(), example("B")
    b["generation"]["hardware"] = {"name": "OtraGPU"}
    declared = divergences([a, b])
    assert [item["kind"] for item in declared] == ["hardware_de_inferencia"]


def test_partial_training_must_be_explicit():
    a, b = example(), example("B")
    b["run_summary"] = {"status": "paused", "completed_steps": 5}
    with pytest.raises(ValueError, match="incompleto"):
        assert_comparable([a, b])
    assert_comparable([a, b], allow_partial=True)


def test_kid_seed_and_subset_settings_are_part_of_the_protocol():
    generation = {
        "sampler": "ddim",
        "steps": 50,
        "seed": 10,
        "batch_size": 1,
        "amp": True,
        "pc3_source_sha256": "synthetic-source-digest",
    }
    a = metric_protocol("digest", "test", ["a", "b"], 2, 2, 50, 42, generation)
    b = metric_protocol("digest", "test", ["a", "b"], 2, 2, 50, 43, generation)
    assert a != b
    assert a == deepcopy(a)


def test_complete_status_is_not_enough_if_image_budget_differs():
    a, b = example(), example("B")
    b["run_summary"] = {"status": "completed", "completed_steps": 10, "consumed_images": 9}
    assert [item["kind"] for item in divergences([a, b])] == ["imagenes_consumidas"]
    with pytest.raises(ValueError, match="imagenes_consumidas"):
        assert_comparable([a, b])
    # --allow-partial sigue siendo lo único que tolera un presupuesto distinto.
    assert_comparable([a, b], allow_partial=True)
