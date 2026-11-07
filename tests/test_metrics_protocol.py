from copy import deepcopy

import pytest

from pc3.metrics import assert_comparable, metric_protocol


def example(label="A"):
    return {
        "label": label,
        "protocol": {"num_images": 1000, "split": "test"},
        "generation": {"hardware": {"name": "RTX"}, "count": 1000},
        "provenance": {"contract": {"max_steps": 10}, "initialization": "pretrained_reference"},
    }


def test_different_sample_counts_cannot_be_compared():
    a, b = example(), example("B")
    b["protocol"]["num_images"] = 999
    with pytest.raises(ValueError, match="protocolo"):
        assert_comparable([a, b])


def test_different_training_contracts_cannot_be_compared():
    a, b = example(), example("B")
    b["provenance"]["contract"]["max_steps"] = 20
    with pytest.raises(ValueError, match="entrenamiento"):
        assert_comparable([a, b])


def test_partial_training_must_be_explicit():
    a, b = example(), example("B")
    b["provenance"]["initialization"] = "random"
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
    b["provenance"]["initialization"] = "pretrained"
    b["run_summary"] = {"status": "completed", "completed_steps": 10, "consumed_images": 9}
    with pytest.raises(ValueError, match="imágenes"):
        assert_comparable([a, b])
