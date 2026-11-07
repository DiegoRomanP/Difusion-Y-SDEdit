from dataclasses import replace

import pytest

from pc3.config import Config, load_config, training_contract


def test_default_profile_targets_eight_gb():
    config = Config()
    config.validate()
    assert config.image_size == 64
    assert config.batch_size * config.accumulation_steps == 16


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"image_size": 128},
        {"learning_rate": -1},
        {"warmup_steps": 40000},
        {"ema_decay": 1.0},
        {"max_steps": True},
    ],
)
def test_invalid_profiles_fail_before_downloads(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


def test_unknown_toml_keys_are_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("batch_szie = 2\n")
    with pytest.raises(ValueError, match="batch_szie"):
        load_config(path)


def test_training_contract_ignores_only_evaluation_output_options():
    config = Config()
    assert training_contract(config) == training_contract(replace(config, sample_count=100))
    assert training_contract(config) != training_contract(replace(config, learning_rate=2e-4))


def test_checked_in_profile_matches_defaults():
    assert load_config("configs/rtx3060_8gb.toml") == Config()
