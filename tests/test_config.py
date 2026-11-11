from dataclasses import replace

import pytest

from pc3.config import Config, load_config, training_contract


def test_default_profile_is_valid_and_uses_the_current_architecture():
    config = Config()
    config.validate()
    assert config.image_size == 64
    assert config.arch == "anime64"
    assert config.batch_size * config.accumulation_steps == 16


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"image_size": 20},
        {"image_size": 100},
        {"arch": "no-existe"},
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


def test_architecture_belongs_to_the_training_contract():
    """Dos arquitecturas distintas no son el mismo experimento con otro nombre."""
    config = Config()
    assert training_contract(config) != training_contract(replace(config, arch="flores64"))


@pytest.mark.parametrize("profile", ["configs/rtx3060_8gb.toml", "configs/rtx3060_12gb.toml"])
def test_historical_profiles_pin_the_architecture_that_produced_them(profile):
    """Las evidencias de la 3060 son de flores64: el perfil debe decirlo, no heredarlo."""
    config = load_config(profile)
    assert config.arch == "flores64"
    assert config.batch_size * config.accumulation_steps == 16
