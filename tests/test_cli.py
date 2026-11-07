import pytest

from pc3.cli import main


def test_help_does_not_start_training(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "train" in capsys.readouterr().out


def test_sampling_requires_an_explicit_weight_source():
    with pytest.raises(SystemExit) as error:
        main(["sample", "--output", "unused"])
    assert error.value.code == 2


def test_training_requires_explicit_initialization():
    with pytest.raises(SystemExit) as error:
        main(["train"])
    assert error.value.code == 2
