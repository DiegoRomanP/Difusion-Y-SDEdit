from types import SimpleNamespace

import pytest
import torch
from diffusers import DDPMScheduler

from pc3.config import Config
from pc3.model import ExponentialMovingAverage, create_model, q_sample, unet_config
from pc3.sampling import sample_tensor


def test_educational_forward_process_matches_scheduler():
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    clean = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4) / 96
    noise = torch.ones_like(clean)
    times = torch.tensor([0, 999])
    actual = q_sample(clean, times, noise, scheduler.alphas_cumprod)
    assert torch.allclose(actual, scheduler.add_noise(clean, noise, times))
    assert torch.allclose(actual[1], noise[1], atol=0.01)


def test_ema_context_restores_raw_weights_without_optimizer_steps():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    ema = ExponentialMovingAverage(model, decay=0.5, device="cpu")
    with torch.no_grad():
        model.weight.fill_(3)
    ema.update(model)
    with ema.apply(model):
        assert torch.equal(model.weight, torch.full_like(model.weight, 2))
    assert torch.equal(model.weight, torch.full_like(model.weight, 3))


@pytest.mark.parametrize("sampler", ["ddim", "ddpm"])
def test_sampling_seed_is_per_image_without_using_a_real_model(sampler):
    class ZeroNoise(torch.nn.Module):
        def forward(self, image, _time):
            return SimpleNamespace(sample=torch.zeros_like(image))

    model = ZeroNoise()
    batched, _ = sample_tensor(
        model, seeds=[10, 11], steps=3, sampler=sampler, device=torch.device("cpu")
    )
    single, _ = sample_tensor(
        model, seeds=[11], steps=3, sampler=sampler, device=torch.device("cpu")
    )
    assert torch.equal(batched[1], single[0])
    assert model.training  # El muestreador restaura el modo previo.


def test_old_checkpoint_cannot_silently_initialize_missing_weights(monkeypatch):
    model = SimpleNamespace(config=unet_config())
    monkeypatch.setattr(
        "pc3.model.UNet2DModel.from_pretrained",
        lambda *args, **kwargs: (model, {"missing_keys": ["layer.weight"]}),
    )
    with pytest.raises(ValueError, match="silenciosamente"):
        create_model(Config(), pretrained=True)


def test_pretrained_source_is_pinned_and_never_loads_remote_code(monkeypatch):
    model = SimpleNamespace(config=unet_config())
    captured = {}

    def fake_download(*args, **kwargs):
        captured.update(kwargs)
        return model, {"missing_keys": [], "unexpected_keys": []}

    monkeypatch.setattr("pc3.model.UNet2DModel.from_pretrained", fake_download)
    assert create_model(Config(), pretrained=True) is model
    assert captured["revision"] == Config().pretrained_revision
    assert captured["output_loading_info"] is True
    assert "trust_remote_code" not in captured
