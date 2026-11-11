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


def test_create_model_builds_from_the_named_architecture_config():
    model = create_model(Config())
    architecture = unet_config(Config().arch)
    assert model.config["block_out_channels"] == architecture["block_out_channels"]
    assert model.config["sample_size"] == architecture["sample_size"]
