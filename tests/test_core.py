"""Pruebas de las piezas que, si fallan, invalidan el experimento sin dar error."""

import json
import random

import torch
from diffusers import UNet2DModel
from PIL import Image
from prepare_dataset import create_splits
from train import ReplayBatchSampler

from pc3.data import load_splits
from pc3.inference import sample_tensor
from pc3.model import create_model, load_weights, noise_scheduler, q_sample, save_weights


def tiny_unet():
    return UNet2DModel(
        sample_size=64,
        block_out_channels=(32, 32),
        norm_num_groups=8,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


def test_q_sample_matches_diffusers():
    scheduler = noise_scheduler()
    clean, noise = torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64)
    times = torch.tensor([0, 10, 500, 999])
    expected = scheduler.add_noise(clean, noise, times)
    assert torch.allclose(
        q_sample(clean, times, noise, scheduler.alphas_cumprod), expected, atol=1e-5
    )


def test_model_has_the_trained_architecture():
    assert sum(p.numel() for p in create_model().parameters()) == 88_217_219


def test_resumed_sampler_continues_the_same_order():
    full = iter(ReplayBatchSampler(size=10, batch_size=3, seed=1))
    batches = [next(full) for _ in range(8)]  # cruza varias épocas
    resumed = iter(ReplayBatchSampler(size=10, batch_size=3, seed=1, start=5))
    assert [next(resumed) for _ in range(3)] == batches[5:]


def test_near_duplicates_stay_in_one_split(tmp_path):
    rng = random.Random(0)
    for index in range(40):
        # Imagen de bloques 8x8 y su reflejo con un píxel distinto: casi duplicados.
        blocks = Image.new("RGB", (8, 8))
        blocks.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(64)])
        image = blocks.resize((64, 64), Image.Resampling.NEAREST)
        image.save(tmp_path / f"a{index}.png")
        mirror = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mirror.putpixel((0, 0), tuple((c + 1) % 256 for c in mirror.getpixel((0, 0))))
        mirror.save(tmp_path / f"b{index}.png")
    splits = create_splits(tmp_path, image_size=64, seed=0)
    split_of = {row["path"]: row["split"] for row in splits["records"]}
    assert len(split_of) == 80 and set(split_of.values()) == {"train", "val", "test"}
    for index in range(40):
        assert split_of[f"a{index}.png"] == split_of[f"b{index}.png"]
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    load_splits(tmp_path / "splits.json")  # no levanta: sin fugas


def test_weights_round_trip(tmp_path):
    model = tiny_unet()
    save_weights(model, tmp_path / "w.safetensors", {"step": 3})
    loaded, metadata = load_weights(tmp_path / "w.safetensors")
    assert metadata["step"] == 3
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_same_seed_gives_same_image_regardless_of_batch():
    torch.manual_seed(0)
    model = tiny_unet()
    alone, _ = sample_tensor(model, seeds=[7], steps=3, device=torch.device("cpu"))
    batch, _ = sample_tensor(model, seeds=[5, 7], steps=3, device=torch.device("cpu"))
    assert torch.allclose(alone[0], batch[1], atol=1e-5)
