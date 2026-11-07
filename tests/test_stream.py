from itertools import islice

from pc3.stream import ReplayBatchSampler, learning_rate_at


def test_resume_has_same_batches_even_across_epochs():
    original = list(islice(iter(ReplayBatchSampler(7, 3, seed=42)), 20))
    resumed = list(islice(iter(ReplayBatchSampler(7, 3, seed=42, start=13)), 7))
    assert resumed == original[13:]
    flattened = [index for batch in original for index, _ in batch]
    assert sorted(flattened[:7]) == list(range(7))
    assert sorted(flattened[7:14]) == list(range(7))


def test_absolute_positions_are_independent_of_worker_prefetch():
    batches = list(islice(iter(ReplayBatchSampler(3, 2, seed=7, start=5)), 4))
    assert [position for batch in batches for _, position in batch] == list(range(10, 18))


def test_lr_is_reconstructed_from_completed_optimizer_updates():
    assert learning_rate_at(0, 100, 10, 1e-4) == 1e-5
    assert learning_rate_at(9, 100, 10, 1e-4) == 1e-4
    assert learning_rate_at(10, 100, 10, 1e-4) == 1e-4
    assert 0 < learning_rate_at(99, 100, 10, 1e-4) < 1e-4
