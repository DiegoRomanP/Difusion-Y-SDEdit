"""Orden de datos reanudable: independiente del prefetch y del número de workers."""

import math
import random


class ReplayBatchSampler:
    """Genera (índice, posición absoluta); start cuenta microbatches consumidos.

    Un sampler aleatorio estándar guarda un generador adelantado por prefetch. Aquí
    cada época se reconstruye desde su semilla y cada flip depende de la posición,
    no del estado oculto de un worker. No se descartan imágenes al cruzar épocas.
    """

    def __init__(self, size: int, batch_size: int, seed: int, start: int = 0):
        if size <= 0 or batch_size <= 0 or start < 0:
            raise ValueError("size/batch_size deben ser positivos y start >=0.")
        self.size, self.batch_size, self.seed, self.start = size, batch_size, seed, start

    def __iter__(self):
        position = self.start * self.batch_size
        cached_epoch, permutation = -1, []
        while True:
            batch = []
            for _ in range(self.batch_size):
                epoch, offset = divmod(position, self.size)
                if epoch != cached_epoch:
                    permutation = list(range(self.size))
                    random.Random(self.seed + epoch).shuffle(permutation)
                    cached_epoch = epoch
                batch.append((permutation[offset], position))
                position += 1
            yield batch


def learning_rate_at(step: int, max_steps: int, warmup: int, base: float) -> float:
    """Warmup lineal + coseno, reconstruido desde actualizaciones completadas."""
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / (max_steps - warmup)))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))
