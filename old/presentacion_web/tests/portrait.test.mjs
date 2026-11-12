import assert from 'node:assert/strict';
import test from 'node:test';
import { createPixels, noisyColor } from '../src/portrait.js';

test('el diagrama es determinista, no consume imágenes o aleatoriedad de modelos', () => {
  const pixels = createPixels();
  assert.equal(pixels.length, 32 * 32);
  assert.deepEqual(pixels, createPixels());
  assert.ok(pixels.some((pixel) => pixel.color[0] > 200));
});

test('ruido cero conserva el dibujo y ruido máximo cambia su apariencia', () => {
  const pixel = createPixels()[12 * 32 + 16];
  assert.deepEqual(noisyColor(pixel, 0), pixel.color);
  assert.notDeepEqual(noisyColor(pixel, 1), pixel.color);
  assert.ok(noisyColor(pixel, 1).every((value) => Number.isFinite(value) && value >= 0 && value <= 255));
});
