import assert from 'node:assert/strict';
import test from 'node:test';
import { routeIndex, routeHash, formatTime, nextIndex } from '../src/navigation.js';

test('los enlaces no permiten índices inválidos ni anexos accidentales', () => {
  assert.equal(routeIndex('#/1'), 0);
  assert.equal(routeIndex('#/7'), 6);
  assert.equal(routeIndex('#/preguntas'), 7);
  assert.equal(routeIndex('#/fuentes'), 8);
  for (const hash of ['', '#/0', '#/99', '#/2xyz', '#/<script>', '#/-1']) {
    assert.equal(routeIndex(hash), 0);
  }
});

test('siguiente no salta del cierre a los anexos', () => {
  assert.equal(nextIndex(0, 1), 1);
  assert.equal(nextIndex(6, 1), 6);
  assert.equal(nextIndex(0, -1), 0);
  assert.equal(nextIndex(7, -1), 6);
  assert.equal(nextIndex(8, 1), 8);
});

test('cada escena tiene un enlace estable y tiempos legibles', () => {
  for (let index = 0; index < 9; index++) {
    assert.equal(routeIndex(routeHash(index)), index);
  }
  assert.equal(formatTime(0), '00:00');
  assert.equal(formatTime(330), '05:30');
  assert.equal(formatTime(360), '06:00');
});
