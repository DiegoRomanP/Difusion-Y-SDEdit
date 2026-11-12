import { test, expect } from '@playwright/test';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const url = pathToFileURL(resolve('presentacion.html')).href;

test('el HTML se abre sin servidor ni red y permite navegar sin errores', async ({ page }) => {
  const errors = [];
  const remote = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (['error', 'warning'].includes(message.type())) errors.push(message.text()); });
  page.on('request', (request) => { if (/^https?:/.test(request.url())) remote.push(request.url()); });
  await page.goto(url);
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await page.getByRole('button', { name: 'Siguiente', exact: true }).click();
  await expect(page).toHaveURL(/#\/2$/);
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  expect(remote).toEqual([]);
  expect(errors).toEqual([]);
});

test('el teclado avanza y el guion se abre y cierra sin perder la escena', async ({ page }) => {
  await page.goto(url);
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await page.keyboard.press('ArrowRight');
  await expect(page).toHaveURL(/#\/2$/);
  await page.keyboard.press('n');
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByRole('dialog')).toContainText('ruido gaussiano');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await expect(page).toHaveURL(/#\/2$/);
});
