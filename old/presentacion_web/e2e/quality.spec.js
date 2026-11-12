import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const url = pathToFileURL(resolve('presentacion.html')).href;
mkdirSync('.qa/screens', { recursive: true });
const diagnostics = new WeakMap();
test.beforeEach(async ({ page }) => {
  const messages = [];
  diagnostics.set(page, messages);
  page.on('pageerror', (error) => messages.push(error.message));
  page.on('console', (message) => { if (['error', 'warning'].includes(message.type())) messages.push(message.text()); });
  page.on('request', (request) => { if (/^https?:/.test(request.url())) messages.push(`Petición remota: ${request.url()}`); });
});
test.afterEach(async ({ page }) => { expect(diagnostics.get(page)).toEqual([]); });

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 },
  { width: 1024, height: 768 }, { width: 768, height: 1024 }, { width: 320, height: 800 }]) {
  test(`todas las escenas caben a ${viewport.width} px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const pages = viewport.width === 1440 ? ['1', '2', '3', '4', '5', '6', '7', 'preguntas', 'fuentes'] : ['1', '2', '3', '4', '5', '6', '7'];
    for (const [index, hash] of pages.entries()) {
      await page.goto(`${url}#/${hash}`);
      await expect(page.locator('.slide')).toHaveAttribute('data-slide', String(index + 1));
      await expect(page.locator('.slide')).toHaveAttribute('data-ready', 'true');
      await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Anterior', exact: true })).toBeVisible();
      const bounds = await page.evaluate(() => {
        const stage = document.querySelector('.stage');
        return { width: document.documentElement.scrollWidth, viewport: innerWidth,
          vertical: stage.scrollHeight - stage.clientHeight,
          overflowing: [...document.querySelectorAll('.stage p,.stage h1,.stage h2')]
            .filter((element) => element.scrollWidth > element.clientWidth + 2).map((element) => element.textContent) };
      });
      expect(bounds.width).toBeLessThanOrEqual(bounds.viewport);
      expect(bounds.overflowing).toEqual([]);
      if (viewport.width >= 1024) expect(bounds.vertical, `Escena ${hash}`).toBeLessThanOrEqual(2);
      await page.screenshot({ path: `.qa/screens/${viewport.width}-${String(index + 1).padStart(2, '0')}.png` });
    }
  });
}

test('las nueve escenas pasan accesibilidad WCAG AA', async ({ page }) => {
  for (const [index, hash] of ['1', '2', '3', '4', '5', '6', '7', 'preguntas', 'fuentes'].entries()) {
    await page.goto(`${url}#/${hash}`);
    await expect(page.locator('.slide')).toHaveAttribute('data-slide', String(index + 1));
    await expect(page.locator('.slide')).toHaveAttribute('data-ready', 'true');
    const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
    expect(audit.violations, `Accesibilidad en ${hash}`).toEqual([]);
  }
});

test('el control de ruido no dispara la navegación por teclado', async ({ page }) => {
  await page.goto(`${url}#/2`);
  const slider = page.getByRole('slider', { name: 'Nivel de ruido' });
  await slider.focus();
  await page.keyboard.press('ArrowRight');
  await expect(slider).toHaveValue('451');
  await expect(page).toHaveURL(/#\/2$/);
});

test('anexos por índice y cierre sin avanzar fuera del recorrido', async ({ page }) => {
  await page.goto(`${url}#/7`);
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await page.keyboard.press('ArrowRight');
  await expect(page).toHaveURL(/#\/7$/);
  await page.getByRole('button', { name: 'Abrir índice' }).click();
  await page.getByRole('dialog').getByRole('button', { name: /Fuentes/ }).click();
  await expect(page).toHaveURL(/#\/fuentes$/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Fuentes');
  await page.getByRole('button', { name: 'Volver al cierre' }).click();
  await expect(page).toHaveURL(/#\/7$/);
});

test('movimiento reducido deja visibles todos los elementos', async ({ page }) => {
  await page.goto(url);
  await expect(page.locator('.deck-shell')).toHaveAttribute('data-motion', 'off');
  await expect(page.locator('canvas')).toHaveAttribute('data-ready', 'true');
  const opacity = await page.locator('.reveal').evaluateAll((elements) => elements.map((element) => Number(getComputedStyle(element).opacity)));
  expect(opacity.every((value) => value === 1)).toBe(true);
});

test('Anime.js reconstruye los píxeles y permite pausar el movimiento', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto(url);
  await expect(page.locator('.deck-shell')).toHaveAttribute('data-motion', 'on');
  await expect(page.locator('canvas')).toHaveAttribute('data-ready', 'false');
  await expect(page.locator('canvas')).toHaveAttribute('data-ready', 'true');
  await page.getByRole('button', { name: 'Pausar animaciones' }).click();
  await expect(page.locator('.deck-shell')).toHaveAttribute('data-motion', 'off');
  await expect(page.getByRole('button', { name: 'Repetir reconstrucción' })).toBeDisabled();
});

test('el cronómetro empieza por decisión del expositor y se puede reiniciar', async ({ page }) => {
  await page.clock.install();
  await page.goto(url);
  await page.getByRole('button', { name: 'Iniciar cronómetro', exact: true }).click();
  await page.clock.fastForward(2000);
  await expect(page.locator('.timer')).toHaveText('00:02');
  await page.getByRole('button', { name: 'Pausar cronómetro' }).click();
  await page.clock.fastForward(2000);
  await expect(page.locator('.timer')).toHaveText('00:02');
  await page.getByRole('button', { name: 'Reiniciar cronómetro' }).click();
  await expect(page.locator('.timer')).toHaveText('00:00');
});

test('pausar y reanudar no descarta fracciones de segundo', async ({ page }) => {
  const start = new Date('2026-11-07T00:00:00Z');
  await page.clock.install({ time: start });
  await page.clock.pauseAt(new Date(start.getTime() + 1000));
  await page.goto(url);
  await page.getByRole('button', { name: 'Iniciar cronómetro', exact: true }).click();
  await page.clock.fastForward(1500);
  await page.getByRole('button', { name: 'Pausar cronómetro', exact: true }).click();
  await page.clock.fastForward(1000);
  await page.getByRole('button', { name: 'Iniciar cronómetro', exact: true }).click();
  await page.clock.fastForward(500);
  await expect(page.locator('.timer')).toHaveText('00:02');
});

test('el guion es modal y su foco no se escapa a la presentación', async ({ page }) => {
  await page.goto(url);
  await page.getByRole('button', { name: 'Abrir guion' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
  expect(audit.violations).toEqual([]);
  await page.keyboard.press('Tab');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('dialog').getByRole('button', { name: 'Cerrar', exact: true })).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Abrir guion' })).toBeFocused();
});

test('pantalla completa se activa y se abandona desde un gesto del usuario', async ({ page }) => {
  await page.goto(url);
  await page.getByRole('button', { name: 'Pantalla completa', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Salir de pantalla completa' })).toBeVisible();
  await page.getByRole('button', { name: 'Salir de pantalla completa' }).click();
  await expect(page.getByRole('button', { name: 'Pantalla completa', exact: true })).toBeVisible();
});

test('espacio abre las preguntas nativas en lugar de interceptarlas como navegación', async ({ page }) => {
  await page.goto(`${url}#/preguntas`);
  await page.locator('summary').first().focus();
  await page.keyboard.press('Space');
  await expect(page.locator('details').first()).toHaveAttribute('open', '');
  await expect(page).toHaveURL(/#\/preguntas$/);
});
