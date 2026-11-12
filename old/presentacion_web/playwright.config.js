import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  workers: 1,
  reporter: 'list',
  outputDir: '.qa/test-output',
  use: { browserName: 'chromium', offline: true, reducedMotion: 'reduce',
    viewport: { width: 1440, height: 900 }, screenshot: 'only-on-failure' },
});
