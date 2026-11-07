import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const args = process.argv.slice(2);
const install = args[0] === '--install';
const command = install ? ['install', 'chromium'] : ['test', ...args];
const result = spawnSync(process.execPath, [resolve(root, 'node_modules/@playwright/test/cli.js'), ...command], {
  cwd: root, stdio: 'inherit',
  env: { ...process.env, PLAYWRIGHT_BROWSERS_PATH: resolve(root, '../.cache/presentation-web-browsers') },
});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
