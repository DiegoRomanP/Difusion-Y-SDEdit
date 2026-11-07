import assert from 'node:assert/strict';
import test from 'node:test';
import { inlineDocument } from '../tools/standalone.mjs';

test('JS y CSS quedan embebidos, no referencias a assets ni scripts remotos', () => {
  const html = '<head><link rel="stylesheet" href="./assets/a.css"></head><body>' +
    '<script type="module" src="./assets/a.js"></script></body>';
  const assets = { './assets/a.css': 'body{color:red}', './assets/a.js': 'const x="</script>";' };
  const result = inlineDocument(html, (file) => assets[file]);
  assert.ok(result.includes('<style>body{color:red}</style>'));
  assert.ok(result.includes('<script type="module">'));
  assert.ok(!result.includes('src="./assets/'));
  assert.ok(result.includes('<\\/script>'));
});

test('no se acepta una referencia remota como un recurso autocontenido', () => {
  assert.throws(() => inlineDocument('<script src="https://example.com/a.js"></script>', () => ''),
    /local/);
});
