import { readFile, writeFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export function inlineDocument(html, asset) {
  const local = (file) => {
    if (!file.startsWith('./assets/')) throw new Error('El recurso debe ser local al build.');
    return asset(file);
  };
  return html
    .replace(/<script\b[^>]*\bsrc="([^"]+)"[^>]*><\/script>/g,
      (_, file) => `<script type="module">${local(file).replaceAll('</script', '<\\/script')}</script>`)
    .replace(/<link\b[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g,
      (_, file) => `<style>${local(file).replaceAll('</style', '<\\/style')}</style>`);
}

async function publish() {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const dist = resolve(root, 'dist');
  let html = inlineDocument(await readFile(resolve(dist, 'index.html'), 'utf8'), (file) => {
    const path = resolve(dist, file);
    if (!path.startsWith(dist + sep)) throw new Error('Recurso fuera del build.');
    return readFileSync(path, 'utf8');
  });
  const policy = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; " +
    "img-src data:; font-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'";
  html = html.replace('<head>', `<head><meta http-equiv="Content-Security-Policy" content="${policy}">`);
  const licenses = {};
  for (const name of ['react', 'react-dom', 'scheduler', 'animejs', 'lucide-react', 'tailwindcss']) {
    const folder = resolve(root, 'node_modules', name);
    for (const filename of ['LICENSE', 'LICENSE.md', 'LICENSE.txt']) {
      try { licenses[name] = await readFile(resolve(folder, filename), 'utf8'); break; }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
    }
    if (!licenses[name]) throw new Error(`Falta licencia de ${name}`);
  }
  const legal = JSON.stringify(licenses).replaceAll('<', '\\u003c');
  html = html.replace('</body>', `<script type="application/json" id="third-party-licenses">${legal}</script></body>`);
  if (/<script[^>]+src=|<link[^>]+rel="stylesheet"|<link[^>]+rel="modulepreload"/.test(html)) {
    throw new Error('Quedan recursos externos en la presentación.');
  }
  await writeFile(resolve(root, 'presentacion.html'), html);
  console.log(`HTML autocontenido: presentacion.html (${Math.round(Buffer.byteLength(html) / 1024)} KiB)`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await publish();
