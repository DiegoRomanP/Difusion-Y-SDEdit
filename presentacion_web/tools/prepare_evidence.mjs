// Uso: node tools/prepare_evidence.mjs <carpeta_de_la_entrega>
// Genera public/evidencia/ a partir de la entrega del entrenamiento (sin pesos ni las 5000 imágenes).
import { cpSync, mkdirSync, readFileSync, readdirSync, writeFileSync, rmSync } from 'node:fs'
import { join } from 'node:path'

const src = process.argv[2]
if (!src) throw new Error('Falta la carpeta de la entrega')
const out = new URL('../public/evidencia/', import.meta.url).pathname
rmSync(out, { recursive: true, force: true })
mkdirSync(join(out, 'previews'), { recursive: true })
mkdirSync(join(out, 'muestras'), { recursive: true })

const run = join(src, 'runs/anime64_4070ti')
const rep = join(src, 'reports/samples/anime64_4070ti')

// history.csv -> JSON compacto. Las validaciones viven en filas "train" con val_mse; train se promedia cada 10 logs.
const [head, ...lines] = readFileSync(join(run, 'history.csv'), 'utf8').trim().split('\n')
const cols = head.split(',')
const rows = lines.map((l) => Object.fromEntries(l.split(',').map((v, i) => [cols[i], v])))
const num = (v) => (v === '' ? null : Number(v))
const trainRows = rows.filter((r) => r.kind === 'train')
const train = []
for (let i = 0; i < trainRows.length; i += 10) {
  const g = trainRows.slice(i, i + 10)
  train.push([num(g.at(-1).step), g.reduce((a, r) => a + Number(r.train_mse), 0) / g.length, num(g.at(-1).learning_rate)])
}
const val = rows.filter((r) => r.val_mse !== '').map((r) => [num(r.step), num(r.val_mse)])
writeFileSync(join(out, 'history.json'), JSON.stringify({ train, val, wall_seconds: Math.max(...rows.map((r) => Number(r.wall_seconds) || 0)) }))

// métricas resumidas (solo lo que se muestra)
const m = JSON.parse(readFileSync(join(rep, 'metrics.json'), 'utf8'))
const d = JSON.parse(readFileSync(join(rep, 'diagnostics.json'), 'utf8'))
const r = JSON.parse(readFileSync(join(run, 'run.json'), 'utf8'))
writeFileSync(join(out, 'metrics.json'), JSON.stringify({
  fid: m.fid, kid_mean: m.kid_mean, kid_std: m.kid_std, num_images: m.protocol.num_images,
  sampler: m.protocol.sampler, sampling_steps: m.protocol.sampling_steps,
  seconds_per_image: m.seconds_per_image, peak_vram_gib: m.peak_vram_gib,
  ssim_mean: d.pairwise_ssim_mean, ssim_std: d.pairwise_ssim_std,
  steps: r.completed_steps, parameter_count: r.parameter_count,
  validation_mse: m.provenance.validation_mse, gpu: r.hardware.name,
}, null, 1))

for (const f of ['grid.png', 'nearest_training.png', 'denoising.gif']) cpSync(join(rep, f), join(out, f))
for (const s of [0, 1000, 5000, 15000, 40000, 87274]) {
  const n = `step_${String(s).padStart(6, '0')}.png`
  cpSync(join(run, 'previews', n), join(out, 'previews', n))
}
const imgs = readdirSync(join(rep, 'images')).sort().slice(0, 24)
for (const f of imgs) cpSync(join(rep, 'images', f), join(out, 'muestras', f))
console.log('evidencia lista:', out, imgs.length, 'muestras')
