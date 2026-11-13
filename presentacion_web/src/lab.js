import { reduceMotion } from './common.js'

// Cuadro persistente de SDEdit en vivo. Vive fuera de reveal.js: sigue igual al avanzar o retroceder.
const SERVER = new URLSearchParams(location.search).get('server') || 'http://127.0.0.1:8765'
const FRAME_MS = 420
const MAX_BYTES = 20 * 1024 * 1024

export function initLab() {
  const canvas = document.getElementById('labCanvas')
  const gif = document.getElementById('labGif')
  const btn = document.getElementById('labBtn')
  const file = document.getElementById('labFile')
  const ctx = canvas.getContext('2d')
  ctx.imageSmoothingEnabled = false
  let run = 0, raf = 0

  const label = (text, disabled) => { btn.textContent = text; btn.disabled = !!disabled }
  const clear = () => ctx.clearRect(0, 0, 64, 64)
  const showCanvas = () => { gif.hidden = true; canvas.hidden = false }
  const flash = (text) => { label(text, true); setTimeout(() => run === current() && label('EMPEZAR'), 2800) }
  const current = () => run

  // recorte cuadrado central a 64×64, como en el entrenamiento (solo para la vista previa inmediata)
  function drawCover(bitmap) {
    const side = Math.min(bitmap.width, bitmap.height)
    ctx.clearRect(0, 0, 64, 64)
    ctx.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, 64, 64)
  }

  function play(frames, id) {
    return new Promise((resolve) => {
      if (reduceMotion()) return resolve()
      const t0 = performance.now(), total = frames.length * FRAME_MS
      const step = (now) => {
        if (id !== run) return resolve()
        const t = Math.min(now - t0, total), f = Math.min(frames.length - 1, Math.floor(t / FRAME_MS))
        const k = (t - f * FRAME_MS) / FRAME_MS
        clear(); ctx.globalAlpha = 1; ctx.drawImage(frames[f], 0, 0)
        if (f + 1 < frames.length) { ctx.globalAlpha = k; ctx.drawImage(frames[f + 1], 0, 0); ctx.globalAlpha = 1 }
        if (t >= total) return resolve()
        raf = requestAnimationFrame(step)
      }
      raf = requestAnimationFrame(step)
    })
  }

  async function start(image) {
    const id = ++run
    cancelAnimationFrame(raf)
    showCanvas()
    label('EJECUTANDO', true)
    try {
      drawCover(await createImageBitmap(image))
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 180000)
      const res = await fetch(`${SERVER}/sdedit`, { method: 'POST', body: image, headers: { 'Content-Type': 'application/octet-stream' }, signal: controller.signal })
      clearTimeout(timer)
      if (!res.ok) throw new Error(String(res.status))
      const data = await res.json()
      const frames = await Promise.all(data.frames.map(async (u) => createImageBitmap(await (await fetch(u)).blob())))
      if (id !== run) return
      await play(frames, id)
      if (id !== run) return
      gif.src = data.gif
      gif.hidden = false; canvas.hidden = true
      label('COMPLETO')
    } catch (error) {
      if (id !== run) return
      console.warn('SDEdit en vivo:', error)
      flash(error instanceof TypeError || error.name === 'AbortError' ? 'SIN SERVIDOR' : 'IMAGEN NO VÁLIDA')
    }
  }

  btn.addEventListener('click', () => { btn.blur(); file.value = ''; file.click() })
  file.addEventListener('change', () => {
    const image = file.files[0]
    if (!image) return
    if (!image.type.startsWith('image/') || image.size > MAX_BYTES) return flash('IMAGEN NO VÁLIDA')
    start(image)
  })
}
