import { reduceMotion } from './common.js'

// Cuadros persistentes en vivo. Viven fuera de reveal.js: siguen igual al avanzar o retroceder.
const SERVER = new URLSearchParams(location.search).get('server') || 'http://127.0.0.1:8765'
const FRAME_MS = 420
const MAX_BYTES = 20 * 1024 * 1024

/** Cuadro + botón. `job({ signal, show })` hace la petición, puede llamar show(bitmap) con una vista previa
 *  y devuelve { frames: [url PNG], gif }. */
function createLab(root, job, { pickFile = false } = {}) {
  const canvas = root.querySelector('canvas'), gif = root.querySelector('img')
  const btn = root.querySelector('button'), input = root.querySelector('input')
  const ctx = canvas.getContext('2d')
  ctx.imageSmoothingEnabled = false
  let run = 0, raf = 0

  const label = (text, disabled) => { btn.textContent = text; btn.disabled = !!disabled }
  const flash = (id, text) => { label(text, true); setTimeout(() => id === run && label('EMPEZAR'), 2800) }
  const fit = (bitmap) => {
    ctx.globalAlpha = 1; ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  }

  function play(frames, id) {
    return new Promise((resolve) => {
      if (reduceMotion()) return resolve()
      const t0 = performance.now(), total = frames.length * FRAME_MS
      const step = (now) => {
        if (id !== run) return resolve()
        const t = Math.max(0, Math.min(now - t0, total)), f = Math.min(frames.length - 1, Math.floor(t / FRAME_MS))
        fit(frames[f])
        if (f + 1 < frames.length) { ctx.globalAlpha = (t - f * FRAME_MS) / FRAME_MS; ctx.drawImage(frames[f + 1], 0, 0, canvas.width, canvas.height); ctx.globalAlpha = 1 }
        if (t >= total) return resolve()
        raf = requestAnimationFrame(step)
      }
      raf = requestAnimationFrame(step)
    })
  }

  async function start(arg) {
    const id = ++run
    cancelAnimationFrame(raf)
    gif.hidden = true; canvas.hidden = false
    label('EJECUTANDO', true)
    try {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 600000)
      const data = await job(arg, { signal: controller.signal, show: (b) => id === run && fit(b) })
      clearTimeout(timer)
      const frames = await Promise.all(data.frames.map(async (u) => createImageBitmap(await (await fetch(u)).blob())))
      if (id !== run) return
      await play(frames, id)
      if (id !== run) return
      gif.src = data.gif
      gif.hidden = false; canvas.hidden = true
      label('COMPLETO')
    } catch (error) {
      if (id !== run) return
      console.warn('Demo en vivo:', error)
      flash(id, error instanceof TypeError || error.name === 'AbortError' ? 'SIN SERVIDOR' : 'NO SE PUDO')
    }
  }

  btn.addEventListener('click', () => {
    btn.blur()
    if (!pickFile) return start()
    input.value = ''; input.click()
  })
  input?.addEventListener('change', () => {
    const image = input.files[0]
    if (!image) return
    if (!image.type.startsWith('image/') || image.size > MAX_BYTES) return flash(run, 'IMAGEN NO VÁLIDA')
    start(image)
  })
}

const bitmap = async (url) => createImageBitmap(await (await fetch(url)).blob())

// SDEdit: la imagen elegida se muestra al instante y el servidor devuelve el proceso completo.
async function sdeditJob(image, { signal, show }) {
  show(await createImageBitmap(image, { resizeWidth: 64, resizeHeight: 64, resizeQuality: 'high' }))
  const res = await fetch(`${SERVER}/sdedit`, { method: 'POST', body: image, headers: { 'Content-Type': 'application/octet-stream' }, signal })
  if (!res.ok) throw new Error(String(res.status))
  return res.json()
}

// Generar: el servidor envía primero el ruido inicial de las 4 y, al terminar, todo el proceso.
async function generateJob(_, { signal, show }) {
  const res = await fetch(`${SERVER}/generate`, { method: 'POST', signal })
  if (!res.ok) throw new Error(String(res.status))
  const reader = res.body.getReader(), decoder = new TextDecoder()
  let buffer = '', result = null
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let end
    while ((end = buffer.indexOf('\n')) >= 0) {
      const text = buffer.slice(0, end); buffer = buffer.slice(end + 1)
      if (!text) continue
      const message = JSON.parse(text)
      if (message.error) throw new Error(message.error)
      if (message.first) show(await bitmap(message.first))
      if (message.frames) result = message
    }
  }
  if (!result) throw new Error('respuesta incompleta')
  return result
}

export function initLab() {
  createLab(document.getElementById('labGen'), generateJob)
  createLab(document.getElementById('labEdit'), sdeditJob, { pickFile: true })
}
