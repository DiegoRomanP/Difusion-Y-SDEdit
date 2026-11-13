import { THREE } from './common.js'

/** Rostro anime esquemático dibujado en canvas 2D (solo para ilustrar el proceso, no es una muestra del modelo). */
export function drawFace(size = 128, bg = null) {
  const c = document.createElement('canvas')
  c.width = c.height = size
  const g = c.getContext('2d')
  if (bg) { g.fillStyle = bg; g.fillRect(0, 0, size, size) }
  g.scale(size / 128, size / 128)
  const ell = (x, y, rx, ry, fill) => { g.fillStyle = fill; g.beginPath(); g.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2); g.fill() }
  ell(64, 62, 52, 56, '#C2306A')
  ell(64, 74, 38, 42, '#FFE3D6')
  g.fillStyle = '#FF6F96'
  g.beginPath(); g.moveTo(20, 64); g.quadraticCurveTo(64, -6, 108, 64); g.quadraticCurveTo(92, 42, 80, 54); g.quadraticCurveTo(64, 30, 48, 54); g.quadraticCurveTo(36, 42, 20, 64); g.fill()
  for (const x of [46, 82]) {
    ell(x, 78, 10, 13, '#FFFFFF'); ell(x, 80, 7.5, 10.5, '#2BB3E8'); ell(x, 81, 3.6, 6.5, '#19082D'); ell(x - 2.5, 75, 2.6, 2.6, '#FFFFFF')
  }
  ell(36, 94, 7, 4, 'rgba(255,120,150,.6)'); ell(92, 94, 7, 4, 'rgba(255,120,150,.6)')
  g.strokeStyle = '#8A3A55'; g.lineWidth = 2.6; g.lineCap = 'round'
  g.beginPath(); g.arc(64, 95, 6, 0.15 * Math.PI, 0.85 * Math.PI); g.stroke()
  return c
}

const rng = (seed) => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }

/** Textura con la fórmula del proceso directo: x_t = √ᾱ·x0 + √(1−ᾱ)·ε. */
export function noisyTexture(alphaBar, size = 96, seed = 7) {
  const c = drawFace(size, '#2A336C')
  const g = c.getContext('2d')
  const img = g.getImageData(0, 0, size, size)
  const r = rng(seed)
  const a = Math.sqrt(alphaBar), b = Math.sqrt(1 - alphaBar)
  const gauss = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r())
  for (let i = 0; i < img.data.length; i += 4) for (let k = 0; k < 3; k++) {
    const x = a * (img.data[i + k] / 127.5 - 1) + b * gauss() * 0.7
    img.data[i + k] = Math.max(0, Math.min(255, (x + 1) * 127.5))
  }
  g.putImageData(img, 0, 0)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.magFilter = THREE.NearestFilter
  return tex
}

/** Puntos (x, y, r, g, b) de los píxeles opacos del rostro, para formarlo con partículas. */
export function faceSamples(n, extent = 2.2) {
  const size = 128
  const d = drawFace(size).getContext('2d').getImageData(0, 0, size, size).data
  const px = []
  for (let i = 0; i < size * size; i++) if (d[i * 4 + 3] > 128) px.push(i)
  const r = rng(42)
  const out = []
  for (let k = 0; k < n; k++) {
    const i = px[Math.floor(r() * px.length)]
    out.push([((i % size) / size - 0.5) * 2 * extent + (r() - 0.5) * 0.03, -(Math.floor(i / size) / size - 0.5) * 2 * extent, d[i * 4] / 255, d[i * 4 + 1] / 255, d[i * 4 + 2] / 255])
  }
  return out
}
