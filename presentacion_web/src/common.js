import * as THREE from 'three'

export { THREE }
export const PAL = { violet: 0x5B3EA4, steel: 0x275883, indigo: 0x2A336C, plum: 0x46194A, night: 0x19082D, glow: 0xA98BFF, cyan: 0x4FD1FF, coral: 0xFF5C8A, amber: 0xFFC24B }
export const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches

export const ease = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x) }

/** Escena three.js asociada a un canvas. fn(t, dt) se llama por frame; en reduced-motion se dibuja un solo frame en staticT. */
export function createStage(canvas, { fov = 45, z = 8, y = 0, staticT = 6 } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 100)
  camera.position.set(0, y, z)
  const stage = { THREE, scene, camera, renderer, fn: () => {}, running: false, raf: 0, t0: 0, last: 0, size: [0, 0] }

  const fit = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight
    if (!w || !h) return false
    if (w !== stage.size[0] || h !== stage.size[1]) {
      stage.size = [w, h]
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    return true
  }
  const frame = (now) => {
    if (!stage.running) return
    stage.raf = requestAnimationFrame(frame)
    if (!fit()) return
    const t = Math.max(0, (now - stage.t0) / 1000)
    stage.fn(t, (now - stage.last) / 1000)
    stage.last = now
    renderer.render(scene, camera)
  }
  stage.start = () => {
    if (stage.running) return
    stage.running = true
    stage.t0 = stage.last = performance.now()
    if (reduceMotion()) {
      requestAnimationFrame(() => { if (fit()) { stage.fn(staticT, 0); renderer.render(scene, camera) } stage.running = false })
      return
    }
    stage.raf = requestAnimationFrame(frame)
  }
  stage.stop = () => { stage.running = false; cancelAnimationFrame(stage.raf) }
  return stage
}

/** Etiqueta de texto como sprite. */
export function label(text, { size = 56, color = '#F3EEFF', scale = 1, weight = 700 } = {}) {
  const c = document.createElement('canvas')
  c.width = 768; c.height = 128
  const g = c.getContext('2d')
  g.font = `${weight} ${size}px Inter, "Segoe UI", Helvetica, Arial, "Liberation Sans", sans-serif`
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = color
  g.fillText(text, 384, 64)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }))
  sprite.scale.set(6 * scale, scale, 1)
  return sprite
}

export function glowLine(points, color = PAL.glow, opacity = 0.8) {
  const geo = new THREE.BufferGeometry().setFromPoints(points)
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }))
}

export function lights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 1.1))
  const d = new THREE.DirectionalLight(0xffffff, 1.6)
  d.position.set(3, 5, 6)
  scene.add(d)
}

let history
export const loadHistory = () => (history ??= fetch(import.meta.env.BASE_URL + 'evidencia/history.json').then((r) => r.json()))
let metrics
export const loadMetrics = () => (metrics ??= fetch(import.meta.env.BASE_URL + 'evidencia/metrics.json').then((r) => r.json()))
