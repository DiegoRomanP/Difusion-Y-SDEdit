import { THREE, createStage, ease, PAL } from '../common.js'
import { faceSamples } from '../faces.js'

// Nube de partículas: ruido -> rostro. loop=true repite (portada); loop=false forma el rostro y se queda (cierre).
export const makeParticles = (loop) => (canvas) => {
  const N = 24000
  const st = createStage(canvas, { z: 6.2, staticT: loop ? 5 : 9 })
  const samples = faceSamples(N, 1.7)
  const noise = new Float32Array(N * 3), target = new Float32Array(N * 3), delay = new Float32Array(N)
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3)
  const violet = new THREE.Color(PAL.glow)
  for (let i = 0; i < N; i++) {
    const [x, y, r, g, b] = samples[i]
    target.set([x, y, 0], i * 3)
    const u = Math.random() * 2 - 1, th = Math.random() * 6.283, s = Math.sqrt(1 - u * u), rad = 1.5 + Math.random() * 3.2
    noise.set([rad * s * Math.cos(th) * 1.5, rad * u * 0.9, rad * s * Math.sin(th)], i * 3)
    delay[i] = Math.random() * 0.3
    col.set([r, g, b], i * 3)
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
  const mat = new THREE.PointsMaterial({ size: 0.034, vertexColors: true, transparent: true, opacity: 1, depthWrite: false })
  const pts = new THREE.Points(geo, mat)
  st.scene.add(pts)
  const shift = canvas.classList.contains('bg') ? 2.9 : 0
  pts.position.x = shift

  st.fn = (t) => {
    let e
    if (loop) { const u = (t % 13) / 13; e = u < 0.38 ? ease(u / 0.38) : u < 0.72 ? 1 : 1 - ease((u - 0.72) / 0.28) }
    else e = ease(t / 5)
    for (let i = 0; i < N; i++) {
      const k = ease((e - delay[i]) / 0.7), j = i * 3
      const w = (1 - k) * 0.12 + 0.012
      pos[j] = noise[j] + (target[j] - noise[j]) * k + Math.sin(t * 1.3 + i) * w
      pos[j + 1] = noise[j + 1] + (target[j + 1] - noise[j + 1]) * k + Math.cos(t * 1.1 + i * 0.7) * w
      pos[j + 2] = noise[j + 2] * (1 - k) + Math.sin(t + i) * w
    }
    geo.attributes.position.needsUpdate = true
    pts.rotation.y = Math.sin(t * 0.25) * 0.25 * (1 - e * 0.7)
    mat.color.copy(violet)
  }
  return st
}
