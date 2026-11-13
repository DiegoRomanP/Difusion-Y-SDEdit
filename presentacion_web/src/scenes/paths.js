import { THREE, createStage, label, PAL, ease } from '../common.js'

// Mismo recorrido ruido -> dato: 1000 evaluaciones (DDPM) frente a 50 (DDIM).
export default (canvas) => {
  const st = createStage(canvas, { z: 12, y: 0, fov: 40, staticT: 8 })
  const g = new THREE.Group(); st.scene.add(g)
  const path = (y, n, amp, size, color) => {
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3)
    const base = new THREE.Color(color), dim = new THREE.Color(PAL.steel)
    for (let i = 0; i < n; i++) {
      const u = i / (n - 1)
      pos.set([-3.8 + u * 7.6, y + Math.sin(u * Math.PI * 2) * 0.28 + (Math.random() - 0.5) * amp * (1 - u), (Math.random() - 0.5) * 0.3], i * 3)
    }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true, transparent: true, depthWrite: false }))
    g.add(pts)
    return (p) => { for (let i = 0; i < n; i++) (i / n <= p ? base : dim).toArray(col, i * 3); geo.attributes.color.needsUpdate = true }
  }
  const a = path(1.35, 1000, 0.55, 0.06, PAL.glow)
  const b = path(-1.35, 50, 0.3, 0.2, PAL.cyan)
  const ta = label('DDPM · 1000 evaluaciones', { scale: 0.6, size: 52, color: '#A98BFF' }); ta.position.set(0, 2.3, 0)
  const tb = label('DDIM · 50 evaluaciones', { scale: 0.6, size: 52, color: '#6FC3FF' }); tb.position.set(0, -0.35, 0)
  const noiseL = label('ruido', { scale: 0.55, size: 50, color: '#C9BEE6', weight: 500 }); noiseL.position.set(-3.8, -2.35, 0)
  const dataL = label('rostro', { scale: 0.55, size: 50, color: '#C9BEE6', weight: 500 }); dataL.position.set(3.8, -2.35, 0)
  g.add(ta, tb, noiseL, dataL)
  st.fn = (t) => {
    const p = ease(((t % 9) / 7))
    a(p); b(p)
    g.rotation.y = Math.sin(t * 0.3) * 0.2
  }
  return st
}
