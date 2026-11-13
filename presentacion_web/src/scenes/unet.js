import { THREE, createStage, label, PAL, lights, ease } from '../common.js'

// U-Net: 3 bloques de bajada, cuello, 3 de subida; las flechas arqueadas son los saltos (skip connections).
export default (canvas) => {
  const st = createStage(canvas, { z: 12, y: 0.3, fov: 40, staticT: 5 })
  lights(st.scene)
  const g = new THREE.Group(); st.scene.add(g)
  // [canal, resolución espacial]
  const levels = [[128, 64], [256, 32], [384, 16], [512, 8]]
  const hOf = (res) => 0.55 + res / 64 * 2.0
  const cols = [-3, -2, -1, 0, 1, 2, 3]
  const spec = [0, 1, 2, 3, 2, 1, 0]
  const blocks = spec.map((lv, i) => {
    const [ch, res] = levels[lv]
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.35 + ch / 512 * 0.55, hOf(res), 0.9), new THREE.MeshStandardMaterial({ color: i < 3 ? PAL.steel : i === 3 ? PAL.plum : PAL.violet, roughness: 0.4, metalness: 0.15 }))
    m.position.set(cols[i], 0, 0)
    g.add(m)
    const l = label(`${ch}`, { scale: 0.5, size: 60 }); l.position.set(cols[i], -hOf(res) / 2 - 0.35, 0.6); g.add(l)
    if (lv >= 2) { const a = label('atención', { scale: 0.4, size: 46, color: '#6FC3FF', weight: 500 }); a.position.set(cols[i], hOf(res) / 2 + 0.3, 0.6); g.add(a) }
    return m
  })
  const cap = label('codificador · cuello · decodificador', { scale: 0.5, size: 44, color: '#C9BEE6', weight: 500 }); cap.position.set(0, -2.35, 0.2); g.add(cap)
  // saltos
  const skips = []
  for (let i = 0; i < 3; i++) {
    const a = new THREE.Vector3(cols[i], hOf(levels[spec[i]][1]) / 2, 0), b = new THREE.Vector3(cols[6 - i], hOf(levels[spec[i]][1]) / 2, 0)
    const c = new THREE.QuadraticBezierCurve3(a, new THREE.Vector3((a.x + b.x) / 2, 2.6 + (2 - i) * 0.4, 0.3), b)
    g.add(new THREE.Mesh(new THREE.TubeGeometry(c, 40, 0.03, 8), new THREE.MeshBasicMaterial({ color: PAL.cyan, transparent: true, opacity: 0.55 })))
    skips.push(c)
  }
  const orb = new THREE.Mesh(new THREE.SphereGeometry(0.17, 20, 20), new THREE.MeshBasicMaterial({ color: PAL.glow }))
  g.add(orb)
  const io = (txt, x, c) => { const l = label(txt, { scale: 0.55, size: 52, color: c }); l.position.set(x, 2.9, 0); g.add(l) }
  io('xₜ + t', -3, '#6FC3FF'); io('ε̂', 3, '#A98BFF')
  st.fn = (t) => {
    const u = (t % 9) / 9
    const k = ease(u)
    // el testigo baja por el codificador y sube por el decodificador
    const x = -3 + k * 6
    const i = Math.min(6, Math.max(0, Math.round(x + 3)))
    orb.position.set(x, -hOf(levels[spec[i]][1]) / 2 - 0.05, 0.6)
    blocks.forEach((b, j) => { b.scale.y = 1 + (j === i ? 0.08 : 0) })
    g.rotation.y = Math.sin(t * 0.35) * 0.25
    g.rotation.x = 0.12 + Math.sin(t * 0.2) * 0.04
  }
  return st
}
