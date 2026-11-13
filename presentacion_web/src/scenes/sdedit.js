import { THREE, createStage, label, glowLine, PAL, ease } from '../common.js'

// Nivel de ruido en el tiempo: SDEdit sube hasta la "fuerza" (proceso directo) y baja de nuevo (muestreo).
export default (canvas) => {
  const st = createStage(canvas, { z: 6.4, y: 0, fov: 40, staticT: 8 })
  const g = new THREE.Group(); st.scene.add(g)
  const X0 = -5, X1 = 5, Y0 = -1.1, H = 2.0
  g.add(glowLine([new THREE.Vector3(X0, Y0, 0), new THREE.Vector3(X1 + 0.2, Y0, 0), new THREE.Vector3(X0, Y0, 0), new THREE.Vector3(X0, Y0 + H + 0.2, 0)], 0xC9BEE6, 0.8))
  const lab = (txt, x, y, c = '#C9BEE6', sc = 0.5) => { const l = label(txt, { scale: sc, size: 46, color: c, weight: 600 }); l.position.set(x, y, 0.1); g.add(l); return l }
  lab('más ruido', -4.1, Y0 + H + 0.1, '#C9BEE6', 0.6); lab('imagen', -4.2, Y0 - 0.3, '#C9BEE6', 0.6)
  const levels = [0.3, 0.6, 0.9], colors = [PAL.cyan, PAL.amber, PAL.coral]
  const tokens = levels.map((s, k) => {
    const w = 2.8, x0 = X0 + 1.2 + k * 3.1, up = new THREE.Vector3(x0 + w * 0.3, Y0 + s * H, 0), peak = up.clone()
    const a = new THREE.Vector3(x0, Y0, 0), b = new THREE.Vector3(x0 + w * 0.6, Y0, 0)
    g.add(glowLine([a, up], PAL.cyan, 0.9), glowLine([up, new THREE.Vector3(up.x + 0.0, up.y, 0), b], PAL.coral, 0.9))
    lab(`${s}`, x0 + w * 0.3, Y0 + s * H + 0.3, '#F3EEFF', 0.65)
    const orb = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 16), new THREE.MeshBasicMaterial({ color: colors[k] }))
    g.add(orb)
    return { orb, a, up, b }
  })
  st.fn = (t) => {
    const u = (t % 8) / 8
    tokens.forEach(({ orb, a, up, b }) => {
      if (u < 0.4) orb.position.lerpVectors(a, up, ease(u / 0.4))
      else if (u < 0.5) orb.position.copy(up)
      else orb.position.lerpVectors(up, b, ease((u - 0.5) / 0.5))
    })
    g.rotation.y = Math.sin(t * 0.3) * 0.12
  }
  return st
}
