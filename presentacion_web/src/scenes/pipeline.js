import { THREE, createStage, label, glowLine, PAL, lights, ease } from '../common.js'

// Flujo del proyecto: datos -> particiones -> U-Net -> EMA -> DDIM -> FID/KID, y DDIM -> SDEdit.
export default (canvas) => {
  const st = createStage(canvas, { z: 14, y: 0, fov: 42, staticT: 4 })
  lights(st.scene)
  const nodes = [
    ['Datos', '63 565 → 41 522', PAL.steel], ['Particiones', 'dHash · 33 218/4 152', PAL.indigo], ['U-Net', '88.2 M · MSE', 0xB8325C],
    ['EMA', 'decay 0.999', PAL.plum], ['DDIM', '50 pasos', PAL.violet], ['FID · KID', 'test N=4 152', PAL.steel], ['SDEdit', 'imagen → ruido → imagen', 0xA8741A],
  ]
  const pts = [[-1.75, 3], [1.75, 3], [1.75, 1], [-1.75, 1], [-1.75, -1], [1.75, -1], [-1.75, -3]].map(([x, y]) => new THREE.Vector3(x, y, 0))
  const group = new THREE.Group(); st.scene.add(group)
  nodes.forEach(([name, sub, color], i) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(3.1, 1.4, 0.5), new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.45 }))
    m.position.copy(pts[i])
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: PAL.glow }))
    edge.position.copy(pts[i])
    const a = label(name, { scale: 0.66, size: 70 }); a.position.set(pts[i].x, pts[i].y + 0.2, 0.4)
    const b = label(sub, { scale: 0.42, size: 44, color: '#C9BEE6', weight: 500 }); b.position.set(pts[i].x, pts[i].y - 0.3, 0.4)
    group.add(m, edge, a, b)
  })
  // recorrido: 0..5 y de DDIM (4) a SDEdit (6)
  const path = [0, 1, 2, 3, 4, 5, 4, 6].map((i) => pts[i])
  for (let i = 0; i < path.length - 1; i++) group.add(glowLine([path[i], path[i + 1]], PAL.glow, 0.7))
  const orb = new THREE.Mesh(new THREE.SphereGeometry(0.2, 24, 24), new THREE.MeshBasicMaterial({ color: PAL.cyan }))
  group.add(orb)
  st.fn = (t) => {
    const u = ((t % 12) / 12) * (path.length - 1)
    const i = Math.min(path.length - 2, Math.floor(u))
    orb.position.lerpVectors(path[i], path[i + 1], ease(u - i)); orb.position.z = 0.5
    group.rotation.y = Math.sin(t * 0.3) * 0.15
    group.rotation.x = Math.sin(t * 0.22) * 0.05
  }
  return st
}
