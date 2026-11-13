import { THREE, createStage, label, glowLine, PAL, ease } from '../common.js'
import { noisyTexture } from '../faces.js'

// Cadena x0 -> xT con la fórmula del proceso directo; un testigo recorre la ida (ruido) y la vuelta (U-Net).
export default (canvas) => {
  const st = createStage(canvas, { z: 11.5, y: 0, fov: 40, staticT: 3 })
  const abs = [1, 0.8, 0.5, 0.25, 0.07, 0.0]
  const names = ['x₀', 'x₂₀₀', 'x₄₀₀', 'x₆₀₀', 'x₈₀₀', 'x₁₀₀₀']
  const pos = abs.map((_, i) => new THREE.Vector3(-2.5 + (i % 3) * 2.5, i < 3 ? 1.35 : -1.55, 0))
  const planes = abs.map((ab, i) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1.8, 1.8), new THREE.MeshBasicMaterial({ map: noisyTexture(ab, 96, 11 + i) }))
    m.position.copy(pos[i])
    const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(1.86, 1.86)), new THREE.LineBasicMaterial({ color: PAL.glow }))
    frame.position.copy(pos[i])
    const l = label(names[i], { scale: 0.5, size: 62 }); l.position.set(pos[i].x, pos[i].y - 1.2, 0)
    st.scene.add(m, frame, l)
    return m
  })
  const top = label('directo: añade ruido (fórmula)', { scale: 0.5, size: 44, color: '#6FC3FF' }); top.position.set(0, 2.9, 0)
  const bot = label('inverso: la U-Net quita el ruido', { scale: 0.5, size: 44, color: '#A98BFF' }); bot.position.set(0, -3.35, 0)
  st.scene.add(top, bot)
  for (let i = 0; i < 5; i++) st.scene.add(glowLine([pos[i], pos[i + 1]], PAL.violet, 0.9))
  const token = new THREE.Mesh(new THREE.SphereGeometry(0.17, 24, 24), new THREE.MeshBasicMaterial({ color: PAL.cyan }))
  st.scene.add(token)

  st.fn = (t) => {
    const u = (t % 12) / 12
    const fwd = u < 0.5
    const p = (fwd ? ease(u / 0.5) : 1 - ease((u - 0.5) / 0.5)) * 5
    const i = Math.min(4, Math.floor(p))
    token.position.lerpVectors(pos[i], pos[i + 1], p - i); token.position.z = 0.2
    token.material.color.setHex(fwd ? PAL.cyan : PAL.glow)
    planes.forEach((m, k) => m.scale.setScalar(1 + 0.1 * Math.max(0, 1 - Math.abs(p - k))))
  }
  return st
}
