import { animate } from 'animejs'
import { loadHistory, reduceMotion } from '../common.js'

// Curva real de entrenamiento (history.csv): MSE de train y de validación por paso, eje Y logarítmico.
export default (canvas) => {
  let data, drawn = 0, anim
  const ctx = canvas.getContext('2d')
  loadHistory().then((h) => { data = h; draw() })

  function draw() {
    const dpr = Math.min(devicePixelRatio, 2), W = canvas.clientWidth, H = canvas.clientHeight
    if (!W || !H || !data) return
    if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, W, H)
    const L = 64, R = 20, T = 18, B = 34
    const maxStep = data.train.at(-1)[0]
    const lo = Math.log10(0.015), hi = Math.log10(1.2)
    const X = (s) => L + (s / maxStep) * (W - L - R)
    const Y = (v) => T + (1 - (Math.log10(v) - lo) / (hi - lo)) * (H - T - B)
    ctx.font = '15px Inter, system-ui, sans-serif'
    ctx.fillStyle = '#C9BEE6'; ctx.strokeStyle = 'rgba(255,255,255,.08)'; ctx.lineWidth = 1
    for (const v of [0.02, 0.05, 0.1, 0.3, 1]) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(String(v), L - 8, Y(v) + 5) }
    ctx.textAlign = 'center'
    for (const s of [0, 20000, 40000, 60000, 80000]) ctx.fillText(s === 0 ? '0' : `${s / 1000}k`, X(s), H - 10)
    const lim = drawn * maxStep
    const line = (rows, idx, color, width, alpha) => {
      ctx.beginPath(); let first = true
      for (const r of rows) { if (r[0] > lim) break; if (r[idx] == null) continue; const x = X(r[0]), y = Y(r[idx]); first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); first = false }
      ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width; ctx.stroke(); ctx.globalAlpha = 1
    }
    line(data.train, 1, '#4FD1FF', 1.6, 0.55)
    line(data.val, 1, '#FF5C8A', 3.2, 1)
    ctx.textAlign = 'left'
    ctx.fillStyle = '#4FD1FF'; ctx.fillText('● MSE entrenamiento', L + 14, T + 14)
    ctx.fillStyle = '#FF5C8A'; ctx.fillText('● MSE validación (fija)', L + 250, T + 14)
    if (drawn >= 1) { ctx.fillStyle = '#FFC24B'; ctx.font = '700 17px "Space Grotesk Variable", Inter, sans-serif'; ctx.textAlign = 'right'; ctx.fillText('0.02388', W - R - 4, Y(0.0239) - 10) }
  }
  new ResizeObserver(draw).observe(canvas)
  return {
    start() { drawn = reduceMotion() ? 1 : 0; anim?.pause(); if (reduceMotion()) return draw(); const o = { v: 0 }; anim = animate(o, { v: 1, duration: 4200, ease: 'inOutSine', onUpdate: () => { drawn = o.v; draw() } }) },
    stop() { anim?.pause() },
  }
}
