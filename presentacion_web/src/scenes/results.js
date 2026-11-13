import { animate, stagger } from 'animejs'
import { reduceMotion } from '../common.js'

// Contadores de métricas y mosaico de muestras reales generadas por el modelo.
export default (canvas, slide) => {
  const mosaic = slide.querySelector('#mosaic')
  for (let i = 0; i < 24; i++) {
    const img = new Image(); img.src = `${import.meta.env.BASE_URL}evidencia/muestras/${String(i).padStart(6, '0')}.png`; img.alt = `Muestra generada ${i + 1}`; img.loading = 'lazy'
    mosaic.append(img)
  }
  const nums = [...slide.querySelectorAll('.n')]
  let anims = []
  const set = () => nums.forEach((n) => { n.textContent = Number(n.dataset.to).toFixed(Number(n.dataset.dec)) })
  return {
    start() {
      anims.forEach((a) => a.pause())
      if (reduceMotion()) return set()
      anims = nums.map((n, i) => { const o = { v: 0 }; return animate(o, { v: Number(n.dataset.to), duration: 2600, delay: 300 + i * 250, ease: 'outExpo', onUpdate: () => { n.textContent = o.v.toFixed(Number(n.dataset.dec)) } }) })
      anims.push(animate(mosaic.children, { opacity: [0, 1], scale: [0.6, 1], delay: stagger(60, { start: 900 }), duration: 1200, ease: 'outQuart' }))
    },
    stop() { anims.forEach((a) => a.pause()) },
  }
}
