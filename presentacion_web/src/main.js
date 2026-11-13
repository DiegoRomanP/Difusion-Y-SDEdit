import Reveal from 'reveal.js'
import Notes from 'reveal.js/plugin/notes'
import 'reveal.js/reveal.css'
import './styles.css'
import { scenes } from './scenes/index.js'
import { transition, BLOCKS } from './transitions.js'
import { initLab } from './lab.js'

const deck = new Reveal({
  width: 1280, height: 720, margin: 0.02, hash: true, controls: true, progress: true, slideNumber: 'c/t',
  embedded: true, transition: 'none', backgroundTransition: 'none', center: false, touch: true, plugins: [Notes],
})

const live = new Map()
function activate(slide) {
  deck.getViewportElement().dataset.block = slide.dataset.block
  document.querySelector('.brand b').textContent = BLOCKS[slide.dataset.block][1]
  const name = slide.dataset.scene
  for (const [n, s] of live) if (n !== name) s.stop()
  if (!name) return
  if (!live.has(name)) live.set(name, scenes[name](slide.querySelector('canvas.viz'), slide))
  live.get(name).start()
}

deck.initialize().then(() => { activate(deck.getCurrentSlide()); transition(null, deck.getCurrentSlide()) })
deck.on('slidechanged', ({ previousSlide, currentSlide, indexh }) => {
  activate(currentSlide)
  transition(previousSlide, currentSlide, indexh >= deck.getIndices(previousSlide).h)
})

initLab()

// El panel de la derecha cambia de ancho: reveal.js vuelve a calcular la escala mientras dura la animación.
let relayout = 0
new ResizeObserver(() => { cancelAnimationFrame(relayout); relayout = requestAnimationFrame(() => deck.layout()) }).observe(document.querySelector('.reveal'))
