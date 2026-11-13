import { animate, stagger, utils } from 'animejs'
import { reduceMotion } from './common.js'

export const BLOCKS = { intro: ['01', 'Introducción', '#4FD1FF'], proceso: ['02', 'Proceso', '#FF5C8A'], cierre: ['03', 'Cierre', '#FFC24B'] }

const curtain = Object.assign(document.createElement('div'), { id: 'curtain', innerHTML: '<p><small></small><span></span></p>' })
Object.assign(curtain.style, { position: 'absolute', inset: 0, zIndex: 50, pointerEvents: 'none', opacity: 0 })
document.querySelector('.reveal').append(curtain)

// Número de diapositiva en el eyebrow (no sirve counter(): reveal.js oculta con display:none las lejanas).
document.querySelectorAll('.slides > section').forEach((s, i) => s.querySelector('.eyebrow')?.setAttribute('data-n', String(i + 1).padStart(2, '0')))

// Cada palabra de los títulos va en una máscara (.w) para entrar desde abajo.
for (const h of document.querySelectorAll('.slides h1, .slides h2')) {
  const walker = document.createTreeWalker(h, NodeFilter.SHOW_TEXT), texts = []
  while (walker.nextNode()) texts.push(walker.currentNode)
  for (const node of texts) {
    const frag = document.createDocumentFragment()
    for (const part of node.textContent.split(/(\s+)/)) {
      if (!part.trim()) { frag.append(part); continue }
      const w = document.createElement('span'), s = document.createElement('span')
      w.className = 'w'; s.textContent = part; w.append(s); frag.append(w)
    }
    node.replaceWith(frag)
  }
}

const TEXT = '.lead, .points > li, .chips > li, .formula, .note, .cmp thead tr, .cmp tbody tr, .kpis > div, .tag'
const MEDIA = '.split > .viz, .copy > .viz, .stack > .viz, .strip img, .gallery > figure, .sdrow'

/** Entrada de una diapositiva. Dentro de un bloque: fundido corto. Entre bloques: cortina del color del bloque con su nombre. */
export function transition(prev, next, forward = true) {
  if (reduceMotion()) return
  const jump = prev && prev.dataset.block !== next.dataset.block
  const start = jump ? 700 : 120
  utils.remove(curtain)
  if (jump) {
    const [num, name, color] = BLOCKS[next.dataset.block]
    curtain.querySelector('small').textContent = `Bloque ${num}`
    curtain.querySelector('span').textContent = name
    curtain.style.background = `radial-gradient(circle at 50% 50%, ${color}, color-mix(in srgb, ${color} 55%, #120621))`
    utils.set(curtain, { opacity: 1, clipPath: 'inset(0% 0% 0% 0%)' })
    animate(curtain.firstChild, { opacity: [0, 1], scale: [0.92, 1], duration: 500, ease: 'outQuart' })
    animate(curtain, { clipPath: forward ? 'inset(0% 0% 0% 100%)' : 'inset(0% 100% 0% 0%)', delay: 520, duration: 900, ease: 'inOutQuart' })
  } else if (prev) {
    utils.set(curtain, { clipPath: 'inset(0% 0% 0% 0%)', background: '#120621' })
    animate(curtain, { opacity: [0.7, 0], duration: 600, ease: 'outSine' })
  }

  const q = (sel) => [...next.querySelectorAll(sel)]
  const eyebrow = q('.eyebrow'), words = q('.w > span'), text = q(TEXT), media = q(MEDIA), bg = q('.viz.bg')
  // anime.js avisa si la lista de objetivos está vacía
  const go = (t, p) => t.length && animate(t, p)
  if (text.length) utils.set(text, { opacity: 0 })
  if (media.length) utils.set(media, { opacity: 0, clipPath: 'inset(0% 0% 100% 0%)' })
  utils.set(eyebrow, { opacity: 0 }); utils.set(words, { translateY: '110%' })
  go(eyebrow, { opacity: [0, 1], translateX: [-24, 0], delay: start, duration: 900, ease: 'outExpo' })
  go(words, { translateY: ['110%', '0%'], rotate: [4, 0], delay: stagger(55, { start: start + 120 }), duration: 1100, ease: 'outExpo' })
  go(text, { opacity: [0, 1], translateY: [22, 0], delay: stagger(70, { start: start + 380 }), duration: 1000, ease: 'outQuart' })
  go(media, { opacity: [0, 1], clipPath: ['inset(0% 0% 100% 0%)', 'inset(0% 0% 0% 0%)'], scale: [1.03, 1], delay: stagger(90, { start: start + 260 }), duration: 1300, ease: 'inOutQuart' })
  go(bg, { opacity: [0, 1], delay: start, duration: 1800, ease: 'outSine' })
}
