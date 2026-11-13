import { animate, stagger } from 'animejs'
import { reduceMotion } from './common.js'

const ORDER = { intro: 0, proceso: 1, cierre: 2 }
const curtain = Object.assign(document.createElement('div'), { id: 'curtain' })
Object.assign(curtain.style, { position: 'fixed', inset: 0, zIndex: 50, pointerEvents: 'none', opacity: 0 })
document.body.append(curtain)

/** Transición lenta entre diapositivas: dentro de un bloque, fundido y entrada escalonada; entre bloques, barrido de color más largo. */
export function transition(prev, next, forward) {
  if (reduceMotion() || !prev) return
  const jump = prev.dataset.block !== next.dataset.block
  const dist = Math.abs(ORDER[next.dataset.block] - ORDER[prev.dataset.block])
  const dur = jump ? 1700 + dist * 300 : 1150
  curtain.style.background = jump
    ? `linear-gradient(${forward ? 90 : 270}deg, #19082D 0%, #46194A 45%, #5B3EA4 100%)`
    : '#19082D'
  if (jump) {
    animate(curtain, { opacity: [1, 1], translateX: [forward ? '0%' : '0%', forward ? '105%' : '-105%'], duration: dur, ease: 'inOutQuart' })
  } else {
    animate(curtain, { opacity: [0.92, 0], translateX: '0%', duration: dur, ease: 'inOutSine' })
  }
  const parts = next.querySelectorAll('.copy > *, .split > .viz, .results .gallery, .gallery > *, .kpis > *, .why > *, .points > *')
  animate(parts, {
    opacity: [0, 1],
    translateY: [jump ? 46 : 26, 0],
    filter: ['blur(10px)', 'blur(0px)'],
    delay: stagger(jump ? 110 : 80, { start: jump ? 500 : 250 }),
    duration: dur + 300,
    ease: 'outExpo',
  })
}
