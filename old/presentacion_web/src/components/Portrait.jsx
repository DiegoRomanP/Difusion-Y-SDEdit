import { useEffect, useRef } from 'react';
import { animate, createScope } from 'animejs';
import { drawPortrait } from '../portrait.js';

export default function Portrait({ noise = 0, motion = false, reconstruct = false }) {
  const canvas = useRef(null);
  useEffect(() => {
    const element = canvas.current;
    element.dataset.ready = 'false';
    const phase = { value: reconstruct && motion ? 0 : 1 };
    const draw = () => drawPortrait(element, reconstruct ? 1 - phase.value : noise, phase.value);
    draw();
    if (!motion || !reconstruct) { element.dataset.ready = 'true'; return; }
    const scope = createScope().add(() => {
      animate(phase, { value: 1, duration: 2200, ease: 'inOut(3)',
        onUpdate: draw, onComplete: () => { element.dataset.ready = 'true'; } });
    });
    return () => scope.revert();
  }, [noise, motion, reconstruct]);
  return <div className="portrait-frame">
    <div className="reticle reticle-one" aria-hidden="true" /><div className="reticle reticle-two" aria-hidden="true" />
    <canvas ref={canvas} width="720" height="720" className="portrait-canvas" role="img"
      aria-label="Ilustración matemática de un rostro pixelado y su corrupción con ruido; no es una muestra del modelo" />
    <span className="coordinate coordinate-top" aria-hidden="true">xₜ · RGB</span>
    <span className="coordinate coordinate-bottom" aria-hidden="true">0.0001 → 0.02</span>
  </div>;
}
