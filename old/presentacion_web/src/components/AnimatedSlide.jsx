import { useLayoutEffect, useRef } from 'react';
import { animate, createScope, stagger } from 'animejs';

export default function AnimatedSlide({ children, motion, index }) {
  const root = useRef(null);
  useLayoutEffect(() => {
    root.current.dataset.ready = motion ? 'false' : 'true';
    if (!motion) return;
    const element = root.current;
    const scope = createScope({ root }).add(() => {
      animate('.reveal', { opacity: [0, 1], y: [26, 0], duration: 640, delay: stagger(75),
        ease: 'out(3)', onComplete: () => { element.dataset.ready = 'true'; } });
      if (element.querySelector('.draw-line')) {
        animate('.draw-line', { strokeDashoffset: [1, 0], duration: 1250, delay: stagger(45), ease: 'out(2)' });
      }
    });
    return () => scope.revert();
  }, [motion]);
  return <section className={`slide scene-${index + 1}`} ref={root} aria-labelledby="slide-title" data-slide={index + 1}>{children}</section>;
}
