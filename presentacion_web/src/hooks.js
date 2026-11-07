import { useEffect, useState } from 'react';
import { routeHash, routeIndex } from './navigation.js';

export function useDeckRoute() {
  const [index, setIndex] = useState(() => routeIndex(window.location.hash));
  useEffect(() => {
    const update = () => setIndex(routeIndex(window.location.hash));
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  useEffect(() => { window.location.hash = routeHash(index); }, [index]);
  return [index, setIndex];
}

export function useMotionPreference() {
  const [reduced, setReduced] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [enabled, setEnabled] = useState(true);
  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  return { active: enabled && !reduced, reduced, toggle: () => setEnabled((value) => !value) };
}

export function useRehearsalTimer() {
  const [clock, setClock] = useState({ running: false, total: 0, started: 0 });
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const update = () => setElapsed(Math.floor((clock.total +
      (clock.running ? performance.now() - clock.started : 0)) / 1000));
    update();
    if (!clock.running) return;
    const interval = setInterval(update, 200);
    return () => clearInterval(interval);
  }, [clock]);
  return {
    elapsed, running: clock.running,
    toggle: () => {
      const now = performance.now();
      setClock((state) => state.running
        ? { running: false, total: state.total + now - state.started, started: 0 }
        : { ...state, running: true, started: now });
    },
    reset: () => setClock({ running: false, total: 0, started: 0 }),
  };
}
