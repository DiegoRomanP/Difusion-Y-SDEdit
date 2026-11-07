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
  const [running, setRunning] = useState(false);
  const [base, setBase] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) return;
    const start = performance.now();
    const interval = setInterval(() => setElapsed(base + Math.floor((performance.now() - start) / 1000)), 200);
    return () => clearInterval(interval);
  }, [running, base]);
  return {
    elapsed, running,
    toggle: () => { if (running) setBase(elapsed); setRunning((value) => !value); },
    reset: () => { setRunning(false); setBase(0); setElapsed(0); },
  };
}
