import { useEffect, useRef, useState } from 'react';
import data from './content.json';
import { nextIndex } from './navigation.js';
import { useDeckRoute, useMotionPreference, useRehearsalTimer } from './hooks.js';
import { Header, Footer, Overlay } from './components/Controls.jsx';
import AnimatedSlide from './components/AnimatedSlide.jsx';
import { Opening, Diffusion } from './scenes/Opening.jsx';
import { Experiment, Training } from './scenes/Experiment.jsx';
import { Evaluation, Results, Closing } from './scenes/Results.jsx';
import { Appendix, appendixNotes } from './scenes/Appendices.jsx';

export default function App() {
  const [index, go] = useDeckRoute();
  const [dialog, setDialog] = useState(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [notice, setNotice] = useState('');
  const motion = useMotionPreference();
  const timer = useRehearsalTimer();
  const stage = useRef(null);
  const slide = index < 7 ? data.slides[index] : appendixNotes(index);
  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
      else setNotice('Usa la pantalla completa del navegador, por ejemplo F11.');
    } catch { setNotice('No se pudo activar pantalla completa. Usa F11 o el menú del navegador.'); }
  };
  useEffect(() => {
    const update = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener('fullscreenchange', update);
    return () => document.removeEventListener('fullscreenchange', update);
  }, []);
  useEffect(() => {
    stage.current.scrollTop = 0;
    stage.current.querySelector('h1')?.focus({ preventScroll: true });
  }, [index]);
  useEffect(() => {
    const onKey = (event) => {
      if (dialog || event.altKey || event.ctrlKey || event.metaKey ||
          event.target.closest('input,textarea,select,[contenteditable="true"]')) return;
      const key = event.key.toLowerCase();
      if (key === ' ' && event.target.closest('button,a,summary')) return;
      const actions = {
        arrowright: () => go(nextIndex(index, 1)), ' ': () => go(nextIndex(index, 1)),
        arrowleft: () => go(nextIndex(index, -1)), home: () => go(0), end: () => go(6),
        n: () => setDialog('notes'), i: () => setDialog('index'),
        h: () => setDialog('help'), '?': () => setDialog('help'),
        f: toggleFullscreen, t: timer.toggle, r: timer.reset,
      };
      const action = /^[1-7]$/.test(key) ? () => go(Number(key) - 1) : actions[key];
      if (action) { event.preventDefault(); action(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [index, dialog, timer.toggle, timer.reset]);
  return <div className="deck-shell" data-motion={motion.active ? 'on' : 'off'}>
    <Header pending={data.status === 'pending_results'} motion={motion} fullscreen={fullscreen} onFullscreen={toggleFullscreen} open={setDialog} />
    <main className="stage" ref={stage}>
      <AnimatedSlide key={index} motion={motion.active} index={index}>
        {index === 0 && <Opening motion={motion.active} />}
        {index === 1 && <Diffusion config={data.config} />}
        {index === 2 && <Experiment config={data.config} />}
        {index === 3 && <Training config={data.config} />}
        {index === 4 && <Evaluation config={data.config} />}
        {index === 5 && <Results evidence={data.evidence} />}
        {index === 6 && <Closing pending={data.status === 'pending_results'} go={go} />}
        {index > 6 && <Appendix index={index} config={data.config} go={go} />}
      </AnimatedSlide>
    </main>
    <Footer index={index} slides={data.slides} timer={timer} go={go} open={setDialog} />
    <Overlay type={dialog} close={() => setDialog(null)} slide={slide} slides={data.slides} go={go} index={index} />
    {notice && <div className="notice" role="status">{notice}<button onClick={() => setNotice('')} aria-label="Cerrar aviso">×</button></div>}
  </div>;
}
