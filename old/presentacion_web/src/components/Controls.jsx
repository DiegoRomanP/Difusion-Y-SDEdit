import { useEffect, useRef } from 'react';
import { ArrowLeft, ArrowRight, BookOpen, Grid2X2, Keyboard, Maximize, Minimize,
  Pause, Play, RotateCcw, Sparkles, X } from 'lucide-react';
import { formatTime } from '../navigation.js';

export function Header({ pending, motion, fullscreen, onFullscreen, open }) {
  return <header className="topbar">
    <a className="brand" href="#/1" aria-label="PC3 / COMPUTACIÓN GRÁFICA, volver a la portada">
      <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
      <span>PC3 <span className="brand-sub">/ COMPUTACIÓN GRÁFICA</span></span>
    </a>
    <span className="status-label"><span className="status-dot" />{pending ? 'Ejecución pendiente' : 'Evidencias incorporadas'}</span>
    <div className="toolbar">
      <button className="icon-button" onClick={() => open('index')} aria-label="Abrir índice" title="Índice [I]"><Grid2X2 size={18} /></button>
      <button className="icon-button" onClick={() => open('notes')} aria-label="Abrir guion" title="Guion [N]"><BookOpen size={18} /></button>
      <button className="icon-button" onClick={motion.toggle} disabled={motion.reduced} aria-pressed={motion.active}
        aria-label={motion.active ? 'Pausar animaciones' : 'Activar animaciones'} title={motion.reduced ? 'Movimiento reducido del sistema' : 'Activar o pausar animaciones'}><Sparkles size={18} /></button>
      <button className="icon-button" onClick={onFullscreen} aria-label={fullscreen ? 'Salir de pantalla completa' : 'Pantalla completa'} title="Pantalla completa [F]">
        {fullscreen ? <Minimize size={18} /> : <Maximize size={18} />}</button>
    </div>
  </header>;
}

export function Footer({ index, slides, timer, go, open }) {
  return <footer className="controls">
    <div className="timer-group">
      <button className="icon-button" onClick={timer.toggle} aria-label={timer.running ? 'Pausar cronómetro' : 'Iniciar cronómetro'} title="Cronómetro [T]">
        {timer.running ? <Pause size={16} /> : <Play size={16} />}</button>
      <span className={`timer ${timer.elapsed >= 360 ? 'over-budget' : ''}`} aria-label="Tiempo transcurrido">{formatTime(timer.elapsed)}</span>
      <span className="timer-budget">/ 05:30</span>
      <button className="icon-button reset-button" onClick={timer.reset} aria-label="Reiniciar cronómetro" title="Reiniciar [R]"><RotateCcw size={14} /></button>
    </div>
    <nav className="slide-navigation" aria-label="Diapositivas principales">
      {slides.map((slide, position) => <button key={position} onClick={() => go(position)}
        className={`step ${position === index ? 'active' : ''}`} aria-current={position === index ? 'step' : undefined}
        aria-label={`Ir a diapositiva ${String(position + 1).padStart(2, '0')}: ${slide.section}`} title={slide.section}>
        <span>{String(position + 1).padStart(2, '0')}</span><i aria-hidden="true" /></button>)}
    </nav>
    <div className="arrow-controls">
      <button className="icon-button help-button" onClick={() => open('help')} aria-label="Ver atajos de teclado" title="Ayuda [H]"><Keyboard size={18} /></button>
      <button className="icon-button" disabled={index === 0} onClick={() => go(index >= 7 ? 6 : index - 1)} aria-label="Anterior"><ArrowLeft size={20} /></button>
      <button className="next-button" disabled={index >= 6} onClick={() => go(index + 1)} aria-label="Siguiente"><span>Siguiente</span><ArrowRight size={20} /></button>
    </div>
  </footer>;
}

export function Overlay({ type, close, slide, slides, go, index }) {
  const ref = useRef(null);
  useEffect(() => {
    if (type && !ref.current.open) ref.current.showModal();
    if (!type && ref.current.open) ref.current.close();
  }, [type]);
  const title = type === 'notes' ? 'Guion del expositor' : type === 'index' ? 'Mapa de la exposición' : 'Controles de la presentación';
  return <dialog ref={ref} onClose={close} className="overlay" aria-labelledby="overlay-title">
    <div className="dialog-header"><div><span className="eyebrow">PC3 / MODO EXPOSITOR</span><h2 id="overlay-title">{title}</h2></div>
      <button className="icon-button" onClick={close} aria-label="Cerrar"><X size={22} /></button></div>
    {type === 'notes' && <div className="notes-content">
      <p className="note-time">{index < 7 ? `Escena ${index + 1} · ${slide.seconds} segundos previstos` : 'Anexo · fuera del recorrido de 5:30'}</p>
      <h3>{slide.title}</h3><p>{slide.speech}</p><aside>{slide.cue}</aside>
      <p className="text-muted text-sm">Duración planificada, no medida. Ensayar con cronómetro.</p>
    </div>}
    {type === 'index' && <div className="index-list">{[...slides, { section: 'Preguntas', seconds: null }, { section: 'Fuentes', seconds: null }].map((entry, position) =>
      <button key={position} onClick={() => { go(position); close(); }}><span>{position < 7 ? String(position + 1).padStart(2, '0') : 'A' + (position - 6)}</span>
        <strong>{entry.section}</strong><small>{entry.seconds ? `${entry.seconds} s` : 'Anexo'}</small><ArrowRight size={16} /></button>)}</div>}
    {type === 'help' && <div className="help-grid">{[
      ['← / →', 'Anterior / siguiente'], ['Espacio', 'Siguiente'], ['1–7', 'Ir a una escena'],
      ['Inicio / Fin', 'Portada / cierre'], ['N', 'Guion del expositor'], ['I', 'Índice y anexos'],
      ['F', 'Pantalla completa'], ['T / R', 'Cronómetro / reiniciar'], ['Esc', 'Cerrar diálogo'],
    ].map(([key, label]) => <div key={key}><kbd>{key}</kbd><span>{label}</span></div>)}</div>}
  </dialog>;
}
