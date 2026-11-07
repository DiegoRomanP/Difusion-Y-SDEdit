import { useEffect, useState } from 'react';
import data from './content.json';
import { nextIndex, routeHash, routeIndex } from './navigation.js';

export default function App() {
  const [index, setIndex] = useState(() => routeIndex(window.location.hash));
  useEffect(() => {
    const update = () => setIndex(routeIndex(window.location.hash));
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  const go = (direction) => { window.location.hash = routeHash(nextIndex(index, direction)); };
  const slide = data.slides[Math.min(index, 6)];
  return <div className="deck-shell">
    <header className="topbar"><strong>PC3 / DIFUSIÓN</strong><span>Ejecución pendiente</span></header>
    <main className="stage"><h1>{slide.title}</h1><p>{slide.section}</p></main>
    <footer className="controls"><button onClick={() => go(-1)}>Anterior</button>
      <span>{index + 1} / 7</span><button onClick={() => go(1)}>Siguiente</button></footer>
  </div>;
}
