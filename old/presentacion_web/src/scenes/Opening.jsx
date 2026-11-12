import { useState } from 'react';
import { ArrowDownRight, ArrowRight, RotateCcw } from 'lucide-react';
import Portrait from '../components/Portrait.jsx';
import SceneTitle from '../components/SceneTitle.jsx';

export function Opening({ motion }) {
  const [replay, setReplay] = useState(0);
  return <div className="hero-grid">
    <div className="hero-copy">
      <SceneTitle number="01" section="La pregunta" className="cover-title">Del ruido<br />al <em>rostro.</em></SceneTitle>
      <p className="hero-description reveal">Generación de rostros anime<br />con modelos de difusión.</p>
      <div className="hero-thesis reveal"><span className="small-cross" aria-hidden="true">+</span><p>¿Aprender desde cero<br />o adaptar lo aprendido?</p></div>
      <div className="hero-tags reveal"><span>PyTorch</span><span>U-Net</span><span>RGB 64 × 64</span></div>
    </div>
    <figure className="hero-art reveal">
      <div className="figure-label"><span>FIG. 01 / DE RUIDO A ESTRUCTURA</span><ArrowDownRight size={20} /></div>
      <Portrait key={replay} reconstruct motion={motion} />
      <figcaption className="art-caption"><span>Ilustración conceptual.<br /><strong>No es una muestra del modelo.</strong></span>
        <button className="icon-button" disabled={!motion} onClick={() => setReplay((value) => value + 1)} aria-label="Repetir reconstrucción" title="Repetir reconstrucción"><RotateCcw size={16} /></button></figcaption>
    </figure>
  </div>;
}

function UNetDiagram() {
  return <svg viewBox="0 0 380 230" role="img" aria-label="Esquema U-Net: encoder, cuello de botella, decoder y conexiones skip" className="unet-svg">
    {Array.from({ length: 6 }, (_, i) => {
      const left = 34 + i * 18, right = 306 - i * 18, y = 18 + i * 32;
      return <g key={i}>
        <path className="draw-line skip-path" pathLength="1" d={`M${left + 30},${y + 12}H${right}`} />
        {i < 5 && <><path className="draw-line net-path" pathLength="1" d={`M${left + 15},${y + 24}L${left + 33},${y + 32}`} />
          <path className="draw-line net-path" pathLength="1" d={`M${right - 3},${y + 32}L${right + 15},${y + 24}`} /></>}
        <rect x={left} y={y} width="30" height="24" rx="3" className="net-block encoder" />
        <rect x={right} y={y} width="30" height="24" rx="3" className="net-block decoder" />
        <text x="5" y={y + 16}>{64 / (2 ** i)}</text>
      </g>;
    })}
    <path className="draw-line net-path" pathLength="1" d="M154 190H216" />
    <text x="44" y="225">encoder</text><text x="277" y="225">decoder</text>
    <text x="159" y="16" className="skip-label">skip</text>
  </svg>;
}

export function Diffusion({ config }) {
  const [time, setTime] = useState(450);
  return <div className="diffusion-scene">
    <SceneTitle number="02" section="La idea">Aprender a <em>quitar ruido.</em></SceneTitle>
    <div className="diffusion-grid">
      <figure className="noise-demo reveal">
        <Portrait noise={time / 999} />
        <figcaption><label htmlFor="noise-level">Nivel de ruido <span className="mono" aria-hidden="true">t = {String(time).padStart(3, '0')}</span></label>
          <input id="noise-level" type="range" min="0" max="999" value={time} onChange={(event) => setTime(Number(event.target.value))} />
          <div className="range-labels"><span>Imagen x₀</span><span>Ruido xT</span></div></figcaption>
      </figure>
      <div className="noise-explanation reveal">
        <div className="formula" aria-label="x sub t es raíz de alfa barra sub t por x cero, más raíz de uno menos alfa barra sub t por épsilon">
          x<sub>t</sub> = √ᾱ<sub>t</sub> x<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) ε
        </div>
        <p>La U-Net recibe <strong>x<sub>t</sub> y t</strong>.<br />Aprende a predecir el ruido <strong>ε</strong>.</p>
        <UNetDiagram />
        <div className="method-strip"><span>MSE de ruido</span><ArrowRight size={16} /><span>DDIM {config.samplingSteps} al generar</span></div>
      </div>
    </div>
    <p className="scene-footnote reveal">Un t aleatorio por imagen al entrenar, no 1000 forwards. El dibujo es didáctico, no una salida entrenada.</p>
  </div>;
}
