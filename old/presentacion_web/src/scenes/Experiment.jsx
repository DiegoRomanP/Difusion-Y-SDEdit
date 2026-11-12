import { ArrowRight, Cpu, Layers, MemoryStick } from 'lucide-react';
import SceneTitle from '../components/SceneTitle.jsx';

const number = (value) => value.toLocaleString('es-PE');

export function Experiment({ config }) {
  return <div className="experiment-grid">
    <div className="experiment-intro">
      <SceneTitle number="03" section="Comparación controlada">Misma red.<br /><em>Distinto inicio.</em></SceneTitle>
      <p className="body-copy reveal">Anime Face Dataset <span className="text-muted">/ v3</span><br />Recorte central, RGB 64 × 64.</p>
      <div className="split-diagram reveal" aria-label="Reparto previsto: aproximadamente 80 por ciento train, 10 validación y 10 test por grupos">
        <div className="split-track"><span style={{ flex: 8 }}>80%</span><span style={{ flex: 1 }}>10</span><span style={{ flex: 1 }}>10</span></div>
        <div className="split-labels"><span>TRAIN</span><span>VAL</span><span>TEST</span></div>
      </div>
      <p className="small-note reveal">SHA-256 + grupos dHash/espejo.<br />Ningún grupo cruza particiones. Reparto aproximado.</p>
    </div>
    <div className="case-list">
      {[
        ['A', 'Desde cero', 'Pesos aleatorios', `${number(config.maxSteps)} updates previstos`],
        ['B', 'Abierto sin adaptar', 'Pesos de flores', '0 updates sobre anime'],
        ['C', 'Fine-tuning completo', 'Los mismos pesos de B', `${number(config.maxSteps)} updates previstos`],
      ].map(([letter, title, initial, budget]) => <article className={`case-row case-${letter.toLowerCase()} reveal`} key={letter}>
        <span className="case-letter">{letter}</span><div><h2>{title}</h2><p>{initial}</p><span className="case-budget">{budget}</span></div><ArrowRight size={23} />
      </article>)}
      <p className="small-note reveal">A y C comparten datos, arquitectura y presupuesto local.<br />El preentrenamiento externo de B/C se declara, no se iguala.</p>
    </div>
  </div>;
}

export function Training({ config }) {
  return <div className="training-grid">
    <div>
      <SceneTitle number="04" section="Diseñado para 8 GB">Menos memoria.<br /><em>Mismo experimento.</em></SceneTitle>
      <div className="batch-equation reveal"><span>{config.batchSize}</span><i>×</i><span>{config.accumulation}</span><i>=</i><span className="text-signal">{config.batchSize * config.accumulation}</span></div>
      <div className="batch-labels reveal"><span>microbatch</span><span>acumulación</span><span>lote efectivo</span></div>
      <p className="small-note reveal">Una GPU, dos entrenamientos secuenciales.<br />El mismo orden de datos, objetivo y optimizador.</p>
    </div>
    <div className="memory-panel">
      <div className="budget reveal"><span className="eyebrow">POR MODELO ENTRENADO</span><strong>{number(config.maxSteps)}</strong><span>actualizaciones previstas</span></div>
      <div className="memory-methods reveal">{[
        [Cpu, 'FP16 + GradScaler', 'Menor memoria de activaciones.'],
        [Layers, 'Gradient checkpointing', 'Recalcular para ahorrar VRAM.'],
        [MemoryStick, 'EMA en CPU', 'Pesos promediados fuera de la GPU.'],
      ].map(([Icon, title, detail]) => <div key={title}><Icon size={21} /><p><strong>{title}</strong><span>{detail}</span></p></div>)}</div>
      <p className="validation-note reveal">Elegir EMA por <strong>MSE de validación fija</strong>.<br />Test queda reservado para evaluar.</p>
    </div>
    <p className="scene-footnote training-footnote reveal">Perfil previsto: falta calibrar VRAM y tiempo en la GPU real. No se promete completar todo en una noche.</p>
  </div>;
}
