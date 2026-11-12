import { ArrowUpRight, Check, Clock3 } from 'lucide-react';
import SceneTitle from '../components/SceneTitle.jsx';

export function Evaluation({ config }) {
  return <div className="evaluation-scene">
    <SceneTitle number="05" section="Evaluación">La calidad<br /><em>no se supone.</em></SceneTitle>
    <div className="evaluation-grid">
      <div className="metric-definitions reveal">
        <div><strong>FID <span>↓</span></strong><p>Distancia entre medias<br />y covarianzas de características.</p></div>
        <div><strong>KID <span>↓</span></strong><p>MMD con kernel polinómico.<br />Media y std de subconjuntos.</p></div>
      </div>
      <div className="protocol-facts reveal">
        <div><span className="eyebrow">MISMO TEST, N Y SEMILLAS</span><strong>Hasta {config.sampleCount.toLocaleString('es-PE')}</strong><p>reales y generadas por caso</p></div>
        <div className="protocol-bottom"><div><span className="eyebrow">MUESTREO COMÚN</span><strong>DDIM {config.samplingSteps}</strong></div><div><span className="eyebrow">COSTE</span><p>s/imagen<br />+ VRAM</p></div></div>
      </div>
    </div>
    <p className="scene-footnote reveal">MSE ≠ calidad visual. Inception tiene límites en anime. La std de KID no es un IC entre entrenamientos.</p>
  </div>;
}

export function Results({ evidence }) {
  const pending = evidence.length === 0;
  return <div className="results-scene">
    <SceneTitle number="06" section="Resultados">Aquí manda <em>la evidencia.</em></SceneTitle>
    <p className="results-subtitle reveal">{pending ? 'Comparación preparada. Muestras y métricas todavía por ejecutar.' :
      `Test · N=${evidence[0].metrics.protocol.num_images} · Cuatro primeras semillas por caso, sin selección manual.`}</p>
    <div className="results-grid">
      {['Desde cero', 'Abierto sin adaptar', 'Fine-tuning'].map((label, index) => {
        const row = evidence[index]?.metrics;
        return <article className={`result-case result-case-${index} reveal`} key={label}>
          <h2><span>{'ABC'[index]}</span>{label}</h2>
          {pending ? <div className="sample-placeholder"><span aria-hidden="true">[ — ]</span><p>Muestras pendientes</p></div> :
            <div className="sample-grid">{evidence[index].images.map((image, number) => <img key={number} src={image} alt={`${label}: muestra de la semilla predefinida ${number + 1}`} width="64" height="64" />)}</div>}
          <dl className="result-metrics">{[
            ['FID ↓', row?.fid.toFixed(2)],
            ['KID ↓', row ? `${row.kid_mean.toPrecision(3)} ± ${row.kid_std.toPrecision(3)}` : null],
            ['s/imagen ↓', row?.seconds_per_image.toFixed(3)],
          ].map(([metric, value]) => <div key={metric}><dt>{metric}</dt><dd>{value ?? '—'}</dd></div>)}</dl>
          <p className="result-provenance">{row ? `Pesos seleccionados: paso ${row.provenance.step}` : 'Sin checkpoint evaluado'}</p>
        </article>;
      })}
    </div>
    <p className="scene-footnote reveal">{pending ? '— significa «no medido», no cero. Aún no existe evidencia para elegir un ganador.' :
      'KID: media ± std entre subconjuntos, no intervalo de confianza. Una ejecución no demuestra superioridad universal.'}</p>
  </div>;
}

export function Closing({ pending, go }) {
  return <div className="closing-grid">
    <div>
      <SceneTitle number="07" section="Conclusión" className="closing-title">Adaptar pesos<br />no garantiza<br /><em>mejores rostros.</em></SceneTitle>
      <p className="closing-summary reveal">Controlar datos, arquitectura y presupuesto<br />permite estudiar el efecto de la inicialización.</p>
      <button className="text-button reveal" onClick={() => go(7)}>Abrir la discusión <ArrowUpRight size={20} /></button>
    </div>
    <div className="takeaways">
      <div className="reveal"><Check size={21} /><span className="eyebrow">{pending ? 'IMPLEMENTADO' : 'OBSERVADO'}</span><h2>{pending ? <>Código reproducible.<br />Protocolo compartido.</> : <>Muestras y métricas.<br />Presupuesto local común.</>}</h2></div>
      <div className="reveal"><Clock3 size={21} /><span className="eyebrow">{pending ? 'POR MEDIR' : 'POR FORTALECER'}</span><h2>{pending ? <>Memoria y tiempos.<br />Calidad y diversidad.</> : <>Variabilidad entre semillas.<br />Diversidad y generalización.</>}</h2></div>
      <p className="small-note reveal">{pending ? 'Una hipótesis bien controlada; todavía no una conclusión experimental.' : 'La conclusión es local y exploratoria, no un ganador universal.'}</p>
    </div>
  </div>;
}
