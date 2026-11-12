import { ArrowLeft, ArrowUpRight } from 'lucide-react';
import SceneTitle from '../components/SceneTitle.jsx';

export const questions = [
  ['¿Por qué pesos de flores y no de anime?', 'Para reducir el riesgo conocido de que un checkpoint anime haya visto el test. La transferencia entre dominios puede ayudar o no; hay que medirla.'],
  ['¿Fine-tuning es lo mismo que sobreajuste?', 'No. Fine-tuning es adaptar pesos. Sobreajuste es mejorar train sin generalizar a validación. Por eso seleccionamos EMA con validación fija, no test.'],
  ['¿Qué dicen SSIM y los vecinos dHash?', 'SSIM entre generadas aproxima similitud estructural y dHash muestra vecinos de train. Son proxies: ni el ruido diverso demuestra calidad ni un vecino cercano prueba memorización.'],
  ['¿Qué demuestran las pruebas de CPU?', 'Verifican herramientas y controles auxiliares. No prueban que la U-Net quepa en 8 GB, que converja ni que produzca un FID determinado.'],
];

export const sources = [
  ['DDPM', 'Ho, Jain y Abbeel · 2020', 'https://arxiv.org/abs/2006.11239'],
  ['DDIM', 'Song, Meng y Ermon · 2020', 'https://arxiv.org/abs/2010.02502'],
  ['FID / TTUR', 'Heusel et al. · 2017', 'https://arxiv.org/abs/1706.08500'],
  ['KID', 'Bińkowski et al. · 2018', 'https://arxiv.org/abs/1801.01401'],
  ['Anime Face Dataset', 'splcher · Kaggle · versión 3', 'https://www.kaggle.com/datasets/splcher/animefacedataset'],
  ['DDPM de flores', 'anton-l · Apache-2.0', 'https://huggingface.co/anton-l/ddpm-ema-flowers-64'],
];

export function appendixNotes(index) {
  return index === 7 ? { title: 'Preguntas para la discusión', speech: questions.map(([q, a]) => `${q} ${a}`).join('\n\n'), cue: 'Anexo: abrir solo si una pregunta lo requiere.' } :
    { title: 'Fuentes y trazabilidad', speech: sources.map(([title, author, url]) => `${title}. ${author}. ${url}`).join('\n\n'), cue: 'Base docente: semanas 8 y 9. Cambio explícito: Huber → MSE y TensorFlow/CelebA → PyTorch/anime.' };
}

export function Appendix({ index, config, go }) {
  return <div className="appendix-scene">
    <SceneTitle number={index === 7 ? 'A1' : 'A2'} section="Fuera del recorrido principal">{index === 7 ? <>Abramos la <em>discusión.</em></> : <>Fuentes y <em>trazabilidad.</em></>}</SceneTitle>
    {index === 7 ? <div className="questions-list">{questions.map(([question, answer]) =>
      <details className="reveal" key={question}><summary>{question}<span aria-hidden="true">+</span></summary><p>{answer}</p></details>)}</div> :
      <><div className="sources-list">{sources.map(([title, author, url]) => <a href={url} key={url} target="_blank" rel="noopener noreferrer" className="reveal">
        <strong>{title}</strong><span>{author}</span><ArrowUpRight size={18} /></a>)}</div>
        <p className="revision reveal">SHA del checkpoint <code>{config.revision}</code></p>
        <p className="small-note reveal">Semanas 8 y 9 · Huber → MSE · TensorFlow/CelebA → PyTorch/anime</p></>}
    <button className="text-button reveal" onClick={() => go(6)}><ArrowLeft size={17} /> Volver al cierre</button>
  </div>;
}
