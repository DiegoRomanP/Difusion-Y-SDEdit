# ADR 001 — Comparación controlada en una GPU de 8 GB

## Estado

Aceptado para la implementación solicitada. Los tiempos, memoria real y resultados
quedan pendientes de ejecución por el compañero; no se ha entrenado ningún modelo.

## Contexto

La PC3 del 13 de noviembre de 2026 pide generación de rostros anime mediante difusión.
Se dispone de una RTX 3060 de 8 GB y una noche o más. El código se crea exclusivamente
en `pc3/`, usando el proyecto `uv` inicializado por el usuario. La presentación será
una entrega posterior: objetivo 5:30, máximo 6 minutos, dentro del límite oficial de 7.

## Decisión

- Usar el Anime Face Dataset `splcher/animefacedataset`, versión 3.
- Comparar una U-Net con pesos aleatorios y la misma U-Net con pesos abiertos,
  ajustando todos sus parámetros sobre las mismas particiones anime.
- Candidato fijado: `anton-l/ddpm-ema-flowers-64`, licencia Apache-2.0,
  revisión `dc3af58142513ad2e41e94b57a35ec3a04286a94`.
- La referencia proviene de flores, no de anime. Esto evita el riesgo conocido de
  usar checkpoints anime cuyo preentrenamiento puede solaparse con el test de Kaggle.
- Conservar resolución nativa 64×64, calendario lineal de 1000 pasos y predicción de
  ruido. El bucle de entrenamiento, evaluación, gestión de datos y evidencias es
  código del proyecto; la arquitectura U-Net es de Hugging Face Diffusers.
- Perfil inicial: microbatch 1, acumulación 16, FP16, checkpointing de gradientes,
  entrenamiento secuencial y EMA en CPU. No afirmar que cabe sin medirlo.
- Presupuesto común inicial: 30 000 actualizaciones del optimizador. El test no se
  usa para escoger checkpoint ni hiperparámetros. La selección usa MSE de validación
  con ruido y tiempos fijos; FID/KID evalúan calidad generativa, no sustituyen esa MSE.

## Alternativas y consecuencias

Stable Diffusion a 512×512 introduciría otra tarea, arquitectura y condicionamiento,
además de mayor coste. Un checkpoint anime puede dar muestras mejores de partida,
pero exige verificar qué imágenes vio. La transferencia desde flores **no garantiza**
una mejora; el experimento puede concluir que no ayuda con este presupuesto.

Se iguala entrenamiento local, no cómputo histórico: los pesos abiertos ya fueron
preentrenados. Una sola semilla es exploratoria; la desviación de KID entre
subconjuntos no mide variabilidad entre entrenamientos. La repetición multisemilla
es una ampliación posible, no un resultado supuesto.

El notebook reutiliza los módulos de `src/pc3/`. No se ejecuta durante su creación,
ni se descargan aquí imágenes, pesos o credenciales. Las pruebas de lógica auxiliar
y validación estructural no son evidencias de entrenamiento ni de ajuste en GPU.

## Fuentes

- Curso: `../semana9/Diffusion Models.ipynb` (TensorFlow, CelebA, Huber y DDPM),
  diapositivas Parte IV, y guías de `semana8/` y `semana9/`, respecto a la raíz del curso.
- https://huggingface.co/anton-l/ddpm-ema-flowers-64
- https://docs.astral.sh/uv/guides/integration/pytorch/
- https://huggingface.co/docs/diffusers/training/unconditional_training
- https://arxiv.org/abs/2006.11239
