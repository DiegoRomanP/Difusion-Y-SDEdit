# Protocolo experimental y lectura de evidencias

Vigente según [ADR 002](decisions/002-cascada-superresolucion.md). La comparación de
inicializaciones que describía [ADR 001](decisions/001-experimento-controlado.md) está
retirada; sus evidencias se conservan como referencia histórica, no como brazos.

## Pregunta y etapas

La pregunta es qué imagen se puede producir, con un presupuesto local dado, mediante
una cascada de dos etapas. No se compara el efecto de la inicialización.

| Etapa | Qué hace | Cómputo |
|---|---|---|
| 1 — Difusión | DDPM propio a 64×64, arquitectura libre | Entrenamiento local |
| 2 — Superresolución | Real-ESRGAN anime ×4 → 256×256 | Preentrenado externo, sin entrenar |

El contraste controlado está **dentro de la etapa 2**: del mismo conjunto de muestras
de 64 px se obtienen el brazo **A** (Real-ESRGAN) y el brazo **B** (bicúbica). Mismo
equipo, mismas semillas, mismo conteo, sin entrenamiento adicional. Es la única
comparación que este protocolo declara comparable sin reservas.

Se declara el cómputo externo: el upsampler ya fue entrenado por terceros y su historia
no entra en el tiempo local, igual que antes se declaraba el del checkpoint de flores.
La etapa 2 **no añade información**: redistribuye la que trae la etapa 1 y puede volver
nítido un artefacto. No se presenta como mejora de la calidad del modelo propio.

## Datos y prevención de fuga

- Kaggle: `splcher/animefacedataset/versions/3`. 63 565 archivos descargados.
- Recorte central RGB64 con Lanczos, seguido de normalización `[-1,1]`.
- SHA-256 de píxeles canónicos para excluir duplicados exactos: 22 043 excluidos.
- dHash de 64 bits, tomando el mínimo del original y su espejo, para agrupar
  posibles casi duplicados con hash idéntico. No es deduplicación perceptual exhaustiva.
- Distribución por grupos: **33 218 train / 4 152 val / 4 152 test** (41 522 efectivas).
- Solo train recibe aumentos. Los píxeles se verifican contra el manifiesto al leerlos.
- El manifiesto no contiene rutas absolutas ni credenciales.

`data/manifest.json` **identifica el experimento**: su SHA fija qué imágenes quedaron en
test y, por tanto, contra qué se mide FID/KID. Debe viajar con el código a cualquier
equipo ejecutor y no regenerarse allí. Regenerarlo en otra máquina puede mover al train
imágenes que estaban en el test de una ejecución anterior; eso es contaminación, no una
diferencia de configuración. Está versionado a propósito, aunque `.gitignore` excluya
el resto de `data/`.

## Arquitectura y perfiles

La arquitectura forma parte del perfil y deja de estar fijada en el código. El perfil
vigente usa 4 bloques, `block_out_channels` `[128, 256, 384, 512]`, autoatención a
**16×16 y 8×8**, `attention_head_dim = 64` y `norm_num_groups = 32`. ADR 002 explica
por qué la anterior (atención únicamente a 4×4, ancho de cabeza 8) limitaba la calidad.

`run.json` registra el hash de la arquitectura además del contrato, el manifiesto, el
hardware, las versiones y el SHA del código. Cambiar la arquitectura ya no aborta una
comparación: queda anotada como divergencia.

## Selección y reanudación

La MSE de validación se mide sobre hasta 256 imágenes fijas, con ruido y tiempos
reproducibles. Se selecciona `best.safetensors` por la menor MSE de validación, no
por FID de test. Menor MSE no es garantía de mejor percepción. El informe distingue
`checkpoint_step` de las actualizaciones totales completadas.

`latest.pt` guarda pesos crudos, EMA, AdamW, GradScaler, RNG, posición e historial.
`best.safetensors` y `last.safetensors` contienen pesos EMA y procedencia dentro del
archivo, no el estado necesario para continuar entrenando. El sampler usa posiciones
absolutas; DataLoader y ruido tienen generadores separados del RNG del forward.

`Ctrl+C` y SIGTERM solicitan terminar el update actual y guardar. SIGKILL, OOM o pérdida
de energía pueden perder trabajo desde el último `latest.pt`. Nunca se promete guardar
un acumulado parcial de gradientes. La reanudación comprueba software, hardware,
configuración, dataset y SHA del código/assets. Cambiar README o notebook no altera
esa huella; cambiar módulos sí.

## Generación, superresolución y evaluación

1. Terminar el entrenamiento o declarar expresamente que la etapa 1 es parcial.
2. Generar con DDIM 50 y semillas fijas a 64 px. Esa es la salida canónica.
3. Calcular **FID/KID en test a 64×64**, con el mismo N disponible, hasta 5000.
4. Derivar los brazos A y B del mismo conjunto de 64 px, sin regenerar muestras.
5. Informe y figuras; las divergencias se imprimen y quedan en `comparison.json`.

**FID/KID se calculan solo a 64×64.** El dataset es nativo de 64 px: no existe verdad
de terreno a 256. Reescalar también las reales mediría el parecido con reales
reescaladas por el mismo procedimiento, no realismo; si alguna vez se reporta, debe
decirse eso exactamente. La etapa 2 se juzga por el contraste A/B y por inspección
visual con las mismas semillas, declarada como inspección y sin selección favorable.

FID usa características Inception de 2048 dimensiones. KID usa el mismo tipo de
extractor, kernel polinómico y subconjuntos con semilla fija. Inception proviene de
imágenes naturales y tiene validez limitada en anime. FID depende del tamaño muestral;
la desviación KID es entre subconjuntos, no entre entrenamientos ni un intervalo de
confianza. KID puede ser ligeramente negativo por estimación.

La inferencia se mide después del warmup y con sincronización CUDA. Incluye muestreo
y transferencia final de tensores, no carga/descarga de pesos ni escritura PNG/GIF.
`seconds_per_image` es tiempo amortizado, no latencia individual cuando el lote es
mayor que uno. VRAM se expresa en GiB de memoria asignada máxima de PyTorch, no toda
la reservada ni el consumo global de `nvidia-smi`. El coste de la etapa 2 se mide y
se informa aparte: no se suma al del muestreo como si fuera un solo número.

El tiempo de entrenamiento registrado incluye validaciones, previews y guardado
durante el bucle; excluye inicialización y carga previas. Una calibración sintética
no mide lectura de datos y no demuestra convergencia.

## Comparabilidad: divergencias y el único límite duro

Contrato de entrenamiento, hardware, presupuesto, muestreador y N divergentes **se
registran, no abortan el informe**. Quedan en `comparison.json` bajo `divergences`, se
imprimen al ejecutar y deben aparecer en cualquier figura o diapositiva que enfrente
esos números. Rechazar la salida ocultaba la limitación; anotarla la pone a la vista.
`--strict` restaura el rechazo para la entrega final.

El límite duro es otro: `manifest_sha256` y `real_subset_sha256` deben coincidir entre
el entrenamiento de un modelo y el test con el que se lo evalúa, y entre brazos
comparados. Eso no describe comparabilidad sino contaminación, y sigue siendo error.

Cruzar equipos: la calidad a 64 px es comparable en cuanto se comparta partición de
test, muestreador y N. **El coste nunca lo es**: segundos por imagen y VRAM de GPUs
distintas no se enfrentan, se informan por separado con el equipo indicado.

## Ablaciones y diagnósticos

- Estudiar 25/50/100 pasos DDIM con los mismos pesos exclusivamente en **val**, si se
  usará para decidir. Después fijar un único protocolo para la evaluación final.
- `ablation-report` exige mismos pesos, hardware y configuración salvo número de pasos;
  rechaza test.
- SSIM entre generadas es un proxy de diversidad estructural, no fidelidad. Ruido puede
  parecer diverso. No se compara contra reales arbitrarias.
- dHash contra train y hojas de vecinos ayudan a inspeccionar similitud gruesa; no
  prueban memorización ni su ausencia. Revisar las imágenes.
- Los diagnósticos se corren sobre las muestras de 64 px, no sobre las reescaladas:
  la etapa 2 altera la textura y haría ilegible el proxy de diversidad.
- Una sola semilla permite conclusiones exploratorias, no afirmaciones de robustez.

## Evidencia histórica de la RTX 3060

En una RTX 3060 de 12 GB se completaron tres brazos de 30 000 updates y se evaluaron en
test con N=4152 y DDIM 50. Resultados en `reports/`:

| Brazo | FID | KID | s/imagen | Horas locales |
|---|---:|---:|---:|---:|
| Desde cero | 24.902 | 0.018887 | 0.1706 | 3.08 |
| Abierto sin adaptar | 302.603 | 0.354365 | 0.1706 | 0 |
| Fine-tuning | 21.306 | 0.015165 | 0.1707 | 3.07 |

Esos números **son reales** y comparten partición de test, muestreador y N con lo que
venga después, así que sirven como referencia de calidad a 64 px. No son un brazo del
experimento vigente, su coste pertenece a otra GPU, y la arquitectura que los produjo
es la que ADR 002 retira. Citarlos exige decir las tres cosas.

## Estado verificable

Ejecutado y comprobado: descarga y preparación del dataset, los tres entrenamientos de
ADR 001 en GPU, muestreo, FID/KID en test, diagnósticos, ablación de pasos en val e
informes con figuras. La ablación dio 34.819 / 34.356 / 37.442 de FID para 25/50/100
pasos sobre 1000 imágenes de val.

Pendiente: la arquitectura liberada, la etapa de superresolución, los perfiles con
arquitectura, el registro de divergencias y la ejecución en la RTX 4070 Ti. Nada de eso
tiene todavía evidencia. El notebook y la presentación siguen describiendo la
comparación retirada y se reescriben cuando existan los resultados nuevos.
