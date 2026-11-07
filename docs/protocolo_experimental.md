# Protocolo experimental y lectura de evidencias

## Pregunta y controles

La pregunta es si el preentrenamiento ayuda, con un presupuesto local dado, a generar
rostros anime. No se comparan dos arquitecturas ni se garantiza que el fine-tuning gane.

| Caso | Inicialización | Actualizaciones anime | Pesos evaluados |
|---|---|---:|---|
| Desde cero | Aleatoria | Perfil común | EMA seleccionada por validación |
| Referencia | DDPM abierto de flores | 0 | Checkpoint público fijado |
| Fine-tuning | La misma referencia | Perfil común | EMA seleccionada por validación |

Ambos entrenamientos usan la misma U-Net, particiones, recorte, flip, ruido/tiempos,
semilla, objetivo, AdamW, warmup/coseno, clipping, EMA y presupuesto. Se ejecutan uno
después del otro en la misma GPU. El lote efectivo es `batch_size * accumulation_steps`.
El perfil inicial contempla 30 000 updates, es decir, 480 000 exposiciones de imágenes
por modelo con lote efectivo 16. No son 30 000 épocas ni 480 000 imágenes distintas.

Se declara el cómputo externo previo: los pesos abiertos ya fueron entrenados en
flores. El tiempo local no incluye esa historia. Elegir pesos de origen distinto
reduce el riesgo de evaluar un checkpoint anime sobre imágenes que ya conocía.

## Datos y prevención de fuga

- Kaggle: `splcher/animefacedataset/versions/3`.
- Recorte central RGB64 con Lanczos, seguido de normalización `[-1,1]`.
- SHA-256 de píxeles canónicos para excluir duplicados exactos.
- dHash de 64 bits, tomando el mínimo del original y su espejo, para agrupar
  posibles casi duplicados con hash idéntico. No es deduplicación perceptual exhaustiva.
- Distribución por grupos, aproximadamente 80/10/10; registrar los conteos efectivos.
- Solo train recibe aumentos. Los píxeles se verifican contra el manifiesto al leerlos.
- El manifiesto no contiene rutas absolutas ni credenciales. Su SHA identifica el dataset
  efectivo del experimento y debe mantenerse junto con imágenes/configuración.

No cambiar `seed`, datos o perfil después de comenzar. Para otra configuración usar
otro manifiesto/directorios y repetir ambos casos. No mezclar métricas de experimentos
con distinta partición, N, arquitectura, pasos, extractor o preprocesamiento.

## Selección y reanudación

La MSE de validación se mide sobre hasta 256 imágenes fijas, con ruido y tiempos
reproducibles. Se selecciona `best.safetensors` por la menor MSE de validación, no
por FID de test. Menor MSE no es garantía de mejor percepción: se informa esa limitación.
El checkpoint elegido puede ser el inicial; el informe distingue `checkpoint_step`
de las actualizaciones totales completadas.

`latest.pt` guarda pesos crudos, EMA, AdamW, GradScaler, RNG, posición e historial.
`best.safetensors` y `last.safetensors` contienen pesos EMA y procedencia dentro del
archivo, no el estado necesario para continuar entrenando. El sampler usa posiciones
absolutas; DataLoader y ruido tienen generadores separados del RNG del forward.
Si AMP desborda, no se avanza al siguiente grupo de imágenes.

`Ctrl+C`/SIGTERM solicitan terminar el update actual y guardar. SIGKILL, OOM o pérdida
de energía pueden perder trabajo desde el último `latest.pt`. Nunca se promete guardar
un acumulado parcial de gradientes. La reanudación comprueba software, hardware,
configuración, dataset y SHA del código/assets. Cambiar README o notebook no altera
esa huella; cambiar módulos o arquitectura sí.

## Obtención y evaluación de resultados

1. Finalizar los entrenamientos o declarar expresamente que la comparación es parcial.
2. Generar los tres casos con DDIM 50, las mismas semillas y lote físico de inferencia.
3. Calcular FID/KID en test con el mismo N, hasta 5000. Si se reduce N, declararlo.
4. Ejecutar diagnósticos opcionales y el informe. Por defecto, este exige presupuestos
   completos e igual número de exposiciones de imágenes en los casos entrenados.

FID usa características Inception de 2048 dimensiones. KID usa el mismo tipo de
extractor, kernel polinómico y subconjuntos con semilla fija. Inception proviene de
imágenes naturales, no garantiza medir todas las cualidades del anime. FID depende
del tamaño muestral; la desviación KID es entre subconjuntos, no entre entrenamientos
ni un intervalo de confianza. KID puede ser ligeramente negativo por estimación.

La inferencia se mide después del warmup y con sincronización CUDA. Incluye muestreo
y transferencia final de tensores, no carga/descarga de pesos ni escritura PNG/GIF.
El GIF se obtiene en otra pasada. `seconds_per_image` es tiempo amortizado por imagen,
no latencia de una petición individual cuando el batch es mayor que uno. VRAM se
expresa en GiB de memoria asignada máxima de PyTorch, no toda la memoria reservada
ni el consumo global que muestra `nvidia-smi`.

El tiempo de entrenamiento registrado incluye validaciones, previews y guardado
durante el bucle; excluye inicialización/carga/descarga previas. Una calibración
sintética no mide lectura de datos ni todas esas tareas y no demuestra convergencia.

## Ablaciones y diagnósticos

- Estudiar 25/50/100 pasos DDIM con los mismos pesos exclusivamente en **val**, si
  se usará para decidir. Después fijar un único protocolo para la evaluación final.
- `ablation-report` exige mismos pesos, hardware y configuración salvo número de pasos;
  rechaza test. No comparar 25 pasos en un modelo con 100 en otro como si fueran iguales.
- SSIM entre generadas es un proxy de diversidad estructural. No se compara contra
  reales arbitrarias ni se interpreta como fidelidad. Ruido puede parecer diverso.
- dHash contra train y hojas de vecinos ayudan a inspeccionar similitud gruesa;
  no prueban memorización ni su ausencia. Revisar las imágenes.
- Una sola semilla permite conclusiones exploratorias; para robustez harían falta
  varias ejecuciones independientes y un diseño emparejado adicional.

## Estado verificable de esta entrega

Se valida estructura del notebook sin ejecutar celdas, imports/CLI, empaquetado y
pruebas CPU con entradas sintéticas. No se ha entrenado una U-Net real, descargado
Anime Face Dataset o pesos preentrenados, ni calculado FID/KID sobre imágenes reales.
No existe aún evidencia de que el entrenamiento completo quepa o converja en la 3060.
Esas verificaciones corresponden al compañero; las salidas no se sustituyen por ejemplos.

Ver [ADR 001](decisions/001-experimento-controlado.md) y las referencias del notebook.
