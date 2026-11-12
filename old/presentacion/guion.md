# Guion de exposición — PC3

**Duración planificada: 5:30. Tope de ensayo: 6:00.** Los tiempos son un presupuesto, no una duración medida; ensayar en voz alta con cronómetro.

**Estado: sin resultados reales; no defender como comparación ejecutada.**

| Diapositiva | Tramo | Duración |
|---|---|---|
| 1. Pregunta | 0:00–0:25 | 25 s |
| 2. Fundamentos | 0:25–1:20 | 55 s |
| 3. Experimento | 1:20–2:10 | 50 s |
| 4. Implementación | 2:10–3:00 | 50 s |
| 5. Evaluación | 3:00–3:45 | 45 s |
| 6. Resultados | 3:45–4:50 | 65 s |
| 7. Cierre | 4:50–5:30 | 40 s |

## 1. ¿El preentrenamiento ayuda a generar anime? (0:00–0:25)

Nuestro proyecto estudia la generación de rostros anime mediante difusión. La pregunta es si, con el mismo presupuesto local, conviene aprender desde cero o adaptar pesos abiertos. Vamos a explicar el modelo, los controles de la comparación y cómo se evaluará sin asumir de antemano un ganador.

**Apoyo visual:** Abrir con la pregunta; no presentar el diagrama como una imagen del modelo.

## 2. Un DDPM aprende a quitar ruido (0:25–1:20)

La difusión tiene dos procesos. En el directo, añadimos ruido gaussiano a una imagen hasta destruir su estructura. La ecuación combina señal y ruido según el tiempo. Durante el entrenamiento escogemos un tiempo aleatorio y la U-Net recibe la imagen corrompida y ese tiempo para predecir el ruido añadido. Minimizamos el error cuadrático entre el ruido real y el estimado. Para generar, partimos de ruido y aplicamos el proceso inverso de forma iterativa. Usamos DDIM con 50 pasos de inferencia. Los mil pasos del calendario no significan mil pasadas de la red por imagen al entrenar: solo se muestrea un tiempo.

**Apoyo visual:** Señalar señal → ruido y luego U-Net. Explicar la ecuación; no leer símbolos uno por uno.

## 3. Mismos datos. Tres puntos de partida (1:20–2:10)

Usamos Anime Face Dataset de Kaggle, versión tres, con recorte central a sesenta y cuatro píxeles RGB. Eliminamos duplicados exactos y agrupamos por un hash perceptual que también contempla espejos. Esos grupos no cruzan entrenamiento, validación y test; el reparto previsto es ochenta, diez y diez. Comparamos la misma U-Net aleatoria, un checkpoint abierto sin adaptar y ese mismo checkpoint ajustado con todos sus parámetros. Los pesos abiertos provienen de flores, no anime, para reducir el riesgo conocido de solapamiento con test. No podemos garantizar ausencia total de similitud entre imágenes.

**Apoyo visual:** Distinguir B, referencia sin entrenamiento local, de C, fine-tuning completo.

## 4. 8 GB: reducir memoria, no la comparación (2:10–3:00)

El perfil prevé 30 000 actualizaciones por cada modelo entrenado, usando un microbatch de 1 y acumulación de 16. El lote efectivo es 16. Para reducir memoria usamos precisión mixta, checkpointing de gradientes y la copia EMA en CPU. Ambos casos corren secuencialmente en la misma GPU y conservan datos, orden de ejemplos, objetivo y optimizador. Seleccionamos los pesos EMA con la menor MSE de validación sobre ruido fijo, nunca mirando test. Los checkpoints guardan optimizador, posición y azar para reanudar. Es un presupuesto previsto: hay que calibrar memoria y tiempo en la GPU real antes de prometer una noche.

**Apoyo visual:** Aclarar que un update no es una época; el preentrenamiento externo no se iguala.

## 5. Medir calidad también exige controles (3:00–3:45)

La calidad se evalúa con FID y KID; valores menores indican mayor cercanía distributiva bajo el mismo protocolo. Se usará el mismo test y el mismo número de imágenes, hasta cinco mil, con iguales semillas y pasos de muestreo. Medimos además segundos por imagen y memoria. La MSE selecciona pesos, pero no garantiza buenas caras. Inception fue entrenada en imágenes naturales y tiene límites en anime. La desviación de KID entre subconjuntos no es un intervalo de confianza entre entrenamientos. Una sola semilla de entrenamiento solo permite conclusiones exploratorias.

**Apoyo visual:** No leer FID/KID como accuracy. Si preguntan por SSIM/dHash, usar el anexo.

## 6. Resultados: medir antes de concluir (3:45–4:50)

Esta es la comparación preparada para los tres casos. Todavía no hay muestras, métricas ni historiales de entrenamiento reales, por eso aparecen como pendientes y no como ceros. No podemos afirmar que el fine-tuning sea mejor ni que el perfil quepa en ocho gigabytes. Al ejecutar, aquí se mostrarán las cuatro primeras semillas de cada caso, sin seleccionar solo imágenes favorables, junto con FID, KID y segundos por imagen. También revisaremos las curvas para detectar sobreajuste. El código ya produce esas evidencias; las pruebas auxiliares en CPU verifican herramientas, pero no reemplazan un entrenamiento ni demuestran calidad.

**Apoyo visual:** Con resultados, señalar A → B → C y dar una interpretación visual, no solo leer cifras. Sin resultados, decirlo explícitamente. No ejecutar una demo en vivo.

## 7. La conclusión debe seguir a las evidencias (4:50–5:30)

La conclusión actual es metodológica: dejamos una comparación reproducible que separa el efecto de la inicialización del entrenamiento local. El software y el protocolo están preparados; falta ejecutar ambos modelos, comprobar memoria y tiempos, e incorporar muestras y métricas reales. El fine-tuning es una hipótesis, no un resultado garantizado. Solo después de esas mediciones podremos responder cuál conviene con este presupuesto. Gracias.

**Apoyo visual:** Cerrar aquí. Los anexos no forman parte del recorrido de 5:30.

## Ensayo y relevo entre expositores

- Recorrido principal: solo diapositivas 1–7; las 8–9 están ocultas para preguntas.
- Si exponen dos personas: una presenta 1–3 (2:10); la otra 4–7 (3:20). El relevo está incluido en el presupuesto, no añade una introducción nueva.
- A 4:50, pasar al cierre. Si hay retraso, omitir detalles de memoria o límites; no omitir el estado de ejecución ni afirmar un ganador sin resultados.
- No hacer descarga, entrenamiento ni demo en vivo. Abrir PPTX y PDF antes de empezar.
- Ensayar con la terminología: ruido, U-Net, MSE, EMA, DDIM, FID y KID.

## Antes de la entrega final

- Añadir los nombres reales del equipo en PowerPoint; no se inventaron integrantes.
- Completar ambos entrenamientos y el flujo sample/evaluate/report del README.
- Reconstruir con `--require-results`; este modo rechaza ausencias, mezcla de protocolos, presupuestos parciales y muestras alteradas.
- Revisar memoria y tiempos reales, curvas, N de test, pasos EMA elegidos y limitaciones. El anexo aclara qué sí demuestran las pruebas CPU.
- No confundir fine-tuning con sobreajuste ni std de KID con un intervalo de confianza.
- Los gráficos ilustrativos de difusión no son imágenes generadas por el modelo.
