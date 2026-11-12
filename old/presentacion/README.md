# Presentación — PC3

## Abrir y ensayar

- **[PowerPoint editable](pc3_difusion_anime.pptx):** 7 diapositivas principales y
  2 anexos ocultos para preguntas/referencias. Incluye notas del expositor.
- **[PDF](pc3_difusion_anime.pdf):** las 7 diapositivas del recorrido principal.
- **[Guion](guion.md):** narración completa, tramos por diapositiva y consejos de ensayo.
- **[Vista general](vista_general.png):** resumen visual de las 9 diapositivas.
- `estado.json`: estado de evidencias, duración planificada y huella del PPTX.

**Estado actual: versión previa a la ejecución.** No existen resultados reales en el
proyecto. La comparación muestra «Pendiente» y el cierre no afirma un ganador.
Los mosaicos de difusión son diagramas vectoriales didácticos, no caras generadas.
La presentación no debe defenderse como un experimento ya ejecutado.

La exposición está presupuestada para **5:30**, con 30 segundos de margen hasta el
máximo propio de **6:00** (el límite oficial es 7 minutos). El guion tiene unas
590 palabras; los tiempos **no han sido medidos con una exposición humana**.
Ensayar en voz alta, incluyendo pausas, señalar diagramas y cambiar diapositivas.
No incluir los anexos ni una demo de entrenamiento en ese recorrido.

| Diapositiva | Tema | Tramo |
|---|---|---|
| 1 | Pregunta y objetivo | 0:00–0:25 |
| 2 | Proceso de difusión y U-Net | 0:25–1:20 |
| 3 | Datos y tres casos | 1:20–2:10 |
| 4 | Presupuesto, memoria y validación | 2:10–3:00 |
| 5 | Métricas y controles | 3:00–3:45 |
| 6 | Comparación visual/numérica | 3:45–4:50 |
| 7 | Conclusión y límites | 4:50–5:30 |

Añadir los nombres reales del equipo antes de entregar; no se inventaron integrantes.
Si exponen dos personas, el guion propone un relevo tras la diapositiva 3.
La distribución de puntos sigue siendo código/evidencias 7, presentación 5, exposición 8.

## Regenerar sin ejecutar modelos

La fuente editable reproducible es `../tools/build_presentation.py`. El grupo `dev`
incluye las herramientas de `presentation`, gestionadas con uv. Se mantiene el extra
de PyTorch que ya use el equipo (`cpu` o `cu128`); no mezclar los dos.

Desde **`pc3/`**, con el entorno instalado:

```bash
uv run --no-sync python tools/build_presentation.py
```

Esto escribe el PPTX, el guion y `estado.json`. **No actualiza el PDF** sin `--pdf`.
La generación sobrescribe esos archivos; guardar en otra carpeta cualquier edición
manual del PowerPoint o del guion que se quiera conservar. Se puede usar
`--output presentacion/otra_version` para obtener una copia independiente.

Para actualizar también el PDF y las vistas se necesita LibreOffice:

```bash
uv run --no-sync python tools/build_presentation.py --pdf
```

Si no está en PATH, añadir `--soffice /ruta/a/soffice`. LibreOffice es solo una
herramienta de exportación: no se necesita para abrir los entregables. Las vistas
individuales de QA se guardan en `vistas/`, excluidas de Git. El PDF excluye los anexos;
la vista general los incluye. No se convierte la presentación en capturas: el PPTX
conserva textos, formas y diagramas editables con tema y diseños reutilizables.

## Incorporar las evidencias del compañero

1. Completar ambos entrenamientos y el flujo `sample`, `evaluate`, `report` del
   [README del proyecto](../README.md). No usar test para elegir parámetros.
2. Conservar las salidas canónicas `reports/samples/{from_scratch,pretrained,finetuned}/`
   y `reports/comparison/comparison.json`, junto con el mismo perfil y código.
3. Reconstruir la versión final:

```bash
uv run --no-sync python tools/build_presentation.py --require-results --pdf
```

El modo final exige los tres casos comparables en **test**, presupuestos completos
y el perfil común. Comprueba procedencia y hashes de los primeros cuatro PNG de cada
caso. No escoge las mejores caras a mano. Con resultados válidos sustituye los estados
pendientes por muestras, FID/KID y s/imagen; actualiza también las notas y el guion.
Si hay métricas inválidas o mezcladas, se detiene, no las oculta como si no existieran.

Revisar e interpretar esas evidencias antes de exponer: N efectivo, paso EMA elegido,
curvas, tiempos y memoria reales, calidad visual, diversidad y límites de una semilla.
No comparar solo FID ni afirmar significancia estadística por la desviación de KID.
Si se cambia DDIM50, cambiar el perfil común antes de la evaluación de los tres casos.

## Verificaciones realizadas

- Estructura/relaciones XML del PPTX validadas y contenido/notas extraídos para revisión.
- Renderizado del PPTX real con LibreOffice 26.8.2; revisión visual de las 9 diapositivas.
- Anexos ocultos, márgenes y contenido pendiente comprobados por pruebas automatizadas.
- La incorporación de resultados se probó **solo con fixtures sintéticos temporales**;
  no se incluyen en esta carpeta ni son evidencias del modelo.
- No se ejecutó el notebook, una calibración de GPU ni un entrenamiento.

La revisión con LibreOffice no sustituye probar el PPTX en el PowerPoint y proyector
del aula. Llevar el PDF como alternativa y ensayar con el cronómetro visible.
