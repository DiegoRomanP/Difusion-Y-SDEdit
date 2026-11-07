# PC3 — Difusión para rostros de anime

Proyecto académico en **PyTorch**, gestionado exclusivamente con **uv**. Compara un
DDPM entrenado desde cero y el fine-tuning de la misma arquitectura sobre
[Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset).
Incluye notebook explicativo, entrenamiento reanudable, inferencia, FID/KID,
diagnósticos y gráficos comparativos.

**Estado:** código y notebook preparados, **sin ejecutar el notebook ni entrenar
modelos** durante su elaboración. No se descargaron aquí imágenes ni pesos del
modelo. No hay resultados reales ni un ganador supuesto. Las pruebas usan CPU y
fixtures sintéticos; no demuestran calidad, convergencia ni ajuste en la GPU prevista.
La [presentación HTML animada](presentacion_web/presentacion.html) y su guion están
preparados como versión previa a la ejecución; la comparación se completará con
evidencias reales. [Abrir, editar y actualizar](presentacion_web/README.md).

## 1. Entorno: trabajar siempre dentro de pc3/

Equipo previsto: **RTX 3060, 8 GB de VRAM**, Linux o Windows, driver compatible con
CUDA 12.8. Se recomienda al menos 16 GB de RAM del sistema y unos 20 GB libres para
datos, dos ejecuciones, checkpoints y temporales. El espacio exacto depende del dataset
extraído y los resultados. No hace falta instalar un toolkit CUDA para compilar código.

El proyecto ya está inicializado: **no volver a ejecutar `uv init`**.

```bash
cd pc3
uv sync --locked --extra cu128 --group notebook --group dev
uv run --no-sync python main.py doctor
```

`uv` usa Python 3.13 según `.python-version`; el lockfile fija dependencias. Los extras
`cpu` y `cu128` son excluyentes. El workspace local vacío evita heredar el entorno del
curso: **`.venv`, `pyproject.toml` y `uv.lock` pertenecen a `pc3/`**.

Para comprobaciones sin GPU:

```bash
uv sync --locked --extra cpu --group dev
uv run --no-sync pytest
```

No instalar con `pip` ni mezclar versiones en el notebook. Los comandos siguientes
usan `--no-sync` porque el entorno ya fue instalado: así no se cambia el backend ni
se retiran dependencias de Jupyter mientras está abierto. Antes de entrenar, comprobar
que `doctor` informa `cuda_available: true` y un wheel CUDA, no `+cpu`.

## 2. Qué se compara

| Caso | Inicialización | Adaptación a anime |
|---|---|---|
| Desde cero | Pesos aleatorios | Entrenamiento de todos los parámetros |
| Abierto sin adaptar | Pesos preentrenados | Ninguna: referencia inicial |
| Fine-tuning | Los mismos pesos abiertos | Entrenamiento de todos los parámetros |

Checkpoint abierto: [`anton-l/ddpm-ema-flowers-64`](https://huggingface.co/anton-l/ddpm-ema-flowers-64),
Apache-2.0, revisión **`dc3af58142513ad2e41e94b57a35ec3a04286a94`**. Es un modelo
de flores, no de anime: permite estudiar transferencia entre dominios sin el riesgo
conocido de reutilizar checkpoints anime que podrían haber visto el test de Kaggle.
El fine-tuning **no tiene una mejora garantizada**.

Ambos modelos entrenados comparten U-Net, datos, orden de ejemplos, aumentos, objetivo
MSE, AdamW, calendario de aprendizaje, EMA y presupuesto local. Se declara la ventaja
de cómputo histórico del checkpoint abierto; no se iguala el preentrenamiento externo.
La arquitectura es de Diffusers; los bucles y herramientas del proyecto están en `src/pc3/`.

Perfil común: [`configs/rtx3060_8gb.toml`](configs/rtx3060_8gb.toml).

| Parámetro | Valor inicial |
|---|---:|
| Resolución | RGB 64×64 |
| Lote físico / acumulación | 1 / 16 |
| Lote efectivo | 16 |
| Updates por modelo | 30 000 |
| Exposiciones de imágenes por modelo | 480 000 |
| Pasos de difusión de entrenamiento | 1000, beta lineal 0.0001…0.02 |
| Optimizador / LR | AdamW / 0.0001, warmup + coseno |
| EMA | 0.999, copias en CPU |
| Precisión / memoria | FP16 + GradScaler + checkpointing de gradientes |
| Validación / checkpoint | Cada 500 updates y al terminar |
| Muestreo final predefinido | DDIM, 50 pasos, eta=0 |

Modificar el perfil **antes** de iniciar ambos entrenamientos. No cambiar configuración,
datos, paquetes o código entre un entrenamiento y su reanudación/evaluación. Se guardan
SHA del manifiesto y del código/assets. Para otra configuración crear otra ejecución
emparejada. Un update no es una época: incluye 16 microbatches en el perfil inicial.

## 3. Descargar y preparar datos

```bash
uv run --no-sync python main.py download
uv run --no-sync python main.py prepare
```

Se fija `splcher/animefacedataset/versions/3` y se descarga en `data/raw/` con KaggleHub.
Si se necesita autenticación, debe existir/configurarse en **el equipo ejecutor**;
tener una clave en otra máquina no la transfiere. Seguir la
[documentación de KaggleHub](https://github.com/Kaggle/kagglehub#authenticate).
No pegar claves en notebook, código, argumentos o commits. El programa no copia ni
muestra credenciales. Un dataset público puede descargarse sin autenticación.

**Alternativa manual:** extraer el dataset dentro de `data/raw/`, conservando las
subcarpetas de imágenes, y ejecutar solamente `prepare`. Si se usa otra ubicación,
cambiar `data_dir` en el perfil antes de preparar/entrenar. Nunca sobrescribir un
directorio de datos no vacío con `download`.

La preparación valida RGB/lectura, recorta a 64×64, elimina duplicados exactos y
mantiene grupos dHash/espejo en una sola partición. Genera `data/manifest.json`
con rutas relativas, hashes, exclusiones y conteos efectivos de train/val/test
(aproximadamente 80/10/10). No garantiza detectar todos los casi duplicados.
Un manifiesto existente no se reemplaza automáticamente.

## 4. Calibrar y entrenar — lo ejecuta el compañero

```bash
uv run --no-sync python main.py calibrate
```

Es una medición sintética de VRAM y segundos por update, no entrenamiento válido ni
evidencia de calidad. La proyección temporal excluye lectura, validación y checkpoints.
**No se promete que ambos casos terminen en una noche.** Usar la medición real y
reservar tiempo adicional. Si cambia el lote físico, mantener el mismo lote efectivo
y perfil en ambos casos. La U-Net abierta es grande aunque trabaje a baja resolución.

Entrenar uno después del otro:

```bash
uv run --no-sync python main.py train --init random
uv run --no-sync python main.py train --init pretrained
```

Los resultados se guardan en `runs/from_scratch/` y `runs/finetuned/`. El segundo
comando descarga los pesos fijados, comprueba su arquitectura y rechaza cargas
parciales. No se usa `strict=False`, código remoto ni inicialización aleatoria
silenciosa para reemplazar parámetros preentrenados que falten.

Para reanudar tras una pausa:

```bash
uv run --no-sync python main.py train --init random --resume
uv run --no-sync python main.py train --init pretrained --resume
```

`Ctrl+C` solicita finalizar el update actual y guardar. No cerrar a la fuerza el
proceso si se quiere ese guardado. Un corte de energía/SIGKILL conserva solo el último
checkpoint completo. Si quedó `.training.lock`, verificar que ningún proceso usa ese
directorio antes de retirar **solo ese archivo** manualmente. No borrar pesos o datos
para intentar solucionar problemas. Un error durante la inicialización puede no haber
producido todavía un checkpoint reanudable.

Los valores del perfil se comprueban antes de descargar/cargar modelos. Para usar
otro perfil, la opción global va **antes del comando**:

```bash
uv run --no-sync python main.py --config configs/rtx3060_8gb.toml train --init random
```

## 5. Generar y evaluar los tres casos

Después de completar ambos entrenamientos:

```bash
uv run --no-sync python main.py sample --weights runs/from_scratch/best.safetensors --output reports/samples/from_scratch --label "Desde cero"
uv run --no-sync python main.py sample --pretrained --output reports/samples/pretrained --label "Abierto sin adaptar"
uv run --no-sync python main.py sample --weights runs/finetuned/best.safetensors --output reports/samples/finetuned --label "Fine-tuning"
```

Por defecto se generan 5000 imágenes por caso, mismas semillas y DDIM50. Repetir el
mismo comando reanuda una generación interrumpida verificando hashes. Si cambia
pesos, conteo, hardware o muestreo, usar **otra salida**, no mezclar imágenes.

```bash
uv run --no-sync python main.py evaluate --samples reports/samples/from_scratch
uv run --no-sync python main.py evaluate --samples reports/samples/pretrained
uv run --no-sync python main.py evaluate --samples reports/samples/finetuned
```

La primera evaluación puede descargar pesos oficiales Inception de Torch-Fidelity.
Se usa test reservado y el mismo N disponible, hasta 5000. Para N menor usar
`--num-images N` en todos los casos y declarar que la comparación es exploratoria.
`--batch-size` controla el lote de extracción de métricas, no el de entrenamiento.

Diagnósticos opcionales, sin descargar otro modelo:

```bash
uv run --no-sync python main.py diagnose --samples reports/samples/from_scratch
uv run --no-sync python main.py diagnose --samples reports/samples/pretrained
uv run --no-sync python main.py diagnose --samples reports/samples/finetuned
```

Informe y gráficos:

```bash
uv run --no-sync python main.py report --metrics reports/samples/from_scratch/metrics.json reports/samples/pretrained/metrics.json reports/samples/finetuned/metrics.json --runs runs/from_scratch runs/finetuned --diagnostics reports/samples/from_scratch/diagnostics.json reports/samples/pretrained/diagnostics.json reports/samples/finetuned/diagnostics.json
```

El informe rechaza diferencias de protocolo, dataset, N, hardware de inferencia o
contrato, y entrenamientos con presupuesto incompleto. `--allow-partial` sirve solo
para exploración y marca expresamente el informe como tal. No permite presentar
una ejecución parcial como comparación de presupuestos completos.

## 6. Ablación de pasos — solo validación

Es opcional. Si se usará para elegir pasos, realizarla **antes de la evaluación
final de test**. Repetir estas dos órdenes cambiando `25` por `50` y `100`:

```bash
uv run --no-sync python main.py sample --weights runs/finetuned/best.safetensors --output reports/ablation/samples_25 --count 1000 --steps 25 --label "Fine-tuning 25 pasos"
uv run --no-sync python main.py evaluate --samples reports/ablation/samples_25 --split val --num-images 1000
```

```bash
uv run --no-sync python main.py ablation-report --metrics reports/ablation/samples_25/metrics.json reports/ablation/samples_50/metrics.json reports/ablation/samples_100/metrics.json
```

El informe exige mismos pesos y protocolo salvo número de pasos, y rechaza test.
Si val tiene menos de 1000 imágenes, reducir el conteo común en todos los casos.
Aplicar después un único protocolo final a los tres modelos. El perfil principal
de 50 pasos ya está predefinido si se omite esta fase.

## 7. Notebook

Abrir [`anime_diffusion.ipynb`](anime_diffusion.ipynb):

```bash
uv run --no-sync jupyter lab
```

Requiere haber incluido `--group notebook` en la instalación. El notebook explica
matemáticas, modelo, adaptaciones de semanas 8/9, datos, ejecución, métricas y límites.
Usa los mismos módulos/comandos que `main.py`; cada entrenamiento corre en un proceso
independiente para liberar VRAM. Seleccionar el kernel de `pc3/.venv`.

Sus interruptores `RUN_*` son `False`: habilitar solo la fase que se quiera ejecutar.
Las figuras pendientes se indican como pendientes, no se rellenan con datos ficticios.
Para modificar su contenido editar `tools/build_notebook.py` y regenerar **sin ejecutar**:

```bash
uv run --no-sync python tools/build_notebook.py
uv run --no-sync python tools/build_notebook.py --check
```

## 8. Archivos y evidencias

```text
pc3/
├── main.py                       # CLI
├── anime_diffusion.ipynb          # Notebook entregado sin ejecutar
├── pyproject.toml / uv.lock       # Dependencias y backends cpu/cu128
├── configs/rtx3060_8gb.toml        # Perfil común
├── src/pc3/
│   ├── data.py / tensor_data.py   # Manifiesto, particiones y lectura verificada
│   ├── model.py / stream.py       # U-Net, DDPM, EMA y orden reanudable
│   ├── training.py               # Train, validación y calibración
│   ├── sampling.py               # DDPM/DDIM, PNG, cuadrículas y GIF
│   ├── metrics.py                # FID/KID y contratos
│   ├── diagnostics.py            # SSIM entre generadas, vecinos dHash
│   └── reporting.py / cli.py      # Gráficos y orquestación
├── tests/                        # Pruebas sin entrenamiento real
├── tools/build_notebook.py        # Fuente reproducible del notebook
└── docs/                         # Protocolo y decisiones
```

Salidas **solo después de ejecutar**, excluidas de Git:

| Ruta | Contenido |
|---|---|
| `runs/*/run.json` | Hardware, versiones, código, contrato, pasos y tiempos |
| `runs/*/history.csv` | Pérdida train/val, LR y evolución |
| `runs/*/latest.pt` | Estado completo para reanudar; no usar para inferencia |
| `runs/*/best.safetensors` | EMA elegida por MSE de validación |
| `runs/*/last.safetensors` | EMA al terminar o pausar ordenadamente |
| `runs/*/previews/` | Evolución con semillas fijas, incluido el paso inicial |
| `reports/samples/*/` | PNG, GIF, cuadrícula, procedencia, tiempos y métricas |
| `reports/comparison/` | CSV, informe Markdown y gráficos comparativos |
| `reports/ablation/` | Métricas y gráfico de pasos, cuando se ejecute |

Para evidencias del curso conservar JSON/CSV, previews, gráficos y muestras junto
con el código exacto. No subir pesos de cientos de MB, dataset, `.venv`, credenciales
o `.git` como si fueran código. Compartir la carpeta de código y las evidencias por
separado; el compañero debe crear su entorno con `uv sync --locked` y el extra correcto.

## 9. Cómo interpretar resultados

- **FID y KID menores** indican mayor cercanía distributiva bajo el mismo protocolo,
  no una certificación absoluta de realismo ni un resultado comparable con otro paper.
- La desviación de KID es entre subconjuntos, **no un IC entre entrenamientos**.
- Menor MSE de ruido no garantiza mejores imágenes. Se publica el paso seleccionado.
- SSIM entre generadas es solo un proxy estructural de diversidad; ruido también
  puede ser diverso. No es SSIM contra una real arbitraria.
- dHash/vecinos no prueban ni descartan memorización. Inspeccionar imágenes.
- Inception usa imágenes naturales: FID/KID tienen limitaciones de dominio en anime.
- Los tiempos de inferencia son amortizados, tras warmup, excluyendo guardar PNG/GIF;
  VRAM es máximo asignado por PyTorch en GiB. Revisar el protocolo antes de compararlos.
- Una semilla permite una conclusión exploratoria, no una afirmación de robustez.

Detalles: [protocolo experimental](docs/protocolo_experimental.md) y
[ADR 001](docs/decisions/001-experimento-controlado.md).

## 10. Verificación y solución de problemas

```bash
uv run --no-sync pytest
uv run --no-sync ruff check src tests tools main.py
uv run --no-sync python tools/build_notebook.py --check
uv lock --check
uv pip check
uv build
```

En la elaboración pasaron **41 pruebas**: configuración, particiones/duplicados,
antifuga, RNG/reanudación de orden, fórmula DDPM, EMA, muestreo con una función cero,
procedencia, contratos, notebook sin salidas e informes/diagnósticos con fixtures
sintéticos temporales. Se verificó que una mutación de la condición antifuga hace
fallar su prueba. **No son 41 pruebas de entrenamiento ni una evaluación en GPU.**
`uv pip check` comprueba coherencia de dependencias, no ausencia de vulnerabilidades.

| Problema | Acción |
|---|---|
| CUDA no disponible | Revisar `doctor`, extra `cu128` y driver; no continuar con `+cpu` por accidente |
| OOM | Calibrar; lotes físicos menores, EMA CPU y checkpointing; mismo perfil para ambos |
| Manifiesto ya existe | Conservarlo; para otra división elegir otro archivo y repetir ambos casos |
| Imagen modificada | Restaurar el dataset original; no desactivar hashes para continuar |
| Cambio de contrato/código al reanudar | Volver a la versión original o iniciar otra ejecución emparejada |
| Checkpoint abierto incompatible | Detenerse y resolver compatibilidad; no ignorar pesos faltantes |
| Generación existente diferente | Usar otra salida; repetir el mismo comando solo para reanudar |
| Informe rechaza resultados | Igualar protocolo/presupuesto, o declarar `--allow-partial` exploratorio |
| Faltan pesos Inception/red | Ejecutar la descarga autorizada con conexión; no inventar métricas |
| Falta espacio | Liberar/copiar artefactos propios con cuidado; el guardado atómico necesita espacio temporal |

## Atribución y siguientes entregas

Base docente: notebook `Diffusion Models.ipynb`, diapositivas Parte IV y guías de
`semana8/` y `semana9/`. El notebook cita DDPM, DDIM, FID/KID y documentación oficial.
Se describe explícitamente el cambio Huber→MSE y TensorFlow/CelebA→PyTorch/anime.
La licencia Apache-2.0 del checkpoint no concede derechos adicionales sobre imágenes
del dataset; revisar sus licencias/procedencia antes de redistribuir o usar comercialmente.

La entrega principal es ahora **[un HTML autocontenido](presentacion_web/presentacion.html)**,
hecho con React, Tailwind y Anime.js dentro de `presentacion_web/`. Se abre sin servidor
ni conexión e incluye 7 escenas, 2 anexos, guion, animaciones y controles de exposición.
El objetivo sigue siendo **5:30 planificados**, máximo **6 minutos**, para el viernes
**13 de noviembre de 2026**. Falta incorporar los resultados reales y ensayar la duración;
los estados pendientes no son métricas cero.

El frontend se puede editar/levantar de forma independiente. Ver
[instrucciones de apertura, desarrollo y actualización](presentacion_web/README.md).
Los archivos [PPTX/PDF y guion anteriores](presentacion/README.md) se conservan.
La suite Python mantiene 49 pruebas; el frontend tiene 7 pruebas unitarias y 17 de
navegador. Ningún generador modifica los módulos experimentales ni ejecuta el notebook.
