# PC3 — Difusión para rostros de anime

Proyecto académico en **PyTorch**, gestionado exclusivamente con **uv**. Entrena un
DDPM propio a 64×64 sobre [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
y reescala sus muestras a 256×256 con un modelo de superresolución preentrenado.
Incluye entrenamiento reanudable, inferencia, FID/KID, diagnósticos y gráficos.

**Estado (11 de noviembre de 2026).** [ADR 002](docs/decisions/002-cascada-superresolucion.md)
retiró la comparación de inicialización (aleatoria vs. fine-tuning de pesos abiertos):
aislar ese efecto obligaba a copiar la arquitectura del checkpoint ajeno, y ese candado
limitaba la calidad. El proyecto pasa a una cascada difusión + superresolución con
arquitectura libre, **sin brazo preentrenado/fine-tuning**.

El entrenamiento de la etapa 1 (`anime64`, arquitectura libre) **se ejecutó completo**
en la RTX 4070 Ti: 87 274/87 274 updates, `status: completed`, MSE de validación
0.02388. Muestreo, FID/KID y diagnósticos ya corrieron sobre esos pesos:
**FID 10.099, KID 0.003942 ± 0.000463** (test, DDIM 50, N según protocolo). El código
que sostenía el brazo preentrenado/fine-tuning (CLI, `Config`, `create_model`,
perfiles y pruebas de la RTX 3060) ya se retiró del repositorio.

**Pendiente:** la etapa 2 (superresolución A/B con Real-ESRGAN vs. bicúbica) y
reescribir el notebook y la presentación, que todavía describen la comparación
retirada. Exposición: viernes **13 de noviembre de 2026**.

## 1. Entorno: trabajar siempre dentro de pc3/

Equipo de referencia: **RTX 4070 Ti, 12 GB de VRAM**, Windows o Linux, driver compatible
con CUDA 12.8. Se recomienda al menos 16 GB de RAM y unos 20 GB libres para datos,
checkpoints y temporales. No hace falta un toolkit CUDA para compilar código. Las
evidencias históricas se produjeron en una RTX 3060 de 12 GB.

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

No instalar con `pip` ni mezclar versiones en el notebook. Los comandos usan `--no-sync`
porque el entorno ya fue instalado: así no se cambia el backend ni se retiran
dependencias de Jupyter mientras está abierto. Antes de entrenar, comprobar que
`doctor` informa `cuda_available: true` y un wheel CUDA, no `+cpu`.

## 2. Qué se produce

Dos etapas. La primera es nuestra; la segunda es un modelo ajeno que se declara como tal.

| Etapa | Qué hace | Cómputo |
|---|---|---|
| 1 — Difusión | DDPM propio a 64×64, arquitectura libre | Entrenamiento local |
| 2 — Superresolución | Real-ESRGAN anime ×4 → 256×256 | Preentrenado externo, sin entrenar |

La comparación controlada está **dentro de la etapa 2**: del mismo conjunto de muestras
de 64 px se derivan el brazo **A** (Real-ESRGAN) y el brazo **B** (bicúbica). Mismo
equipo, mismas semillas, mismo conteo, sin entrenamiento extra. La etapa 2 **no añade
información**: puede volver nítido un artefacto, así que se muestran pares con las
mismas semillas y sin escoger las caras favorables.

**FID y KID se calculan solo a 64×64.** El dataset es nativo de 64 px, así que no hay
verdad de terreno a 256; reescalar también las reales mediría el parecido con reales
reescaladas, no realismo. Detalle en el [protocolo](docs/protocolo_experimental.md).

Lo que ya **no** se hace: cargar `anton-l/ddpm-ema-flowers-64`, entrenar un segundo
modelo y enfrentar inicializaciones. Esa comparación y el código que la sostenía
(`--init pretrained`, `sample --pretrained`, los perfiles de la RTX 3060, la
arquitectura `flores64`) se retiraron del repositorio; lo que queda de esas evidencias
es el resumen numérico en [ADR 001](docs/decisions/001-experimento-controlado.md) y en
[ADR 002](docs/decisions/002-cascada-superresolucion.md), no los artefactos originales.

## 3. Descargar y preparar datos

```bash
uv run --no-sync python main.py download
uv run --no-sync python main.py prepare
```

Se fija `splcher/animefacedataset/versions/3` y se descarga en `data/raw/` con KaggleHub.
Si se necesita autenticación debe existir en **el equipo ejecutor**; tener una clave en
otra máquina no la transfiere. Seguir la
[documentación de KaggleHub](https://github.com/Kaggle/kagglehub#authenticate).
No pegar claves en notebook, código, argumentos o commits.

**Alternativa manual:** extraer el dataset dentro de `data/raw/` conservando las
subcarpetas, y ejecutar solo `prepare`. Si se usa otra ubicación, cambiar `data_dir`
en el perfil antes de preparar. Nunca sobrescribir un directorio no vacío con `download`.

La preparación valida RGB/lectura, recorta a 64×64, elimina duplicados exactos y mantiene
grupos dHash/espejo en una sola partición. De 63 565 archivos excluye 22 043 duplicados
y reparte **33 218 train / 4 152 val / 4 152 test**.

`data/manifest.json` **está versionado a propósito**, aunque `.gitignore` excluya el resto
de `data/`. Su SHA fija qué imágenes quedaron en test y por tanto contra qué se mide
FID/KID. Debe viajar con el código a cualquier equipo ejecutor y **no regenerarse allí**:
regenerarlo puede mover al train imágenes que estaban en el test de otra ejecución, y
eso es contaminación, no una diferencia de configuración.

## 4. Arquitectura y perfiles

La arquitectura pasa a formar parte del perfil, con su hash registrado en `run.json`
**(pendiente)**. Hoy sigue fijada en `src/pc3/assets/unet_config.json` y comprobada por
huella en `src/pc3/model.py`.

| Parámetro | Anterior (ADR 001, retirado) | Vigente (ADR 002) |
|---|---|---|
| Bloques | 6 | 4 |
| `block_out_channels` | `[128,128,256,256,512,512]` | `[128,256,384,512]` |
| Resoluciones de atención | 4×4 (y 2×2 intermedio) | **16×16 y 8×8** |
| `attention_head_dim` | 8 | 64 |
| Parámetros | 113.7 M | **88.2 M** (medido: `parameter_count` en `run.json`) |

Perfiles en `configs/`. El de la noche de trabajo, `configs/rtx4070ti_12gb.toml`, ya
corrió completo con lote físico 32, EMA en CUDA y fp16: la calibración de la 4070 Ti dio
`skipped_fp16_attempts: 0`, así que fp16 es estable en esa tarjeta. `--config` no tiene
valor por defecto en `src/pc3/cli.py`: hay que pasarlo siempre explícitamente. Los
perfiles de la RTX 3060 (`rtx3060_8gb.toml`, `rtx3060_12gb.toml`) y la arquitectura
`flores64` que fijaban se retiraron junto con la comparación de ADR 001.

Constantes en ambas etapas: RGB 64×64, 1000 pasos de difusión con beta lineal
0.0001…0.02, predicción de ruido, AdamW con warmup y coseno, EMA 0.999 en CPU,
validación y checkpoint cada 500 updates, muestreo final DDIM 50 con eta=0.

Un update no es una época: incluye `batch_size * accumulation_steps` imágenes.
Modificar el perfil **antes** de empezar, no entre un entrenamiento y su reanudación.

```bash
uv run --no-sync python main.py --config configs/rtx4070ti_12gb.toml config
```

La opción global `--config` va **antes** del comando.

## 5. Entrenar

```bash
uv run --no-sync python main.py calibrate
```

Es una medición sintética de VRAM y segundos por update, no entrenamiento válido ni
evidencia de calidad. La proyección excluye lectura de datos, validación y checkpoints.
Usar la medición real y reservar tiempo adicional.

Con `--updates 3` (el valor por defecto) el promedio queda dominado por el arranque:
contexto CUDA, autotune de cuDNN y crecimiento del asignador. Para un número utilizable
hay que pedir más updates y **el perfil explícito**, porque el `--config` por defecto
apunta a un perfil de lote físico 1 con acumulación 16:

```bash
uv run --no-sync python main.py --config configs/rtx4070ti_12gb.toml calibrate --updates 30
```

```bash
uv run --no-sync python main.py --config configs/rtx4070ti_12gb.toml train --run anime64_4070ti
```

Con ADR 002 hay un solo entrenamiento, siempre con inicialización aleatoria: la CLI ya
no tiene `--init` ni soporta pesos preentrenados. Para reanudar tras una pausa:

```bash
uv run --no-sync python main.py --config configs/rtx4070ti_12gb.toml train --run anime64_4070ti --resume
```

`Ctrl+C` solicita terminar el update actual y guardar; no cerrar a la fuerza si se quiere
ese guardado. Un SIGKILL o corte de energía conserva solo el último checkpoint completo.
Si quedó `.training.lock`, verificar que ningún proceso usa ese directorio antes de
retirar **solo ese archivo**. No borrar pesos o datos para intentar arreglar problemas.

Para seguir el avance con registro legible y log en archivo:

```bash
uv run --no-sync python tools/entrenar_con_registro.py --config configs/rtx4070ti_12gb.toml train --run anime64_4070ti
```

Escribe `reports/evidencia/log_<run>.log`, así que se puede consultar reconectando sin
depender del scrollback de la terminal. `tools/monitor_gpu.py` registra VRAM y utilización.

## 6. Generar, superresolución y evaluar

```bash
uv run --no-sync python main.py sample --weights runs/<run>/best.safetensors --output reports/samples/<run> --label "Cascada etapa 1"
uv run --no-sync python main.py evaluate --samples reports/samples/<run>
uv run --no-sync python main.py diagnose --samples reports/samples/<run>
```

Por defecto 5000 imágenes, mismas semillas, DDIM 50. Repetir el mismo comando reanuda una
generación interrumpida verificando hashes. Si cambian pesos, conteo, hardware o muestreo,
usar **otra salida**, no mezclar imágenes. La primera evaluación puede descargar pesos
Inception de Torch-Fidelity. `--batch-size` controla el lote de métricas, no el de
entrenamiento. Los diagnósticos se corren sobre las muestras de 64 px, no sobre las
reescaladas.

La etapa 2 y sus brazos A/B son **(pendiente)**. Derivan del mismo conjunto de 64 px sin
regenerar muestras, y su coste se informa aparte del muestreo.

```bash
uv run --no-sync python main.py report --metrics reports/samples/<run>/metrics.json --runs runs/<run> --diagnostics reports/samples/<run>/diagnostics.json
```

El informe pasa a **registrar divergencias** de contrato, hardware, presupuesto,
muestreador y N en `comparison.json` e imprimirlas, en vez de abortar; `--strict`
restaura el rechazo **(pendiente)**. Hoy todavía rechaza, en `src/pc3/metrics.py`.
Sigue siendo **error duro** que no coincidan `manifest_sha256` o `real_subset_sha256`:
eso no es comparabilidad, es contaminación entre train y test.

Ablación de pasos, opcional y solo en **val**, antes de la evaluación final de test:

```bash
uv run --no-sync python main.py sample --weights runs/<run>/best.safetensors --output reports/ablation/samples_25 --count 1000 --steps 25 --label "25 pasos"
uv run --no-sync python main.py evaluate --samples reports/ablation/samples_25 --split val --num-images 1000
uv run --no-sync python main.py ablation-report --metrics reports/ablation/samples_25/metrics.json reports/ablation/samples_50/metrics.json reports/ablation/samples_100/metrics.json
```

La ablación histórica dio FID 34.819 / 34.356 / 37.442 para 25/50/100 pasos. El perfil de
50 pasos ya está predefinido si se omite esta fase.

## 7. La noche en la RTX 4070 Ti, por AnyDesk

El equipo de entrenamiento se usa por escritorio remoto y **no tiene credenciales de
GitHub**: el código llega por clon y las evidencias vuelven por transferencia de archivos.

Antes de salir de este equipo:

1. `git add -f data/manifest.json` y confirmar. Sin el manifiesto no hay experimento válido.
2. Empujar también los commits locales pendientes, o el clon no traerá nada de esto.
3. Comprobar que el repositorio es accesible sin credenciales. El remoto configurado es
   SSH (`git@github.com:…`); un clon anónimo necesita la URL HTTPS y un repo público.
   Si es privado, llevar el repositorio en el mismo medio que los datos.

En el equipo ejecutor:

1. Clonar, `uv sync --locked --extra cu128 --group dev`, `main.py doctor`.
2. Conseguir los datos: credenciales de Kaggle allí —la versión está fijada, así que una
   descarga nueva debería coincidir con el manifiesto— o copiar `data/raw/` (546 MB,
   63 565 archivos) por la transferencia de archivos de AnyDesk.
3. **Desactivar suspensión**, o el entrenamiento muere a mitad de la noche:
   `powercfg /change standby-timeout-ac 0` y `powercfg /change hibernate-timeout-ac 0`.
   El bloqueo de pantalla es inofensivo; suspender no. Pausar los reinicios de Windows
   Update. Activar acceso desatendido en AnyDesk para poder reconectar.
4. Lanzar con `tools/entrenar_con_registro.py`. El proceso pertenece a la sesión de
   escritorio, así que **sobrevive a la desconexión** de AnyDesk; el entrenamiento es
   reanudable, de modo que incluso un reinicio no obliga a empezar de cero.
5. Muestrear y calcular FID/KID **allí**. Los artefactos grandes no deben viajar:
   `reports/samples` ocupa del orden de 179 MB y un `latest.pt`, 1.7 GB.

De vuelta solo hacen falta `run.json`, `history.csv`, los previews, las figuras y CSV/JSON
de `reports/comparison/`, y una cuadrícula de semillas fijas por brazo. El empaquetado
reproducible está **(pendiente)** en `tools/empaquetar_resultados.py`. Para el checkpoint
EMA, transferirlo por AnyDesk o guardar solo pesos EMA en fp16.

## 8. Notebook

Abrir [`anime_diffusion.ipynb`](anime_diffusion.ipynb):

```bash
uv run --no-sync jupyter lab
```

Requiere `--group notebook` en la instalación. Usa los mismos módulos y comandos que
`main.py`; seleccionar el kernel de `pc3/.venv`. Sus interruptores `RUN_*` son `False`:
habilitar solo la fase que se quiera ejecutar.

**El notebook describe todavía la comparación de tres brazos retirada por ADR 002.** Se
reescribe cuando existan los resultados de la cascada. Para modificarlo se edita
`tools/build_notebook.py` y se regenera **sin ejecutar**:

```bash
uv run --no-sync python tools/build_notebook.py
uv run --no-sync python tools/build_notebook.py --check
```

## 9. Archivos y evidencias

```text
pc3/
├── main.py                        # CLI
├── anime_diffusion.ipynb          # Notebook (framing anterior, se reescribe)
├── pyproject.toml / uv.lock       # Dependencias y backends cpu/cu128
├── configs/                       # Perfiles por equipo
├── data/manifest.json             # Versionado: identifica el experimento
├── src/pc3/
│   ├── data.py / tensor_data.py   # Manifiesto, particiones y lectura verificada
│   ├── model.py / stream.py       # U-Net, DDPM, EMA y orden reanudable
│   ├── training.py                # Train, validación y calibración
│   ├── sampling.py                # DDPM/DDIM, PNG, cuadrículas y GIF
│   ├── metrics.py                 # FID/KID, contratos y divergencias
│   ├── diagnostics.py             # SSIM entre generadas, vecinos dHash
│   └── reporting.py / cli.py      # Gráficos y orquestación
├── tests/                         # Pruebas sin entrenamiento real
├── tools/
│   ├── build_notebook.py          # Fuente reproducible del notebook
│   ├── build_presentation.py      # PPTX/PDF y guion
│   ├── entrenar_con_registro.py   # Lanzador con registro legible y log
│   └── monitor_gpu.py             # VRAM y utilización durante el entrenamiento
└── docs/                          # Protocolo y decisiones (ADR 001, ADR 002)
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
| `reports/comparison/` | CSV, informe Markdown, gráficos y divergencias |
| `reports/ablation/` | Métricas y gráfico de pasos |
| `reports/evidencia/` | Logs y JSON del lanzador con registro |
| `capturas_entrenamiento/` | Capturas de la noche de entrenamiento |

No subir pesos de cientos de MB, el dataset, `.venv`, credenciales o `.git` como si
fueran código. Compartir código y evidencias por separado; quien reciba el código crea
su entorno con `uv sync --locked` y el extra correcto.

## 10. Cómo interpretar resultados

- **FID y KID menores** indican mayor cercanía distributiva bajo el mismo protocolo, no
  una certificación de realismo ni un número comparable con otro paper.
- Se miden **a 64×64**. No hay verdad de terreno a 256 px: la etapa de superresolución no
  tiene un FID propio legítimo contra este dataset.
- La desviación de KID es entre subconjuntos, **no un IC entre entrenamientos**.
- Menor MSE de ruido no garantiza mejores imágenes. Se publica el paso seleccionado.
- La superresolución no añade información. Un brazo A más nítido que B no demuestra que
  el modelo de difusión sea mejor, solo que el upsampler es convincente.
- SSIM entre generadas es un proxy estructural de diversidad; ruido también puede ser
  diverso. dHash y vecinos no prueban ni descartan memorización: inspeccionar imágenes.
- Inception usa imágenes naturales: FID/KID tienen validez limitada en anime.
- Tiempos amortizados tras warmup, sin escritura de PNG/GIF; VRAM es el máximo asignado
  por PyTorch en GiB. **No comparar coste entre GPUs distintas.**
- Una semilla permite una conclusión exploratoria, no una afirmación de robustez.

Detalles: [protocolo experimental](docs/protocolo_experimental.md),
[ADR 002](docs/decisions/002-cascada-superresolucion.md) y
[ADR 001](docs/decisions/001-experimento-controlado.md).

## 11. Verificación y solución de problemas

```bash
uv run --no-sync pytest
uv run --no-sync ruff check src tests tools main.py
uv run --no-sync python tools/build_notebook.py --check
uv lock --check
uv pip check
uv build
```

Las pruebas cubren configuración, particiones y duplicados, antifuga, RNG y reanudación
de orden, fórmula DDPM, EMA, muestreo con una función cero, procedencia, contratos,
notebook sin salidas e informes con fixtures sintéticos temporales. Ver la salida de
`pytest` para el número vigente: **no son pruebas de entrenamiento ni evaluación en GPU**,
y las de CPU con fixtures no demuestran calidad ni convergencia. `uv pip check` comprueba
coherencia de dependencias, no ausencia de vulnerabilidades.

| Problema | Acción |
|---|---|
| CUDA no disponible | Revisar `doctor`, extra `cu128` y driver; no continuar con `+cpu` |
| OOM | Calibrar; lote físico menor, EMA en CPU y checkpointing de gradientes |
| Manifiesto ya existe | **Conservarlo.** No regenerarlo en otro equipo |
| Imagen modificada | Restaurar el dataset original; no desactivar hashes para continuar |
| Cambio de contrato o código al reanudar | Volver a la versión original o iniciar otra ejecución |
| Generación existente diferente | Usar otra salida; repetir el mismo comando solo para reanudar |
| Informe rechaza resultados | Revisar la divergencia que señala; si es el SHA del manifiesto, **detenerse** |
| Faltan pesos Inception o red | Ejecutar la descarga autorizada con conexión; no inventar métricas |
| El entrenamiento murió de noche | Reanudar con `--resume`; revisar suspensión y Windows Update |
| Falta espacio | Liberar artefactos propios con cuidado; el guardado atómico necesita temporal |

## Atribución y siguientes entregas

Base docente: notebook `Diffusion Models.ipynb`, diapositivas Parte IV y guías de
`semana8/` y `semana9/`. Se describe explícitamente el cambio Huber→MSE y
TensorFlow/CelebA→PyTorch/anime. El upsampler de superresolución es un modelo externo
preentrenado y se declara como tal. La licencia del dataset y de cualquier checkpoint de
terceros no concede derechos adicionales sobre las imágenes: revisar procedencia antes
de redistribuir o usar comercialmente.

La entrega es **[un HTML autocontenido](presentacion_web/presentacion.html)** hecho con
React, Tailwind y Anime.js en `presentacion_web/`, con los
[PPTX/PDF y guion anteriores](presentacion/README.md) conservados. Objetivo **5:30**,
máximo **6 minutos**, para el viernes **13 de noviembre de 2026**.

**La presentación describe todavía la comparación retirada** y marca los resultados como
pendientes. Se reescribe cuando la ejecución en la RTX 4070 Ti produzca evidencias; los
estados pendientes no son métricas cero.
[Abrir, editar y actualizar](presentacion_web/README.md).
