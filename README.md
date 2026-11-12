# PC3: difusión para rostros de anime

DDPM no condicionado en **PyTorch** que genera rostros anime de 64×64 desde ruido,
entrenado con
[Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset).
Sobre esos pesos hay además una exploración **SDEdit**: partir de una foto real en vez
de ruido.

**Resultado:** 87 274 updates en una RTX 4070 Ti, con **FID 10.099** y
**KID 0.003942 ± 0.000463** sobre test (N = 4 152, DDIM 50). El detalle está en
[docs/decisiones.md](docs/decisiones.md). Exposición: viernes 13 de noviembre de 2026.

## Entorno

Se usa **uv** con Python 3.13. El equipo de referencia es una GPU NVIDIA de 12 GB con
un driver compatible con CUDA 12.8.

```bash
uv sync --locked --extra cu128 --group dev     # GPU
uv sync --locked --extra cpu --group dev       # sin GPU (solo pruebas)
```

Los extras `cpu` y `cu128` son excluyentes. No instalar con `pip`.

## Un script por tarea

Todos se ejecutan con `uv run python scripts/<script>.py --help`, que muestra el uso
completo. La configuración tiene dos niveles:

- **`configs/proyecto.toml`**: datos, semilla, generación y evaluación. Es una sola y
  queda fija; todos los scripts la leen solos.
- **`configs/entrenamiento/<gpu>.toml`**: lo que depende de la GPU (lote, updates,
  memoria, frecuencia de validación). Lo pide `train.py` con `--profile`. Para una GPU
  nueva, `calibrate.py --hours N` lo genera solo.

| Script | Qué hace | Salida |
|---|---|---|
| `prepare_dataset.py` | Descarga el dataset (`--download`) y lo reparte en train/val/test sin casi duplicados entre particiones | `data/dataset_splits.json` |
| `calibrate.py` | Prueba lote físico y ubicación de la EMA con datos sintéticos, elige lo más rápido que cabe en la VRAM y calcula los updates para `--hours` | `configs/entrenamiento/<gpu>.toml`, `reports/calibration/<gpu>.json` |
| `train.py` | Entrena; Ctrl+C guarda y `--resume` continúa | `runs/<run>/` |
| `generate_images.py` | Genera N caras con unos pesos y mide la **inferencia** | `images/`, `grid.png`, `denoising.gif`, `generation.json` |
| `evaluate_quality.py` | FID/KID contra test, diversidad entre generadas y vecino más cercano en train | `quality.json`, `nearest_training.png` |
| `sdedit.py` | Foto real → rostro anime a varias intensidades | `reports/sdedit/` |

Flujo completo:

```bash
P=configs/entrenamiento/rtx4070ti_12gb.toml
uv run python scripts/prepare_dataset.py --download
uv run python scripts/calibrate.py --hours 6        # opcional: perfil para una GPU nueva
uv run python scripts/train.py --profile $P --run anime64_4070ti
uv run python scripts/generate_images.py --weights runs/anime64_4070ti/best.safetensors \
    --output reports/samples/anime64_4070ti
uv run python scripts/evaluate_quality.py --samples reports/samples/anime64_4070ti
uv run python scripts/sdedit.py --weights runs/anime64_4070ti/best.safetensors \
    --source data/real_faces_demo
```

Para comprobar en minutos que todo corre, sin obtener resultados válidos: entrenar con
`--profile configs/entrenamiento/ensayo_rapido.toml` (60 updates), generar con
`--count 64` y evaluar con `--num-images 64`.

## Código

```text
src/pc3/            lo que comparten los scripts
  config.py         lee configs/proyecto.toml y los perfiles de entrenamiento
  data.py           recorte 64×64, dHash, lectura de particiones
  model.py          U-Net, programa de ruido, EMA, guardar/cargar pesos
  inference.py      muestreo DDIM/DDPM y SDEdit
scripts/            un script por tarea (tabla de arriba)
tools/monitor_gpu.py  uso de GPU durante el entrenamiento
tests/test_core.py  piezas que, si fallan, invalidan el experimento sin dar error
docs/decisiones.md  qué se entrenó, cómo se mide, resultados y SDEdit
old/                presentación, notebook y respaldos del planteamiento anterior
```

## Qué produce un entrenamiento (`runs/<run>/`)

| Archivo | Contenido |
|---|---|
| `best.safetensors` | Pesos EMA con la menor MSE de validación: son los que se usan |
| `last.safetensors` | Pesos EMA al terminar o pausar |
| `latest.pt` | Estado completo para `--resume`; no sirve para inferencia |
| `history.csv` | MSE de train y val, LR y tiempos |
| `previews/` | 16 caras con semillas fijas en cada validación |
| `run.json` | Perfil, estado, updates, mejor MSE y horas |

Cada `.safetensors` guarda en sus metadatos la configuración de su U-Net, así que se
carga sin depender del código que lo entrenó.

## Datos

`data/dataset_splits.json` **está versionado**: define qué imágenes son de test. No
regenerarlo en otra máquina, porque eso podría pasar a train imágenes de test e
invalidaría el FID. `prepare_dataset.py` se niega a sobrescribirlo.

Las credenciales de Kaggle viven en el equipo que descarga
([KaggleHub](https://github.com/Kaggle/kagglehub#authenticate)). Nunca van en el código,
en argumentos ni en commits. Otra opción es extraer el dataset a mano en `data/raw/`.

## Pruebas

```bash
uv run pytest
uv run ruff check src scripts tests tools
```

Las pruebas comprueban:

- que el proceso directo coincide con Diffusers;
- que la arquitectura tiene los parámetros entrenados;
- que reanudar mantiene el orden de los datos;
- que los casi duplicados quedan en una sola partición;
- que los pesos se guardan y cargan sin cambios;
- que una semilla da la misma cara aunque cambie el lote.

No entrenan nada en GPU.

## Atribución

Base docente: notebook `Diffusion_Models.ipynb` del curso (TensorFlow/CelebA, pérdida
Huber). Este proyecto pasa a PyTorch, Anime Face Dataset y pérdida MSE, y añade la
evaluación cuantitativa (FID/KID, diversidad, vecinos). La licencia del dataset no
concede derechos sobre las imágenes generadas ni sobre las fotos usadas en SDEdit.
