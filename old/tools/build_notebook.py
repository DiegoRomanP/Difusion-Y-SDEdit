"""Construir/validar el notebook sin ejecutar ninguna de sus celdas.

uv run --extra cpu --group dev python tools/build_notebook.py [--check]
Este archivo es la fuente editable del notebook; los módulos implementan la lógica.
"""

import argparse
import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip() + "\n"))

    md(r"""
    # PC3 — Difusión para generación de rostros de anime

    **Entrenamiento desde cero frente a fine-tuning de pesos abiertos**

    - Dataset: [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset),
      versión 3.
    - Entorno: `uv`, Python 3.13, PyTorch y Hugging Face Diffusers.
    - Equipo de ejecución previsto: RTX 3060 de **8 GB**; una noche o más, según calibración.
    - Base docente: notebook de difusión y Parte IV de semana 9,
      más fundamentos generativos de semana 8.
    - Entrega/exposición: viernes **13 de noviembre de 2026**.

    **Estado de entrega: notebook NO ejecutado, sin métricas ni muestras inventadas.**
    Sus celdas de descarga, entrenamiento y resultados están desactivadas por defecto.
    Las evidencias se producirán en el equipo del compañero.
    """)
    md(r"""
    ## 1. Pregunta, alcance y criterio de comparación

    ¿Con el mismo conjunto anime y presupuesto local conviene entrenar desde cero o
    adaptar un DDPM preentrenado? Comparamos tres estados:

    | Caso | Pesos iniciales | Datos anime usados para entrenar |
    |---|---|---|
    | A — Desde cero | Aleatorios | Solo train |
    | B — Referencia | Abiertos, sin adaptar | Ninguno |
    | C — Fine-tuning | Los mismos pesos de B | El mismo train que A |

    **A y C comparten arquitectura**, transformaciones, objetivo, optimizador,
    lote efectivo, orden de datos y número de actualizaciones. B contextualiza el
    punto de partida, no tiene el mismo entrenamiento local.

    El checkpoint abierto es `anton-l/ddpm-ema-flowers-64` (Apache-2.0), fijado por SHA.
    Está entrenado en flores, no anime. Medimos transferencia entre dominios sin
    recurrir a checkpoints anime potencialmente contaminados por el test de Kaggle.
    No se supone que C gane. El preentrenamiento histórico es una ventaja externa
    declarada, no cómputo total igualado. «Desde cero» se refiere a los pesos, no a
    reescribir la arquitectura de Diffusers.
    """)
    md(r"""
    ## 2. Instalación exclusivamente con uv

    Desde una terminal situada en **`pc3/`**:

    ```bash
    uv sync --locked --extra cu128 --group notebook --group dev
    uv run --extra cu128 --group notebook jupyter lab
    ```

    Elegir el kernel Python del entorno `pc3/.venv`. Se requiere un driver NVIDIA
    compatible con CUDA 12.8; el wheel incluye el runtime, no hace falta instalar
    un toolkit CUDA para compilar este proyecto. Para pruebas sin GPU existe el
    extra `cpu`, mutuamente excluyente con `cu128`.

    No usar `pip`, `%pip`, `!pip` ni instalar otras versiones dentro del notebook.
    No copiar tokens a celdas. KaggleHub gestiona credenciales existentes cuando
    son necesarias; también se pueden extraer imágenes manualmente en `data/raw/`.
    """)
    code("""
    import json
    import os
    import shlex
    import subprocess
    import sys
    from pathlib import Path

    candidates = [Path.cwd(), Path.cwd() / "pc3", Path.cwd().parent]
    ROOT = next((p.resolve() for p in candidates
                 if (p / "main.py").is_file()
                 and (p / "configs/rtx3060_8gb.toml").is_file()), None)
    if ROOT is None:
        raise RuntimeError("Abrir Jupyter desde pc3/ siguiendo el README.")
    os.chdir(ROOT)
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Seleccionar el kernel de pc3/.venv, gestionado con uv.")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from IPython.display import Image as DisplayImage, Markdown, display
    from PIL import Image

    from pc3.config import load_config
    from pc3.data import canonical_image, load_manifest, resolve_image
    from pc3.model import noise_scheduler, q_sample, unet_config

    PROFILE = ROOT / "configs/rtx3060_8gb.toml"
    cfg = load_config(PROFILE)
    print("Proyecto:", ROOT)
    print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())
    """)
    md("""
    ### Interruptores de ejecución: leer antes de habilitarlos

    Cambiar solo los interruptores de la fase que se va a ejecutar. `RUN_TRAINING`
    ejecuta A y C secuencialmente; `RUN_RESULTS` genera y evalúa los tres casos y
    produce el informe. `RESUME_IF_AVAILABLE` reanuda únicamente si ya hay checkpoint.
    La calibración usa datos sintéticos y **no es evidencia de entrenamiento**.
    """)
    code("""
    RUN_DOWNLOAD = False
    RUN_PREPARE = False
    RUN_CALIBRATION = False
    RUN_TRAINING = False
    RESUME_IF_AVAILABLE = True
    RUN_RESULTS = False
    RUN_ABLATION = False

    def run_cli(*arguments, enabled=False):
        command = [sys.executable, str(ROOT / "main.py"), "--config", str(PROFILE),
                   *map(str, arguments)]
        print(shlex.join(command))
        if enabled:
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            print("No ejecutado: interruptor de esta fase desactivado.")

    def show_image(path, caption):
        path = Path(path)
        display(Markdown(f"**{caption}**"))
        if path.exists():
            display(DisplayImage(filename=str(path)))
        else:
            print("Pendiente de ejecución:", path)
    """)
    code("""
    display(pd.Series(cfg.__dict__, name="configuración común").to_frame())
    print("Lote efectivo:", cfg.batch_size * cfg.accumulation_steps)
    print("Exposiciones de imágenes por modelo:",
          cfg.max_steps * cfg.batch_size * cfg.accumulation_steps)
    run_cli("doctor")  # Mostrar el comando; no descargar ni iniciar entrenamiento.
    """)
    md(r"""
    ## 3. Dataset, preprocesamiento y separación sin fuga conocida

    1. Fijar versión 3 del dataset.
    2. Verificar lectura, EXIF, RGB, recorte central y remuestreo Lanczos a 64×64.
    3. Excluir corrupción y duplicados exactos de píxeles ya preprocesados (SHA-256).
    4. Agrupar posibles casi duplicados con **dHash idéntico e invariante al espejo**.
    5. Asignar grupos completos a train/val/test, aproximadamente **80/10/10**.
    6. Normalizar a `[-1,1]`. Solo train recibe reflexión horizontal determinista.

    El manifiesto registra rutas relativas, hashes y particiones. No se regenera al
    reanudar; un cambio en una imagen se detecta al leerla. dHash no descubre todos
    los casi duplicados ni garantiza ausencia de similitud semántica. Los porcentajes
    efectivos pueden variar porque un grupo nunca se divide para forzar el 80/10/10.
    """)
    code("""
    run_cli("download", enabled=RUN_DOWNLOAD)
    if Path(cfg.manifest).exists():
        print("Se conserva el manifiesto existente:", cfg.manifest)
    else:
        run_cli("prepare", enabled=RUN_PREPARE)

    manifest = load_manifest(cfg.manifest) if Path(cfg.manifest).exists() else None
    if manifest:
        display(pd.Series(manifest["counts"], name="imágenes por partición").to_frame())
        display(pd.Series(manifest["excluded"], name="exclusiones").to_frame())
    else:
        print("Aún no hay datos preparados. Esto es normal en la entrega sin ejecutar.")
    """)
    code("""
    if manifest:
        examples = [row for row in manifest["records"] if row["split"] == "train"][:12]
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, examples):
            axis.imshow(canonical_image(resolve_image(Path(cfg.data_dir), row["path"])))
        fig.suptitle("Imágenes reales de train, no resultados del generador")
        plt.tight_layout()
        plt.show()
    """)
    md(r"""
    ## 4. DDPM: añadir ruido y aprender a revertirlo

    Sea $x_0$ una imagen real, $\beta_t$ la varianza de ruido y
    $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$.

    $$q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I).$$

    La forma cerrada permite obtener cualquier nivel sin recorrer toda la cadena:

    $$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
      \qquad \epsilon\sim\mathcal N(0,I).$$

    Usamos 1000 pasos y un calendario lineal de $10^{-4}$ a $0.02$. Cada imagen
    recibe un tiempo aleatorio y una realización nueva de ruido en cada exposición.
    **No se hacen 1000 forwards por imagen al entrenar**: solo el tiempo muestreado.
    El código usa índices `0..999`; el primer índice ya contiene una pequeña corrupción.
    Es una cadena de Markov, no un proceso de decisión: no hay acciones ni recompensas.
    """)
    code("""
    schedule = noise_scheduler()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(schedule.betas.numpy())
    axes[0].set(title="Calendario lineal", xlabel="t", ylabel="beta_t")
    axes[1].plot(schedule.alphas_cumprod.numpy())
    axes[1].set(title="Señal acumulada", xlabel="t", ylabel="alpha_bar_t")
    plt.tight_layout()
    plt.show()
    """)
    code("""
    if manifest:
        row = next(row for row in manifest["records"] if row["split"] == "train")
        image = canonical_image(resolve_image(Path(cfg.data_dir), row["path"]))
        x0 = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
        epsilon = torch.randn(x0.shape, generator=torch.Generator().manual_seed(cfg.seed))
        times = [0, 100, 300, 600, 999]
        fig, axes = plt.subplots(1, 6, figsize=(13, 3))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        for axis, timestep in zip(axes[1:], times):
            xt = q_sample(x0, torch.tensor([timestep]), epsilon, schedule.alphas_cumprod)
            axis.imshow(((xt[0].permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
            axis.set_title(f"t={timestep}")
        for axis in axes:
            axis.axis("off")
        plt.tight_layout()
        plt.show()
    """)
    md(r"""
    Esta visualización compara **marginales con epsilon fijo** para facilitar la
    lectura. No es una trayectoria conjunta muestreada del proceso de Markov.
    Tampoco son imágenes finales generadas: son datos reales corrompidos.

    ### ¿Qué aprende la red?

    La U-Net recibe $(x_t,t)$ y predice el ruido $\epsilon_\theta(x_t,t)$.

    $$\mathcal L_{simple}=\mathbb E_{x_0,t,\epsilon}
      [\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2].$$

    El notebook docente usa **Huber**. Aquí elegimos **MSE**, el objetivo simplificado
    estándar de DDPM, igual en A y C. Es una adaptación explícita, no una descripción
    retrospectiva incorrecta del código de clase. Reducir esta pérdida no garantiza
    una mejora equivalente en FID ni en percepción humana.
    """)
    md(r"""
    ## 5. Arquitectura compartida y adaptación respecto al curso

    La `UNet2DModel` utiliza convoluciones, bloques residuales, GroupNorm, SiLU,
    embeddings sinusoidales del tiempo y atención en baja resolución. Las conexiones
    skip permiten recuperar detalle espacial al reconstruir la resolución.

    ```text
    (x_t RGB64, t) → encoder 64→32→16→8→4→2
                         ↓ bottleneck
                    decoder 2→4→8→16→32→64 → epsilon_predicho RGB64
                              + skips del encoder
    ```

    La red de referencia es relativamente grande; **no es una U-Net compacta**.
    Se conserva su configuración en el paquete para que A se pueda inicializar sin
    descargar pesos. C comprueba que la carga del checkpoint sea estricta: si faltan
    parámetros, se detiene en vez de llamarlo falsamente un modelo preentrenado completo.

    | Material docente | Adaptación de esta entrega | Motivo |
    |---|---|---|
    | TensorFlow/Keras + addons | PyTorch + Diffusers | Entorno uv fijado y código compartido |
    | CelebA | Anime Face Dataset | Objetivo de la PC3 |
    | Rutas `/content/drive/...` | Rutas relativas y manifiesto | Portabilidad |
    | Huber | MSE de ruido | Objetivo DDPM común y explícito |
    | Solo train | Train/val/test por grupos | Evaluación independiente del ajuste local |
    | Sin comparación final verificada | Tres casos + FID/KID + coste | Evidencia cuantitativa |
    """)
    code("""
    display(pd.Series(unet_config(), name="arquitectura común").to_frame())
    # No se instancia aquí una U-Net grande ni se descargan pesos para mostrar su configuración.
    """)
    md(r"""
    ## 6. Entrenamiento en 8 GB y reanudación

    - **Microbatch 1 × acumulación 16** equivale a un lote de 16 imágenes por update.
      No equivale a 16 updates: `max_steps` cuenta actualizaciones del optimizador.
    - FP16 reduce activaciones; la MSE se calcula en FP32 y `GradScaler` protege gradientes.
    - Checkpointing de gradientes recalcula activaciones: menos VRAM, más tiempo.
    - EMA mantiene $\theta_{EMA}\leftarrow\lambda\theta_{EMA}+(1-\lambda)\theta$.
      Sus copias se guardan en CPU para ahorrar VRAM; introduce transferencias.
    - AdamW, warmup + coseno y clipping a norma 1, con el mismo perfil para A y C.
    - No se cargan ambos modelos en GPU a la vez; cada comando es un proceso separado.

    El sampler reconstruye cada época desde semilla y posición absoluta. El flip no
    depende del RNG de los workers. Los checkpoints incluyen red cruda, EMA,
    optimizador, scaler, RNG, posición e historial. Si FP16 desborda, se repiten las
    mismas imágenes y ruido antes de avanzar. `Ctrl+C` solicita guardar en el siguiente
    límite seguro de update; SIGKILL/corte de luz solo conservan el último checkpoint.

    Reanudar exige mismos datos, contrato, software y hardware. No se promete identidad
    bit a bit entre GPUs. Cambiar el presupuesto/hiperparámetros implica otra ejecución
    emparejada, no editar silenciosamente un experimento ya iniciado.
    """)
    code("""
    run_cli("calibrate", enabled=RUN_CALIBRATION)
    if Path("reports/calibration.json").exists():
        display(pd.Series(json.loads(Path("reports/calibration.json").read_text())).to_frame())
    """)
    md("""
    La estimación sintética excluye lectura de imágenes, validación y checkpoints.
    Reservar tiempo adicional y medir en el equipo real antes de lanzar ambos casos.
    Si hay OOM, reducir primero los lotes físicos de entrenamiento/inferencia y
    mantener el lote efectivo común. No reducir solo la arquitectura de A para que quepa.

    ### A — Entrenamiento desde cero
    """)
    code("""
    run_a = Path(cfg.runs_dir) / "from_scratch"
    arguments = ["train", "--init", "random", "--run-dir", run_a]
    if RESUME_IF_AVAILABLE and (run_a / "latest.pt").exists():
        arguments.append("--resume")
    run_cli(*arguments, enabled=RUN_TRAINING)
    """)
    md("""
    ### C — Fine-tuning completo con el mismo train

    Se ajustan **todos** los parámetros: no se introduce LoRA ni una capacidad diferente.
    El modelo abierto B se evaluará sin adaptación en la fase de resultados.
    """)
    code("""
    run_c = Path(cfg.runs_dir) / "finetuned"
    arguments = ["train", "--init", "pretrained", "--run-dir", run_c]
    if RESUME_IF_AVAILABLE and (run_c / "latest.pt").exists():
        arguments.append("--resume")
    run_cli(*arguments, enabled=RUN_TRAINING)
    """)
    md(r"""
    ## 7. Evidencias y selección del checkpoint

    `history.csv`, `run.json`, `latest.pt`, `best.safetensors`, `last.safetensors`
    y `previews/step_*.png` se guardan automáticamente. Los previews usan las mismas
    16 semillas y DDIM de 20 pasos: comparan evolución, no sustituyen la evaluación final.

    La selección de `best` usa MSE de validación sobre un subconjunto fijo, con
    epsilon/t fijos por imagen. Se eligen pesos **EMA**; `latest.pt` preserva pesos crudos
    y estado para continuar. Un mejor checkpoint puede ser temprano o incluso el
    inicial si el fine-tuning empeora. Registrar siempre su paso, no confundirlo con
    las 30 000 actualizaciones totales presupuestadas.
    """)
    code("""
    histories = {}
    for name, directory in [("Desde cero", run_a), ("Fine-tuning", run_c)]:
        if (directory / "run.json").exists():
            display(Markdown(f"### {name}"))
            display(pd.Series(json.loads((directory / "run.json").read_text())).to_frame())
        if (directory / "history.csv").exists():
            histories[name] = pd.read_csv(directory / "history.csv")
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, history in histories.items():
            if "train_mse" in history:
                train_rows = history.dropna(subset=["train_mse"])
                axes[0].plot(train_rows["step"], train_rows["train_mse"], label=name, alpha=0.8)
            if "val_mse" in history:
                val_rows = history.dropna(subset=["val_mse"])
                axes[1].plot(val_rows["step"], val_rows["val_mse"], marker=".", label=name)
        axes[0].set(title="Entrenamiento", xlabel="Updates", ylabel="MSE de ruido")
        axes[1].set(title="Validación fija", xlabel="Updates", ylabel="MSE de ruido")
        for axis in axes:
            axis.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Sin historiales reales todavía. No se dibujan curvas de ejemplo.")
    """)
    code("""
    for directory in [run_a, run_c]:
        previews = sorted((directory / "previews").glob("step_*.png"))
        for path in (previews[:1] + previews[-1:] if len(previews) > 1 else previews):
            show_image(path, f"{directory.name}: {path.stem}")
    """)
    md(r"""
    ## 8. Proceso inverso y obtención de resultados

    El DDPM aproxima el proceso inverso mediante una media aprendida:

    $$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
      \left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).$$

    Se añade ruido con la varianza posterior, salvo en la transición final. El
    scheduler implementa correctamente las transiciones. **DDIM con eta=0** permite
    usar menos pasos de inferencia sin reentrenar y es determinista dado el ruido
    inicial bajo el mismo entorno. Menos pasos no garantiza la misma calidad.

    La evaluación principal usa DDIM de 50 pasos para los tres casos. Cada imagen
    tiene su propia semilla; una semilla compartida controla azar, no fuerza la misma
    identidad facial entre modelos. La generación guarda hashes, configuración,
    procedencia, cuadrícula y GIF. Puede reanudarse repitiendo el mismo comando.
    """)
    code("""
    cases = [
        ("Desde cero", Path("reports/samples/from_scratch"), run_a / "best.safetensors"),
        ("Abierto sin adaptar", Path("reports/samples/pretrained"), None),
        ("Fine-tuning", Path("reports/samples/finetuned"), run_c / "best.safetensors"),
    ]
    for label, directory, weights in cases:
        source = ["--pretrained"] if weights is None else ["--weights", weights]
        run_cli("sample", *source, "--output", directory, "--label", label,
                enabled=RUN_RESULTS)
    """)
    code("""
    for label, directory, _ in cases:
        show_image(directory / "grid.png", label)
        show_image(directory / "denoising.gif", f"{label}: proceso inverso de la primera semilla")
    """)
    md(r"""
    ## 9. Evaluación: calidad, coste y límites de las métricas

    **FID ↓** compara medias y covarianzas de características Inception:

    $$FID=\|\mu_r-\mu_g\|_2^2+
      \operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}).$$

    **KID ↓** estima una MMD al cuadrado con kernel polinómico sobre características.
    Reportamos media y desviación de subconjuntos; **no es un intervalo de confianza**
    entre entrenamientos. Una estimación KID pequeña puede ser negativa por muestreo:
    no se recorta artificialmente a cero.

    Se usa el mismo test, subconjunto real, N, preprocesamiento, extractor y muestreo.
    Objetivo: hasta 5000 reales y 5000 generadas. Si no hay suficientes, se declara el
    N común disponible; con N pequeño FID es más inestable y sesgado. Inception está
    entrenada en imágenes naturales: sus métricas en anime tienen limitaciones.

    El tiempo se mide tras warmup y sincronización CUDA. Es coste amortizado por
    imagen, incluye muestreo/transferencias, pero excluye cargar pesos y guardar PNG/GIF.
    La captura del GIF se hace en una pasada separada para no contaminar los tiempos.
    No usamos accuracy, PSNR o SSIM contra reales arbitrarias: no existen pares objetivo.
    """)
    code("""
    for _, directory, _ in cases:
        run_cli("evaluate", "--samples", directory, "--split", "test", enabled=RUN_RESULTS)
    """)
    md(r"""
    ### Diversidad y posible memorización: diagnósticos auxiliares

    `diagnose` calcula SSIM **entre generadas** como proxy de similitud estructural
    (no de fidelidad a reales), e identifica vecinos de train mediante dHash. Los
    histogramas y hojas de contacto complementan FID/KID. Ruido puro también puede
    resultar diverso; un vecino cercano por dHash no demuestra copia ni su ausencia
    demuestra que no haya memorización. No usar estos proxies como ganador automático.
    """)
    code("""
    for _, directory, _ in cases:
        run_cli("diagnose", "--samples", directory, enabled=RUN_RESULTS)

    run_cli("report", "--metrics", *[directory / "metrics.json" for _, directory, _ in cases],
            "--runs", run_a, run_c,
            "--diagnostics", *[directory / "diagnostics.json" for _, directory, _ in cases],
            enabled=RUN_RESULTS)
    """)
    code("""
    comparison = Path("reports/comparison")
    if (comparison / "comparison.csv").exists():
        display(pd.read_csv(comparison / "comparison.csv"))
    else:
        print("Comparación pendiente de ejecución real; no hay un ganador supuesto.")
    for name, caption in [
        ("training_curves.png", "Aprendizaje: train y validación"),
        ("comparison_quality.png", "Calidad distributiva con protocolo común"),
        ("comparison_cost.png", "Coste local de entrenamiento e inferencia"),
        ("quality_vs_speed.png", "Compromiso calidad/velocidad"),
        ("diagnostics.png", "Diagnósticos auxiliares, no pruebas definitivas"),
    ]:
        show_image(comparison / name, caption)
    if (comparison / "results.md").exists():
        display(Markdown((comparison / "results.md").read_text()))
    """)
    md(r"""
    ## 10. Ablación opcional: pasos de muestreo

    Mantener los mismos pesos de C y comparar DDIM de **25, 50 y 100 pasos**.
    Esta decisión se estudia en **val**, no test. Usar el mismo N=1000 (o reducirlo
    explícitamente en todos los casos si val es menor). El comando `ablation-report`
    rechaza pesos distintos y resultados de test. Si se elige otro número de pasos,
    aplicar luego el mismo número a los tres modelos en una evaluación final nueva;
    no elegir A con 100 pasos y C con 25 sin declarar otra comparación.

    Con el perfil principal de 50 pasos ya predefinido, esta fase es opcional y puede
    ejecutarse antes de la evaluación final si se usará para tomar decisiones.
    """)
    code("""
    ablation_metrics = []
    for steps in [25, 50, 100]:
        directory = Path(f"reports/ablation/samples_{steps}")
        run_cli("sample", "--weights", run_c / "best.safetensors", "--output", directory,
                "--count", 1000, "--steps", steps, "--label", f"Fine-tuning {steps} pasos",
                enabled=RUN_ABLATION)
        run_cli("evaluate", "--samples", directory, "--split", "val", "--num-images", 1000,
                enabled=RUN_ABLATION)
        ablation_metrics.append(directory / "metrics.json")
    run_cli("ablation-report", "--metrics", *ablation_metrics, enabled=RUN_ABLATION)
    show_image("reports/ablation/ablation_steps.png", "Ablación de pasos: solo validación")
    """)
    md(r"""
    ## 11. Discusión que se completa tras ejecutar

    1. ¿Ambos modelos completaron las mismas actualizaciones y exposiciones de imágenes?
    2. ¿Cuál fue el paso seleccionado por validación en cada caso?
    3. ¿Cómo cambian FID/KID y las muestras de B a C? ¿Ayudó la adaptación a anime?
    4. Con el presupuesto común, ¿A o C ofrece mejor calidad y a qué coste?
    5. ¿Train mejora mientras val empeora? Eso sería una señal de sobreajuste, no un
       sinónimo de fine-tuning. No se concluye solo con un gráfico ruidoso.
    6. ¿Hay artefactos, poca variedad o vecinos demasiado similares? Mostrar ejemplos
       con semillas predefinidas, no solo las mejores imágenes elegidas a mano.
    7. ¿Los cambios de pasos mejoran calidad lo suficiente para justificar más tiempo?

    **Conclusión pendiente de resultados reales.** Una semilla no permite afirmar
    robustez estadística. No comparar nuestros FID con cifras del paper o diapositivas
    obtenidas en otros datasets/resoluciones/protocolos.

    ## 12. Evidencias para la entrega

    - Código: `main.py`, módulos, perfil, `uv.lock` y este notebook.
    - Entrenamiento: `run.json`, `history.csv`, previews y checkpoints reales.
    - Resultados: `generation.json`, `metrics.json`, CSV, gráficos y GIF.
    - Documentación: README, protocolo y registro de decisiones.
    - Presentación posterior: seleccionar pocas evidencias y preparar guion de
      **5:30, máximo 6 minutos**. La rúbrica original es código/evidencias 7p,
      presentación 5p y exposición 8p; la comparación es un contenido adicional,
      no un cambio inventado en la puntuación.
    """)
    md(r"""
    ## Referencias y atribución

    - Curso: [`Diffusion Models.ipynb`](../semana9/Diffusion%20Models.ipynb),
      guías de [semana 8](../semana8/Guia_de_estudio.md) y
      [semana 9](../semana9/Guia_de_estudio.md), diapositivas Parte IV.
    - Ho, Jain y Abbeel, [DDPM](https://arxiv.org/abs/2006.11239).
    - Song, Meng y Ermon, [DDIM](https://arxiv.org/abs/2010.02502).
    - [FID / TTUR](https://arxiv.org/abs/1706.08500) y
      [KID](https://arxiv.org/abs/1801.01401).
    - [Dataset Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset):
      citar al publicador y la procedencia original; revisar licencias de contenidos
      antes de redistribuir imágenes o emplearlas comercialmente.
    - [Pesos abiertos de flores](https://huggingface.co/anton-l/ddpm-ema-flowers-64), Apache-2.0.
    - [Diffusers: entrenamiento incondicional](https://huggingface.co/docs/diffusers/training/unconditional_training).
    - [uv y PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
    - [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
    - [TorchMetrics FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)
      y [KID](https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html).

    La arquitectura procede de Diffusers. Los bucles de entrenamiento y herramientas
    de preparación, reanudación, generación y comparación están implementados en
    `src/pc3/` y son compartidos por CLI y notebook. No se atribuyen a este proyecto
    los resultados publicados por terceros.
    """)
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "PC3 — Python (uv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        digest = hashlib.sha256(cell.source.encode()).hexdigest()[:12]
        cell.id = f"pc3-{index:02}-{digest}"
    nbformat.validate(notebook)
    return nbformat.writes(notebook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "anime_diffusion.ipynb"
    expected = build()
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit("Notebook desactualizado: regenerar con tools/build_notebook.py.")
        print("Notebook reproducible y estructuralmente válido; no se ejecutó ninguna celda.")
    else:
        path.write_text(expected, encoding="utf-8")
        print(f"Escrito {path.name} sin ejecutar sus celdas.")
