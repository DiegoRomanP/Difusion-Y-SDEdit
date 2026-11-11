"""Ejecuta `main.py train` mostrando un registro legible encima de la barra de progreso.

No modifica el entrenamiento: lanza el mismo comando de la CLI como subproceso, deja la
barra tqdm en la última línea e imprime encima una línea cada `log_every` updates. Vive
fuera de src/pc3/, así que no altera la huella del código.

Toda la información sale de la barra que el entrenamiento ya imprime: nunca se abren
history.csv, run.json ni checkpoints mientras el entrenamiento vive. En Windows, un
archivo abierto por otro proceso hace fallar el reemplazo atómico que usa el
entrenamiento (WinError 5) y lo detiene.

Uso (mismos argumentos que main.py):
    uv run --no-sync python tools/entrenar_con_registro.py \
        --config configs/rtx4070ti_12gb.toml train --run anime64_4070ti

La salida JSON final de la CLI se guarda en reports/evidencia/result_<run>.json y el
registro legible en reports/evidencia/log_<run>.log, en vez de mostrarse en pantalla.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pc3.config import load_config  # noqa: E402
from pc3.data import load_manifest  # noqa: E402
from pc3.stream import learning_rate_at  # noqa: E402

EVIDENCE = ROOT / "reports" / "evidencia"
BAR = re.compile(r"(\d+)/(\d+)\s*\[.*?([\d.]+)(it/s|s/it).*?(?:mse=([\d.]+))?\]")


def parse_arguments(arguments: list[str]):
    """Mismo directorio y perfil que elegiría la CLI para estos argumentos."""
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument("command")
    cli.add_argument("--init", choices=["random", "pretrained"], default="random")
    cli.add_argument("--run")
    cli.add_argument("--run-dir", type=Path)
    cli.add_argument("--resume", action="store_true")
    known, _ = cli.parse_known_args(arguments)
    if known.command != "train":
        raise SystemExit("This wrapper only runs the train command.")
    config = load_config(known.config)
    if known.run_dir:
        return config, known.run_dir
    if not known.run:
        raise SystemExit("Se requiere --run NOMBRE (o --run-dir RUTA).")
    return config, Path(config.runs_dir) / known.run


class Console:
    """Imprime líneas de registro sobre la barra, que siempre se redibuja al final."""

    def __init__(self, log_path: Path):
        self.lock = threading.Lock()
        self.bar = ""
        self.width = max(40, shutil.get_terminal_size((120, 20)).columns - 1)
        self.log = log_path.open("w", encoding="utf-8")

    def _draw_bar(self):
        sys.stdout.write("\r" + self.bar[: self.width].ljust(self.width))
        sys.stdout.flush()

    def update_bar(self, text: str):
        with self.lock:
            self.bar = text
            self._draw_bar()

    def line(self, text: str, *, stamp=True):
        message = f"[{datetime.now():%H:%M:%S}] {text}" if stamp else text
        with self.lock:
            sys.stdout.write("\r" + message.ljust(self.width) + "\n")
            self.log.write(message + "\n")
            self.log.flush()
            if self.bar:
                self._draw_bar()
            else:
                sys.stdout.flush()


class ProgressLog:
    """Convierte la barra tqdm en una línea legible cada `log_every` updates."""

    def __init__(self, config, train_images: int, console: Console):
        self.config, self.console = config, console
        self.images_per_update = config.batch_size * config.accumulation_steps
        self.train_images = train_images
        self.next_step = None

    def feed(self, text: str):
        match = BAR.search(text)
        if not match:
            return
        step, total, speed, unit, mse = match.groups()
        step, total, speed = int(step), int(total), float(speed)
        every = self.config.log_every
        if self.next_step is None:
            self.next_step = (step // every + 1) * every
        if step < self.next_step or mse is None:
            return
        self.next_step = (step // every + 1) * every
        seconds = 1 / speed if unit == "it/s" else speed
        # El update número `step` usó la tasa calculada con los updates previos.
        rate = learning_rate_at(
            step - 1, self.config.max_steps, self.config.warmup_steps, self.config.learning_rate
        )
        epochs = step * self.images_per_update / self.train_images
        self.console.line(
            f"update {step:>6}/{total} | train MSE {float(mse):.4f} | lr {rate:.2e} | "
            f"{seconds:.2f} s/update | {epochs:.2f} epochs"
        )


def pump_stderr(stream, console: Console, progress: ProgressLog):
    """tqdm reescribe su línea con '\\r'; cualquier otro texto se trata como registro."""
    buffer = ""
    while chunk := stream.read(1):
        if chunk in "\r\n":
            text = buffer.strip()
            buffer = ""
            if not text:
                continue
            if BAR.search(text):
                console.update_bar(text)
                progress.feed(text)
            else:
                console.line(text, stamp=False)
        else:
            buffer += chunk
    if buffer.strip():
        console.line(buffer.strip(), stamp=False)


def gpu_name() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown GPU"


def main():
    arguments = sys.argv[1:]
    config, run_dir = parse_arguments(arguments)
    # El manifiesto no se reescribe durante el entrenamiento: leerlo aquí es seguro.
    manifest = load_manifest(config.manifest)
    train_images = sum(row["split"] == "train" for row in manifest["records"])
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    console = Console(EVIDENCE / f"log_{run_dir.name}.log")
    console.line("command: python main.py " + " ".join(arguments))
    console.line(
        f"run directory: {run_dir} | GPU {gpu_name()} | {train_images} training images | "
        f"effective batch {config.batch_size * config.accumulation_steps}"
    )
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "TQDM_NCOLS": str(console.width),
    }
    # Ctrl+C llega igualmente al entrenamiento; el envoltorio sigue vivo para informar.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    child = subprocess.Popen(
        [sys.executable, "-u", "main.py", *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = []
    readers = [
        threading.Thread(
            target=pump_stderr,
            args=(child.stderr, console, ProgressLog(config, train_images, console)),
            daemon=True,
        ),
        threading.Thread(target=lambda: output.append(child.stdout.read()), daemon=True),
    ]
    for reader in readers:
        reader.start()
    code = child.wait()
    for reader in readers:
        reader.join(5)
    result_path = EVIDENCE / f"result_{run_dir.name}.json"
    result_path.write_text("".join(output), encoding="utf-8")
    # El entrenamiento ya terminó: leer run.json ya no puede interferir con él.
    try:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        hours = meta.get("wall_seconds", 0) / 3600
        console.line(
            f"finished: status {meta.get('status')} | {meta.get('completed_steps')} updates | "
            f"{hours:.2f} h | best val MSE {meta.get('best_validation_mse', float('nan')):.4f}"
        )
    except (OSError, ValueError):
        console.line("finished: could not read run.json")
    console.line(f"exit code {code} | CLI result saved to {result_path.relative_to(ROOT)}")
    console.bar = ""
    sys.exit(code)


if __name__ == "__main__":
    main()
