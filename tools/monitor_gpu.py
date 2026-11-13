"""Panel compacto de la GPU: uso, memoria, temperatura y procesos Python que la usan.

En Windows (WDDM), nvidia-smi lista también todas las apps que dibujan en pantalla;
aquí solo se muestran los procesos Python y el comando que ejecutan.

Uso: uv run --no-sync python tools/monitor_gpu.py [--interval 2] [--once]
"""

import argparse
import os
import subprocess
import time
from datetime import datetime

import psutil

GPU_FIELDS = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def query(*arguments: str) -> list[list[str]]:
    output = subprocess.run(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [[field.strip() for field in line.split(",")] for line in output.splitlines() if line]


def command_of(pid: int) -> str:
    try:
        parts = psutil.Process(pid).cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "?"
    # Mostrar desde el script en adelante, sin la ruta larga del intérprete.
    for index, part in enumerate(parts):
        if part.endswith(".py"):
            return "python " + " ".join([os.path.basename(part), *parts[index + 1 :]])
    return " ".join(parts)


def render() -> str:
    lines = [f"GPU status  {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    for name, usage, used, total, temperature, power in query(f"--query-gpu={GPU_FIELDS}"):
        lines += [
            f"  GPU          {name}",
            f"  Utilization  {usage} %",
            f"  Memory       {float(used) / 1024:.2f} / {float(total) / 1024:.2f} GB",
            f"  Temperature  {temperature} °C",
            f"  Power        {power} W",
        ]
    pythons = [
        (pid, path)
        for pid, path, *_ in query("--query-compute-apps=pid,process_name")
        if "python" in os.path.basename(path).lower()
    ]
    lines += ["", "Python processes on the GPU:"]
    if not pythons:
        lines.append("  (none)")
    for pid, _ in pythons:
        lines.append(f"  PID {pid:>6}  {command_of(int(pid))}")
    return "\n".join(lines)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--interval", type=float, default=2.0)
    cli.add_argument("--once", action="store_true")
    args = cli.parse_args()
    while True:
        panel = render()
        if args.once:
            print(panel)
            return
        os.system("cls" if os.name == "nt" else "clear")
        print(panel, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
