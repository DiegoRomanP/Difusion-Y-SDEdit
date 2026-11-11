"""CLI con comandos explícitos y cargas diferidas; --help no carga modelos."""

import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from pc3.config import load_config
from pc3.io import write_json


def machine_slug(calibration: dict) -> str:
    """Nombre de archivo estable por equipo: tarjeta y VRAM redondeada."""
    hardware = calibration["hardware"]
    name = re.sub(r"[^a-z0-9]+", "-", hardware["name"].lower())
    name = re.sub(r"^(nvidia-|geforce-)+", "", name).strip("-")
    return f"{name}-{round(hardware['total_vram_gib'])}gb"


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="PC3 — DDPM anime a 64 px y superresolución en cascada"
    )
    # Sin valor por defecto: elegir el perfil a ciegas fue la causa de una
    # calibración engañosa; el perfil es parte del experimento, no una comodidad.
    cli.add_argument("--config", required=True, type=Path)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("config", help="Validar y mostrar el perfil sin descargas")
    commands.add_parser("doctor", help="Comprobar entorno y CUDA sin descargar modelos")
    commands.add_parser("download", help="Descargar explícitamente Anime Face Dataset versión 3")
    commands.add_parser("prepare", help="Validar imágenes y crear particiones sin fuga por grupos")
    calibration = commands.add_parser("calibrate", help="Medir VRAM/tiempo con datos sintéticos")
    calibration.add_argument("--device", default="cuda")
    calibration.add_argument("--updates", type=int, default=20)
    calibration.add_argument("--output", type=Path)
    training = commands.add_parser(
        "train", help="Entrenar una inicialización con el contrato común"
    )
    training.add_argument("--init", choices=["random", "pretrained"], default="random")
    destination = training.add_mutually_exclusive_group(required=True)
    destination.add_argument("--run", help="Nombre de la ejecución bajo runs_dir")
    destination.add_argument("--run-dir", type=Path, help="Ruta explícita, para casos raros")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--device", default="cuda")
    sampling = commands.add_parser("sample", help="Obtener imágenes/GIF y medir inferencia")
    source = sampling.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path, help="best.safetensors o last.safetensors local")
    source.add_argument("--pretrained", action="store_true", help="Referencia abierta sin adaptar")
    sampling.add_argument("--output", type=Path, required=True)
    sampling.add_argument("--label")
    sampling.add_argument("--count", type=int)
    sampling.add_argument("--steps", type=int)
    sampling.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    sampling.add_argument("--device", default="cuda")
    evaluation = commands.add_parser("evaluate", help="Calcular FID/KID sobre val o test reservado")
    evaluation.add_argument("--samples", type=Path, required=True)
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--num-images", type=int)
    evaluation.add_argument("--batch-size", type=int, default=4)
    evaluation.add_argument("--device", default="cuda")
    diagnostic = commands.add_parser("diagnose", help="Diversidad estructural y vecinos de train")
    diagnostic.add_argument("--samples", type=Path, required=True)
    diagnostic.add_argument("--limit", type=int, default=64)
    reporting = commands.add_parser("report", help="Comparar métricas reales y producir gráficos")
    reporting.add_argument("--metrics", nargs="+", type=Path, required=True)
    reporting.add_argument("--runs", nargs="*", type=Path, default=[])
    reporting.add_argument("--diagnostics", nargs="*", type=Path, default=[])
    reporting.add_argument("--output", type=Path, default=Path("reports/comparison"))
    reporting.add_argument(
        "--strict", action="store_true", help="Rechazar en vez de declarar divergencias"
    )
    ablation = commands.add_parser(
        "ablation-report", help="Graficar calidad/tiempo frente a pasos en val"
    )
    ablation.add_argument("--metrics", nargs="+", type=Path, required=True)
    ablation.add_argument("--output", type=Path, default=Path("reports/ablation"))
    return cli


def main(argv=None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "config":
            result = asdict(config)
        elif args.command == "doctor":
            import torch

            from pc3.sampling import device_info
            from pc3.training import runtime_versions

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            result = {
                "versions": runtime_versions(),
                "hardware": device_info(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "manifest_exists": Path(config.manifest).is_file(),
                "note": "No descarga datos/pesos ni demuestra que un entrenamiento quepa en VRAM.",
            }
        elif args.command == "download":
            from pc3.data import download_dataset

            try:
                result = {
                    "downloaded_to": download_dataset(config.dataset_handle, Path(config.data_dir))
                }
            except Exception as error:
                # Los SDK de autenticación pueden incluir datos sensibles en su
                # excepción; ofrecer instrucciones, no volcar la respuesta remota.
                raise RuntimeError(
                    "Kaggle no pudo descargar. Revisar credenciales/permisos/conexión y destino; "
                    "también se puede extraer manualmente el dataset en data_dir."
                ) from error
        elif args.command == "prepare":
            from pc3.data import create_manifest

            manifest = create_manifest(
                Path(config.data_dir),
                Path(config.manifest),
                image_size=config.image_size,
                seed=config.seed,
            )
            result = {key: value for key, value in manifest.items() if key != "records"}
        elif args.command == "calibrate":
            from pc3.training import calibrate

            if args.updates < 1:
                raise ValueError("--updates debe ser >=1.")
            result = calibrate(config, device_name=args.device, updates=args.updates)
            output = args.output or Path("reports/calibration") / f"{machine_slug(result)}.json"
            write_json(output, result)
            result = {**result, "written_to": output.as_posix()}
        elif args.command == "train":
            from pc3.training import train

            run = args.run_dir or Path(config.runs_dir) / args.run
            result = train(
                config,
                initialization=args.init,
                run_dir=run,
                resume=args.resume,
                device_name=args.device,
            )
        elif args.command == "sample":
            from pc3.sampling import generate

            updates = {}
            if args.count is not None:
                updates["sample_count"] = args.count
            if args.steps is not None:
                updates["sampling_steps"] = args.steps
            config = replace(config, **updates)
            config.validate()
            label = args.label or (
                args.weights.parent.name if args.weights else "pretrained_reference"
            )
            result = generate(
                config,
                output=args.output,
                weights=args.weights,
                device_name=args.device,
                sampler=args.sampler,
                label=label,
            )
            result = {key: value for key, value in result.items() if key != "images"}
        elif args.command == "evaluate":
            from pc3.metrics import evaluate

            result = evaluate(
                config,
                samples=args.samples,
                output=args.output,
                split=args.split,
                num_images=args.num_images,
                device_name=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "diagnose":
            from pc3.diagnostics import diagnose

            result = diagnose(config, samples=args.samples, limit=args.limit)
        elif args.command == "report":
            from pc3.reporting import report

            result = report(
                args.metrics, output=args.output, runs=args.runs, strict=args.strict
            )
            if args.diagnostics:
                from pc3.diagnostics import plot_diagnostics

                plot_diagnostics(args.diagnostics, output=args.output / "diagnostics.png")
        elif args.command == "ablation-report":
            from pc3.reporting import report_ablation

            result = report_ablation(args.metrics, output=args.output)
        else:
            raise ValueError("Comando no implementado.")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as error:
        cli.exit(2, f"Error: {error}\n")
