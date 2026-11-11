"""Gráficos e informe a partir de evidencias reales, nunca valores de ejemplo."""

from copy import deepcopy
from pathlib import Path

from pc3.io import read_json, write_json
from pc3.metrics import BUDGET_KINDS, assert_comparable, divergences


def report(metrics_paths, *, output: Path, runs=(), strict=False) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    results = [read_json(path) for path in metrics_paths]
    declared = divergences(results)
    if strict and declared:
        raise ValueError(f"--strict y hay divergencias: {declared}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = [result["label"] for result in results]
    palette = ["#9C3B79", "#368C94", "#E39E36", "#626D88"]
    colors = [palette[index % len(palette)] for index in range(len(results))]
    table = pd.DataFrame(
        [
            {
                "modelo": result["label"],
                "FID": result["fid"],
                "KID": result["kid_mean"],
                "KID_std_subconjuntos": result["kid_std"],
                "N_evaluadas": result["protocol"]["num_images"],
                "segundos_por_imagen": result["seconds_per_image"],
                "imagenes_por_segundo": result["images_per_second"],
                "VRAM_GiB": result["peak_vram_gib"],
                "checkpoint_step": result["provenance"]["step"],
                "updates_completados": (result.get("run_summary") or {}).get("completed_steps", 0),
                "horas_entrenamiento_local": (result.get("run_summary") or {}).get(
                    "wall_seconds", 0
                )
                / 3600,
            }
            for result in results
        ]
    )
    table.to_csv(output / "comparison.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].bar(labels, table["FID"], color=colors)
    axes[0].set(title="FID ↓", ylabel="FID (mismo protocolo)")
    axes[1].bar(labels, table["KID"], yerr=table["KID_std_subconjuntos"], color=colors, capsize=4)
    axes[1].set(title="KID ↓", ylabel="Media ± desviación de subconjuntos (no IC)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.15)
    figure.savefig(output / "comparison_quality.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, column, title in zip(
        axes,
        ["segundos_por_imagen", "VRAM_GiB", "horas_entrenamiento_local"],
        [
            "Inferencia ↓ (s/imagen amortizados)",
            "VRAM inferencia (GiB)",
            "Entrenamiento local (horas)",
        ],
        strict=True,
    ):
        axis.bar(labels, table[column], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
    figure.savefig(output / "comparison_cost.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    for color, result in zip(colors, results, strict=True):
        axis.scatter(result["seconds_per_image"], result["fid"], color=color, s=90)
        axis.annotate(
            result["label"],
            (result["seconds_per_image"], result["fid"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set(xlabel="Segundos por imagen ↓", ylabel="FID ↓", title="Calidad frente a coste")
    figure.savefig(output / "quality_vs_speed.png", dpi=180)
    plt.close(figure)
    if runs:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        plotted = False
        for run in map(Path, runs):
            metadata = read_json(run / "run.json")
            if metadata["manifest_sha256"] != results[0]["protocol"]["manifest_sha256"]:
                raise ValueError("El historial pertenece a otro dataset.")
            if metadata["contract"] != results[0]["provenance"]["contract"]:
                declared.append(
                    {
                        "kind": "contrato_del_historial",
                        "label": run.name,
                        "detail": "La curva proviene de otro contrato de entrenamiento.",
                    }
                )
            history = pd.read_csv(run / "history.csv")
            if "train_mse" in history:
                rows = history.dropna(subset=["train_mse"])
                axes[0].plot(
                    rows["step"], rows["train_mse"].rolling(5, min_periods=1).mean(), label=run.name
                )
            if "val_mse" in history:
                rows = history.dropna(subset=["val_mse"])
                axes[1].plot(rows["step"], rows["val_mse"], label=run.name, marker=".")
            plotted = True
        if plotted:
            axes[0].set(
                title="MSE train (media móvil de 5 registros)", xlabel="Updates", ylabel="MSE ↓"
            )
            axes[1].set(title="MSE validación con ruido fijo", xlabel="Updates", ylabel="MSE ↓")
            for axis in axes:
                axis.legend()
            figure.savefig(output / "training_curves.png", dpi=180)
        plt.close(figure)
    best = min(results, key=lambda result: result["fid"])
    warnings = list(dict.fromkeys(warning for result in results for warning in result["warnings"]))
    if any(item["kind"] in BUDGET_KINDS for item in declared):
        warnings.insert(0, "INFORME EXPLORATORIO: algún entrenamiento no completó su presupuesto.")
    lines = [
        "# Comparación DDPM anime — resultados ejecutados",
        "",
        f"Partición: **{results[0]['protocol']['split']}**, "
        f"N={results[0]['protocol']['num_images']}.",
        "",
        "| Modelo | FID ↓ | KID ↓ | Std KID | s/imagen ↓ | Updates completos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        updates = (result.get("run_summary") or {}).get("completed_steps", 0)
        lines.append(
            f"| {result['label']} | {result['fid']:.4f} | {result['kid_mean']:.6f} | "
            f"{result['kid_std']:.6f} | {result['seconds_per_image']:.4f} | {updates} |"
        )
    lines += [
        "",
        f"El menor FID observado corresponde a **{best['label']}** en esta ejecución.",
        "Esto no establece un ganador universal ni significancia estadística. Revisar KID, "
        "coste, diversidad y muestras antes de concluir.",
        "",
        "## Divergencias declaradas",
        *(
            [
                "Estas filas no son comparables sin reservas. Se declaran en vez de "
                "ocultarlas rechazando el informe; cualquier figura que enfrente estos "
                "números debe repetirlas.",
                "",
                "| Tipo | Serie | Detalle |",
                "|---|---|---|",
            ]
            + [
                f"| {item['kind']} | {item['label']} | {item['detail']} |"
                for item in declared
            ]
            if declared
            else ["Ninguna: protocolo, contrato, equipo y presupuesto coinciden."]
        ),
        "",
        "## Límites",
        *[f"- {warning}" for warning in warnings],
        "",
        "## Figuras",
        "- comparison_quality.png",
        "- comparison_cost.png",
        "- quality_vs_speed.png",
        "- training_curves.png (si se proporcionaron historiales)",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "comparison.json", results)
    write_json(output / "divergences.json", declared)
    return {
        "output": str(output),
        "models": labels,
        "num_images": results[0]["protocol"]["num_images"],
        "divergences": declared,
    }


def report_ablation(metrics_paths, *, output: Path) -> dict:
    """Ablación de pasos sobre los mismos pesos; exclusivamente en validación."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    results = [read_json(path) for path in metrics_paths]
    normalized = deepcopy(results)
    for index, result in enumerate(normalized):
        if result["protocol"]["split"] != "val":
            raise ValueError("Elegir pasos con val; no ajustar decisiones mirando test.")
        if result["generation"]["source_hash"] != results[0]["generation"]["source_hash"]:
            raise ValueError("Una ablación de pasos requiere exactamente los mismos pesos.")
        result["protocol"]["sampling_steps"] = 0
        result["label"] = str(index)
    assert_comparable(normalized)
    table = pd.DataFrame(
        [
            {
                "pasos": result["protocol"]["sampling_steps"],
                "FID": result["fid"],
                "KID": result["kid_mean"],
                "KID_std": result["kid_std"],
                "segundos_por_imagen": result["seconds_per_image"],
            }
            for result in results
        ]
    ).sort_values("pasos")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "ablation.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, column in zip(axes, ["FID", "KID", "segundos_por_imagen"], strict=True):
        axis.plot(table["pasos"], table[column], marker="o", color="#9C3B79")
        axis.set(xlabel="Pasos DDIM", ylabel=column, title=f"{column} — validación")
    figure.savefig(output / "ablation_steps.png", dpi=180)
    plt.close(figure)
    write_json(output / "ablation.json", results)
    return {"output": str(output), "steps": table["pasos"].tolist()}
