"""Compartir protocolo, notas y evidencias ya validadas; no ejecutar modelos."""

import argparse
import base64
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def export(require_results=False):
    shared = runpy.run_path(str(ROOT / "tools/build_presentation.py"))
    config = shared["load_config"](ROOT / "configs/rtx3060_8gb.toml")
    evidence = shared["load_evidence"](ROOT, config)
    if require_results and not evidence:
        raise FileNotFoundError("Faltan los resultados reales de sample/evaluate/report.")
    cases = []
    for item in evidence:
        cases.append({
            "label": item["label"], "metrics": item["metrics"],
            "images": ["data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
                       for path in item["images"]],
        })
    payload = {
        "status": "executed_results" if cases else "pending_results",
        "slides": shared["talk_plan"](evidence, config), "evidence": cases,
        "config": {
            "batchSize": config.batch_size, "accumulation": config.accumulation_steps,
            "maxSteps": config.max_steps, "sampleCount": config.sample_count,
            "samplingSteps": config.sampling_steps, "pretrainedId": config.pretrained_id,
            "revision": config.pretrained_revision,
        },
    }
    path = ROOT / "presentacion_web/src/content.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Contenido compartido: {path.relative_to(ROOT)} ({payload['status']}). Sin ejecutar modelos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-results", action="store_true")
    args = parser.parse_args()
    export(args.require_results)
