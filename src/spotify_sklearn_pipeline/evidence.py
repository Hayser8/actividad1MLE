"""Generación de evidencia portable a partir de una ejecución real."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import sklearn

from spotify_sklearn_pipeline.train import TrainingResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def build_evidence_payload(result: TrainingResult) -> dict[str, Any]:
    """Build a JSON-safe environment and result summary."""
    return {
        "status": "SUCCESS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_os": os.getenv("RUNNER_OS", platform.system()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "rows": result.audit["output_rows"],
        "split": result.metrics["split"],
        "metrics": result.metrics["model"],
        "baseline_metrics": result.metrics["baseline"],
        "mae_improvement_percent": result.metrics["mae_improvement_percent"],
        "artifacts": {
            "model": result.model_path.name,
            "metrics": result.metrics_path.name,
            "predictions": result.predictions_path.name,
        },
    }


def write_execution_evidence(
    result: TrainingResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write matching JSON and screenshot-style PNG evidence."""
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = build_evidence_payload(result)

    json_path = artifact_dir / "execution_evidence.json"
    png_path = artifact_dir / "execution_evidence.png"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = payload["metrics"]
    lines = [
        "$ spotify-train Most_Streamed_Spotify_Songs_2024.csv",
        "",
        "[SUCCESS] Pipeline ejecutado correctamente",
        f"Runner OS   : {payload['runner_os']}",
        f"Plataforma  : {payload['platform']}",
        f"Python      : {payload['python']}",
        f"scikit-learn: {payload['sklearn']}",
        f"Filas útiles: {payload['rows']:,}",
        (
            "Partición   : "
            f"{payload['split']['train_rows']:,} train / "
            f"{payload['split']['test_rows']:,} test"
        ),
        "",
        f"MAE         : {metrics['mae']:.4f}",
        f"RMSE        : {metrics['rmse']:.4f}",
        f"R²          : {metrics['r2']:.4f}",
        (
            "Mejora MAE  : "
            f"{payload['mae_improvement_percent']:.2f}% vs. baseline"
        ),
        "",
        f"Generado UTC: {payload['generated_at_utc']}",
    ]

    figure = plt.figure(figsize=(12, 6), dpi=120, facecolor="#0b1020")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_facecolor("#0b1020")
    axis.axis("off")
    axis.text(
        0.045,
        0.94,
        "EVIDENCIA DE EJECUCIÓN · SPOTIFY SKLEARN PIPELINE",
        color="#7dd3fc",
        fontsize=17,
        fontweight="bold",
        family="monospace",
        va="top",
    )
    axis.text(
        0.045,
        0.865,
        "\n".join(lines),
        color="#e2e8f0",
        fontsize=11.5,
        family="monospace",
        linespacing=1.35,
        va="top",
    )
    figure.savefig(
        png_path,
        facecolor=figure.get_facecolor(),
        bbox_inches=None,
    )
    plt.close(figure)
    return json_path, png_path
