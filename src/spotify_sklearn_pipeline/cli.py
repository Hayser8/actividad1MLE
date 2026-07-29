"""Interfaz de línea de comandos para entrenar y documentar el pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from spotify_sklearn_pipeline.evidence import write_execution_evidence
from spotify_sklearn_pipeline.train import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el pipeline de Track Score y genera modelo, métricas "
            "y evidencia de ejecución."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Ruta del CSV de Spotify.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directorio para el modelo, métricas y evidencia.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para la separación y el modelo.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_and_evaluate(
        args.csv_path,
        args.output_dir,
        random_state=args.random_state,
    )
    evidence_json, evidence_png = write_execution_evidence(
        result, args.output_dir
    )

    metrics = result.metrics["model"]
    print("Pipeline ejecutado correctamente.")
    print(f"Filas útiles: {result.audit['output_rows']}")
    print(
        f"MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"R²={metrics['r2']:.4f}"
    )
    print(f"Modelo: {result.model_path}")
    print(f"Evidencia JSON: {evidence_json}")
    print(f"Evidencia PNG: {evidence_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
