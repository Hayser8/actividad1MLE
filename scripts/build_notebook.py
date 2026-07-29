"""Build the reader-facing CRISP-DM notebook with nbformat."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "spotify_pipeline_crisp_dm.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook.cells = [
        markdown(
            """
            # Pipeline de Machine Learning con scikit-learn

            **Dataset:** `Most_Streamed_Spotify_Songs_2024.csv`  
            **Objetivo:** estimar `Track Score` mediante regresión.  
            **Metodología:** primeras dos etapas de CRISP-DM y un flujo
            reproducible de preparación, entrenamiento y evaluación.
            """
        ),
        markdown(
            """
            ## tl;dr

            La ejecución validada usa **4,598 canciones** después de eliminar
            dos duplicados. El pipeline obtiene **MAE 9.07**, **RMSE 19.46** y
            **R² 0.732** en 920 filas de prueba. Su MAE mejora **53.44%** frente
            a un regresor base que siempre predice la mediana.

            El resultado es exploratorio: identifica asociaciones en las
            señales de plataformas, no demuestra qué causa el éxito de una
            canción.
            """
        ),
        markdown(
            """
            ## 1. Contexto y métodos

            ### Business Understanding

            Un equipo de música necesita priorizar canciones con mayor
            potencial de rendimiento digital. Usaremos `Track Score` como una
            medida agregada de rendimiento y construiremos un modelo que la
            estime a partir de conteos, popularidad, fecha de lanzamiento y si
            la canción es explícita.

            **Criterio de éxito:** el modelo debe reducir el MAE frente a
            `DummyRegressor(strategy="median")`. También reportamos RMSE y R².

            ### Supuestos clave

            - `Track Score` es una aproximación útil al rendimiento global.
            - Las filas se consideran observaciones independientes.
            - `All Time Rank` se excluye porque está estrechamente ligado al
              objetivo y generaría fuga de información.
            - Nombres, artista, álbum e ISRC se excluyen por su alta cardinalidad.
            - La evaluación describe este CSV; no garantiza generalización a
              lanzamientos futuros.
            """
        ),
        code(
            """
            from pathlib import Path
            import json

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from sklearn import set_config

            from spotify_sklearn_pipeline.data import (
                CATEGORICAL_COLUMNS,
                CONTINUOUS_COLUMNS,
                COUNT_COLUMNS,
                DATE_COLUMNS,
                LEAKAGE_COLUMNS,
                MODEL_FEATURES,
                filter_training_rows,
                load_dataset,
            )
            from spotify_sklearn_pipeline.pipeline import build_model_pipeline
            from spotify_sklearn_pipeline.train import train_and_evaluate

            CSV_NAME = "Most_Streamed_Spotify_Songs_2024.csv"
            current = Path.cwd().resolve()
            PROJECT_ROOT = next(
                candidate
                for candidate in [current, *current.parents]
                if (candidate / CSV_NAME).is_file()
            )
            CSV_PATH = PROJECT_ROOT / CSV_NAME
            NOTEBOOK_ARTIFACTS = PROJECT_ROOT / "artifacts" / "notebook"

            pd.set_option("display.max_columns", 12)
            plt.style.use("seaborn-v0_8-whitegrid")
            print(f"Proyecto: {PROJECT_ROOT}")
            print(f"Fuente: {CSV_PATH.name}")
            """
        ),
        markdown(
            """
            ## 2. Data Understanding

            La extracción usa `latin-1`, que es la codificación requerida por
            el archivo. Primero revisamos tamaño, muestra, tipos, duplicados y
            valores faltantes antes de construir el modelo.
            """
        ),
        code(
            """
            data = load_dataset(CSV_PATH)
            print(f"Dimensiones: {data.shape[0]:,} filas × {data.shape[1]} columnas")
            print(f"Duplicados exactos: {data.duplicated().sum():,}")
            display(data.head(5))
            """
        ),
        code(
            """
            column_profile = pd.DataFrame({
                "tipo": data.dtypes.astype(str),
                "faltantes": data.isna().sum(),
                "porcentaje_faltante": data.isna().mean().mul(100).round(2),
                "valores_unicos": data.nunique(dropna=True),
            }).sort_values("porcentaje_faltante", ascending=False)
            display(column_profile)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            missing_top = column_profile.head(10).sort_values("porcentaje_faltante")
            axes[0].barh(missing_top.index, missing_top["porcentaje_faltante"], color="#1DB954")
            axes[0].set_title("Columnas con mayor porcentaje de faltantes")
            axes[0].set_xlabel("% faltante")

            axes[1].hist(data["Track Score"], bins=35, color="#4F46E5", edgecolor="white")
            axes[1].set_title("Distribución de Track Score")
            axes[1].set_xlabel("Track Score")
            axes[1].set_ylabel("Canciones")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ### Hallazgos de calidad

            - Los conteos de plataformas llegan como texto con comas y deben
              convertirse a números.
            - `Release Date` requiere extracción de componentes temporales.
            - Hay faltantes importantes; `TIDAL Popularity` está totalmente
              vacía y no se utiliza.
            - Existen dos duplicados exactos.
            - `Track Score` presenta una cola derecha; por eso MAE es una métrica
              principal más interpretable que depender solo de RMSE.
            """
        ),
        markdown(
            """
            ## 3. Preparación y separación

            El filtrado elimina duplicados y objetivos inválidos. Después se
            reserva 20% para prueba con semilla 42. Las imputaciones, escalado y
            codificación se aprenden **solo** al ajustar el conjunto de
            entrenamiento porque viven dentro del pipeline.
            """
        ),
        code(
            """
            X, y, audit = filter_training_rows(data)
            preparation_summary = pd.DataFrame({
                "grupo": ["Conteos", "Continuas", "Fecha", "Categóricas"],
                "columnas": [
                    len(COUNT_COLUMNS),
                    len(CONTINUOUS_COLUMNS),
                    len(DATE_COLUMNS),
                    len(CATEGORICAL_COLUMNS),
                ],
                "tratamiento": [
                    "Quitar comas → mediana → log1p → escala",
                    "Mediana → escala",
                    "Año/mes/día/antigüedad → imputación → escala",
                    "Moda → one-hot",
                ],
            })
            print("Auditoría del filtrado:", audit)
            print("Columnas excluidas:", ", ".join(LEAKAGE_COLUMNS))
            display(preparation_summary)
            """
        ),
        markdown(
            """
            ## 4. Pipeline y diagrama

            El objeto siguiente es el pipeline real entrenado. El diagrama
            generado por sklearn permite expandir cada rama del
            `ColumnTransformer` y comprobar cómo se manejan los distintos tipos
            de variables.
            """
        ),
        code(
            """
            result = train_and_evaluate(CSV_PATH, NOTEBOOK_ARTIFACTS)
            set_config(display="diagram")
            result.pipeline
            """
        ),
        markdown(
            """
            ## 5. Resultados

            La comparación usa exactamente las mismas 920 canciones de prueba
            para el bosque aleatorio y la línea base.
            """
        ),
        code(
            """
            metrics_table = pd.DataFrame({
                "Random Forest": result.metrics["model"],
                "Baseline (mediana)": result.metrics["baseline"],
            }).T
            display(metrics_table.style.format({
                "mae": "{:.3f}", "rmse": "{:.3f}", "r2": "{:.3f}"
            }))
            print(
                "Mejora de MAE frente al baseline: "
                f"{result.metrics['mae_improvement_percent']:.2f}%"
            )
            print("Partición:", result.metrics["split"])
            """
        ),
        code(
            """
            predictions = result.predictions
            maximum = max(
                predictions["actual_track_score"].max(),
                predictions["predicted_track_score"].max(),
            )
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(
                predictions["actual_track_score"],
                predictions["predicted_track_score"],
                alpha=0.5,
                color="#4F46E5",
                edgecolor="none",
            )
            ax.plot([0, maximum], [0, maximum], "--", color="#DC2626", label="Predicción perfecta")
            ax.set(
                title="Track Score real frente a predicción",
                xlabel="Track Score real",
                ylabel="Track Score predicho",
            )
            ax.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 6. Takeaways y limitaciones

            - El pipeline reduce el MAE de **19.49 a 9.07**, una mejora de
              **53.44%** frente a la mediana.
            - Un R² de **0.732** indica que el modelo captura una parte
              sustancial de la variación del conjunto de prueba, aunque aún hay
              errores grandes visibles en canciones con scores extremos.
            - El pipeline empaqueta conversión de conteos, imputación,
              ingeniería de fecha, one-hot encoding y el regresor; por eso puede
              reutilizarse con `joblib` sin repetir preparación manual.
            - La separación es aleatoria, no temporal. Una validación futura
              debería probar lanzamientos posteriores y revisar estabilidad por
              artista y plataforma.
            - Las señales de plataformas pueden formar parte del cálculo del
              propio `Track Score`; el modelo sirve para demostrar el flujo
              técnico, no para afirmar causalidad.
            """
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook creado: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
