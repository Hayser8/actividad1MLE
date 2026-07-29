# Pipeline de Spotify con scikit-learn

Proyecto reproducible para estimar `Track Score` a partir del dataset
`Most_Streamed_Spotify_Songs_2024.csv`. El entregable aplica Business
Understanding y Data Understanding de CRISP-DM, prepara diferentes tipos de
variables, separa entrenamiento y prueba, muestra el pipeline y permite
compartirlo como un paquete de Python.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

En Linux o macOS, active el entorno con `source .venv/bin/activate`.

## Ejecución

```powershell
spotify-train Most_Streamed_Spotify_Songs_2024.csv --output-dir artifacts
```

El comando produce:

- `artifacts/pipeline.joblib`: pipeline entrenado y reutilizable.
- `artifacts/metrics.json`: MAE, RMSE, R², baseline y detalles de la partición.
- `artifacts/predictions.csv`: valores reales y predicciones del test.
- `artifacts/execution_evidence.json`: entorno y resultados medidos.
- `artifacts/execution_evidence.png`: captura portable de la ejecución.

## Notebook

Abra `notebooks/spotify_pipeline_crisp_dm.ipynb` para revisar el análisis,
diagrama de sklearn y resultados ejecutados:

```powershell
python -m jupyter lab notebooks/spotify_pipeline_crisp_dm.ipynb
```

## Pruebas

```powershell
python -m pytest -q
```

## Evidencia en computadoras diferentes

El workflow `.github/workflows/pipeline.yml` usa computadoras independientes
administradas por GitHub con Windows y Ubuntu. Cada trabajo instala el paquete,
ejecuta las pruebas, entrena con el CSV y publica su propia imagen
`execution_evidence.png` junto con el modelo y las métricas. La imagen registra
el sistema operativo, las versiones y las métricas de esa ejecución real.

## Alcance CRISP-DM

- **Business Understanding:** estimar el rendimiento global de una canción para
  apoyar una priorización exploratoria. No se interpreta como causalidad.
- **Data Understanding:** documentar forma, tipos, faltantes, duplicados y
  distribución del objetivo antes de modelar.

`All Time Rank`, identificadores y texto de alta cardinalidad se excluyen para
evitar fuga de información y complejidad innecesaria.
