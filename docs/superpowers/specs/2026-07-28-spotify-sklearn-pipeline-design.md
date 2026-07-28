# Diseño: pipeline reproducible de Spotify con scikit-learn

## Objetivo

Construir un entregable académico reproducible que use
`Most_Streamed_Spotify_Songs_2024.csv` para entrenar un modelo de regresión que
prediga `Track Score`. El proyecto cubrirá las dos primeras etapas de CRISP-DM,
hará explícita la preparación de datos, mostrará el diagrama del pipeline,
permitirá instalar el código como paquete y preparará evidencia de ejecución en
Windows y Ubuntu.

## Business Understanding

El caso de negocio simula una decisión de priorización para un equipo musical:
estimar el rendimiento global de una canción a partir de señales observables en
plataformas digitales. `Track Score` será la variable objetivo. El resultado no
se presentará como causal ni como garantía de éxito, sino como una estimación
exploratoria.

El criterio de éxito técnico será superar un regresor base que siempre predice
la mediana del conjunto de entrenamiento. Se reportarán MAE, RMSE y R² sobre un
conjunto de prueba reservado.

## Data Understanding

El CSV contiene 4,600 filas y 29 columnas y requiere codificación `latin-1`.
Incluye:

- Variables numéricas reales, como `Spotify Popularity`.
- Conteos numéricos almacenados como texto con separadores de miles.
- Una fecha en `Release Date`.
- Una variable categórica/binaria en `Explicit Track`.
- Identificadores y texto de alta cardinalidad, como `ISRC`, `Track`, `Album
  Name` y `Artist`.
- Valores faltantes, dos filas duplicadas y una columna completamente vacía,
  `TIDAL Popularity`.

`All Time Rank` será excluida porque está estrechamente relacionada con el
objetivo y puede introducir fuga de información. Los identificadores y textos
de alta cardinalidad tampoco se usarán como predictores en esta primera versión.

## Arquitectura

El código instalable vivirá bajo `src/spotify_sklearn_pipeline/` y tendrá
responsabilidades separadas:

- `data.py`: localizar, extraer, validar y filtrar el CSV.
- `pipeline.py`: definir transformadores y construir el `sklearn.pipeline.Pipeline`.
- `train.py`: separar entrenamiento/prueba, ajustar, evaluar y guardar artefactos.
- `cli.py`: ofrecer un comando reproducible para ejecutar el flujo completo.

El notebook será el documento principal y consumirá el paquete; no duplicará su
lógica. `pyproject.toml` declarará las dependencias, el comando de consola y la
configuración de construcción.

## Flujo de datos

1. Extraer el CSV con codificación `latin-1`.
2. Validar que existan el objetivo y las columnas necesarias.
3. Eliminar duplicados exactos y filas sin un `Track Score` numérico válido.
4. Separar `X` e `y`.
5. Crear una partición determinista 80/20 con `random_state=42`.
6. Dentro del pipeline de sklearn:
   - convertir conteos con comas a valores numéricos;
   - imputar medianas y aplicar `log1p` a conteos no negativos;
   - imputar y escalar variables numéricas continuas;
   - convertir la fecha en año, mes, día de la semana y antigüedad aproximada;
   - imputar y codificar `Explicit Track` con one-hot encoding;
   - entrenar un `RandomForestRegressor`.
7. Comparar el modelo con `DummyRegressor(strategy="median")`.
8. Guardar el pipeline ajustado con `joblib` y las métricas en JSON.

El filtrado que cambia el número de filas se ejecutará antes de la separación
para mantener alineados `X` e `y`. Toda transformación de columnas aprenderá
sus parámetros usando únicamente el conjunto de entrenamiento, evitando fuga
de información.

## Notebook y diagrama

`notebooks/spotify_pipeline_crisp_dm.ipynb` seguirá este recorrido:

1. Resumen ejecutivo.
2. Business Understanding.
3. Data Understanding y controles de calidad.
4. Preparación y separación de datos.
5. Visualización del pipeline con `sklearn.set_config(display="diagram")`.
6. Entrenamiento y comparación con la línea base.
7. Resultados, limitaciones y conclusiones.

El notebook se ejecutará de principio a fin y conservará salidas acotadas. El
diagrama HTML generado por sklearn quedará incrustado en la salida del notebook.

## Evidencia multiplataforma

`.github/workflows/pipeline.yml` ejecutará instalación, pruebas y entrenamiento
en runners independientes `windows-latest` y `ubuntu-latest`. Cada runner
generará:

- un JSON con sistema operativo, versión de Python, versión de sklearn,
  cantidad de filas y métricas;
- una imagen PNG legible que resume la ejecución exitosa y esos datos;
- el modelo y las métricas como artefactos descargables.

Las imágenes se generarán a partir de resultados reales del runner, no de
valores escritos manualmente. Una ejecución local adicional verificará el flujo
en la computadora de desarrollo. Si no se autoriza publicar cambios al remoto,
el workflow quedará listo para producir las dos evidencias al realizar el push.

## Manejo de errores

El flujo fallará con mensajes claros cuando:

- el archivo no exista;
- no pueda decodificarse;
- falte alguna columna requerida;
- no queden suficientes filas después del filtrado;
- las variables numéricas no puedan convertirse;
- no sea posible crear el directorio de salida.

Los valores faltantes esperados no serán errores: se resolverán mediante los
imputadores dentro del pipeline.

## Pruebas y aceptación

Las pruebas automatizadas cubrirán:

- lectura con `latin-1`;
- eliminación de duplicados;
- conversión de conteos con comas;
- extracción de componentes de fecha;
- ajuste y predicción sin valores no finitos;
- creación de artefactos;
- ausencia de `All Time Rank` entre los predictores;
- ejecución del comando de consola.

El trabajo se considerará terminado cuando el paquete pueda instalarse, las
pruebas pasen, el notebook se ejecute de principio a fin, el pipeline y las
métricas se guarden correctamente y la automatización multiplataforma esté
configurada.

## Límites

Esta versión no realizará búsqueda exhaustiva de hiperparámetros, despliegue de
servicio web ni inferencia causal. Tampoco usará nombres de artistas o canciones
como predictores, para evitar una codificación de alta cardinalidad innecesaria
en un dataset pequeño.
