# Evidencia de ejecución

## Evidencia local incluida

`local-windows/execution_evidence.png` fue generada por una ejecución real del
comando instalable en Windows. El JSON del mismo directorio conserva los datos
estructurados usados para crear la imagen.

## Evidencia en computadoras independientes

El workflow `.github/workflows/pipeline.yml` ejecuta el mismo paquete y CSV en
dos computadoras efímeras administradas por GitHub:

- `windows-latest`
- `ubuntu-latest`

Cada trabajo publica un artefacto llamado `pipeline-evidence-<OS>-py3.11` con su
propia captura PNG, datos del entorno, métricas, predicciones y modelo. Las
capturas no contienen valores escritos manualmente: `spotify-train` las genera
después de completar entrenamiento y evaluación.

Para obtenerlas:

1. Publique este repositorio en GitHub.
2. Abra **Actions → Verificar pipeline en Windows y Ubuntu**.
3. Ejecute **Run workflow** o realice un push.
4. Descargue los dos artefactos al final de la página de la ejecución.

Este repositorio local no tiene un remoto configurado. Por esa razón no se
incluyen imágenes que pretendan ser de runners que todavía no se han ejecutado.
