import json
from pathlib import Path

import pandas as pd
from PIL import Image

from spotify_sklearn_pipeline.cli import main
from spotify_sklearn_pipeline.data import MODEL_FEATURES


def make_cli_csv(tmp_path: Path, rows: int = 30) -> Path:
    records: list[dict[str, object]] = []
    for index in range(rows):
        row = {column: f"{5_000 + index * 97:,}" for column in MODEL_FEATURES}
        row["Release Date"] = f"{(index % 12) + 1}/1/{2021 + index % 4}"
        row["Explicit Track"] = index % 2
        row["Spotify Popularity"] = 50 + index
        row["Apple Music Playlist Count"] = index + 1
        row["Deezer Playlist Count"] = index + 2
        row["Amazon Playlist Count"] = index + 3
        row["Track Score"] = 40 + index * 3
        records.append(row)
    csv_path = tmp_path / "cli-songs.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="latin-1")
    return csv_path


def test_cli_trains_and_writes_real_execution_evidence(tmp_path: Path) -> None:
    csv_path = make_cli_csv(tmp_path)
    output_dir = tmp_path / "run"

    exit_code = main([str(csv_path), "--output-dir", str(output_dir)])

    assert exit_code == 0
    evidence_json = output_dir / "execution_evidence.json"
    evidence_png = output_dir / "execution_evidence.png"
    assert evidence_json.is_file()
    assert evidence_png.is_file()

    payload = json.loads(evidence_json.read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS"
    assert payload["rows"] == 30
    assert payload["metrics"]["mae"] >= 0

    with Image.open(evidence_png) as image:
        assert image.width >= 1000
        assert image.height >= 500
