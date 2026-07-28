# Spotify sklearn Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable, tested and documented sklearn regression pipeline for `Most_Streamed_Spotify_Songs_2024.csv`, with an executed CRISP-DM notebook and Windows/Ubuntu evidence automation.

**Architecture:** A small `src` package owns extraction, filtering, feature engineering, model construction and training. The notebook imports that package as the single source of truth. A CLI and GitHub Actions matrix make the same workflow runnable locally and on independent operating-system runners.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn, joblib, matplotlib, Jupyter, pytest, hatchling, GitHub Actions.

## Global Constraints

- Predict `Track Score` as a regression target.
- Exclude `All Time Rank`, identifiers and high-cardinality text from predictors.
- Read the source CSV with `latin-1`.
- Use an 80/20 split and `random_state=42`.
- Report MAE, RMSE and R² against a median `DummyRegressor`.
- Keep filtering before the split and learned preprocessing inside the sklearn pipeline.
- Execute the notebook from top to bottom.
- Prepare independent `windows-latest` and `ubuntu-latest` evidence jobs.

---

## File Map

- `pyproject.toml`: build metadata, dependencies, console command and pytest settings.
- `src/spotify_sklearn_pipeline/__init__.py`: public package version and exports.
- `src/spotify_sklearn_pipeline/data.py`: schema, CSV extraction, validation and row filtering.
- `src/spotify_sklearn_pipeline/pipeline.py`: custom transformers and sklearn pipeline factory.
- `src/spotify_sklearn_pipeline/train.py`: deterministic split, fit, baseline, metrics and artifacts.
- `src/spotify_sklearn_pipeline/evidence.py`: portable JSON/PNG execution evidence.
- `src/spotify_sklearn_pipeline/cli.py`: command-line entry point.
- `tests/`: focused data, transformer, training and CLI tests.
- `notebooks/spotify_pipeline_crisp_dm.ipynb`: executed reader-facing assignment.
- `.github/workflows/pipeline.yml`: Windows/Ubuntu installation, test, training and artifact matrix.
- `README.md`: installation, execution, deliverables and CI evidence instructions.

### Task 1: Package foundation and data contract

**Files:**
- Create: `pyproject.toml`
- Create: `src/spotify_sklearn_pipeline/__init__.py`
- Create: `src/spotify_sklearn_pipeline/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces: `load_dataset(path: str | Path) -> pd.DataFrame`
- Produces: `filter_training_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]`
- Produces: constants `TARGET_COLUMN`, `MODEL_FEATURES`, `LEAKAGE_COLUMNS`

- [ ] **Step 1: Write failing extraction and filtering tests**

```python
def test_load_dataset_reads_latin1(tmp_path):
    path = tmp_path / "songs.csv"
    pd.DataFrame({"Track": ["Canción"], "Track Score": [1.0]}).to_csv(
        path, index=False, encoding="latin-1"
    )
    assert load_dataset(path).loc[0, "Track"] == "Canción"


def test_filter_training_rows_deduplicates_and_excludes_leakage():
    row = valid_model_row()
    data = pd.DataFrame([row, row])
    X, y, audit = filter_training_rows(data)
    assert len(X) == len(y) == 1
    assert "All Time Rank" not in X
    assert audit["duplicates_removed"] == 1
```

- [ ] **Step 2: Run `pytest tests/test_data.py -v` and confirm import failures**

- [ ] **Step 3: Add package metadata and exact dependencies**

```toml
[project]
name = "spotify-sklearn-pipeline"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "joblib>=1.3",
  "matplotlib>=3.8",
  "numpy>=1.26",
  "pandas>=2.0",
  "scikit-learn>=1.4",
]

[project.optional-dependencies]
dev = ["jupyter>=1.0", "nbclient>=0.10", "nbformat>=5.10", "pytest>=8.0"]

[project.scripts]
spotify-train = "spotify_sklearn_pipeline.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Implement schema validation, latin-1 extraction and filtering**

```python
def load_dataset(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")
    return pd.read_csv(csv_path, encoding="latin-1")


def filter_training_rows(data: pd.DataFrame):
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")
    filtered = data.drop_duplicates().copy()
    filtered[TARGET_COLUMN] = pd.to_numeric(filtered[TARGET_COLUMN], errors="coerce")
    filtered = filtered.loc[filtered[TARGET_COLUMN].notna()]
    if len(filtered) < 10:
        raise ValueError("Se requieren al menos 10 filas válidas para entrenar.")
    X = filtered.loc[:, MODEL_FEATURES].copy()
    y = filtered[TARGET_COLUMN].copy()
    audit = {
        "input_rows": len(data),
        "duplicates_removed": len(data) - len(data.drop_duplicates()),
        "output_rows": len(filtered),
    }
    return X, y, audit
```

- [ ] **Step 5: Run `pytest tests/test_data.py -v` and commit the passing data contract**

### Task 2: Feature transformers and estimator pipeline

**Files:**
- Create: `src/spotify_sklearn_pipeline/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `MODEL_FEATURES` from `data.py`
- Produces: `CommaNumberTransformer.fit/transform`
- Produces: `DateFeatureTransformer.fit/transform`
- Produces: `build_model_pipeline(random_state: int = 42) -> Pipeline`

- [ ] **Step 1: Write failing transformer tests**

```python
def test_comma_number_transformer_parses_and_preserves_missing():
    values = pd.DataFrame({"streams": ["1,234", None, "bad"]})
    result = CommaNumberTransformer().fit_transform(values)
    assert result[0, 0] == 1234
    assert np.isnan(result[1, 0])
    assert np.isnan(result[2, 0])


def test_date_transformer_returns_four_finite_features():
    values = pd.DataFrame({"Release Date": ["4/26/2024", None]})
    result = DateFeatureTransformer(reference_year=2024).fit_transform(values)
    assert result.shape == (2, 4)
    assert result[0].tolist() == [2024, 4, 4, 0]
```

- [ ] **Step 2: Run `pytest tests/test_pipeline.py -v` and confirm failures**

- [ ] **Step 3: Implement sklearn-compatible numeric and date transformers**

```python
class CommaNumberTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        return frame.apply(
            lambda column: pd.to_numeric(
                column.astype("string").str.replace(",", "", regex=False),
                errors="coerce",
            )
        ).to_numpy(dtype=float)


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_year=2024):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dates = pd.to_datetime(pd.DataFrame(X).iloc[:, 0], errors="coerce")
        return np.column_stack([
            dates.dt.year,
            dates.dt.month,
            dates.dt.dayofweek,
            self.reference_year - dates.dt.year,
        ]).astype(float)
```

- [ ] **Step 4: Compose numeric-count, continuous, date and categorical branches**

```python
preprocessor = ColumnTransformer([
    ("counts", Pipeline([
        ("parse", CommaNumberTransformer()),
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ]), COUNT_COLUMNS),
    ("continuous", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), CONTINUOUS_COLUMNS),
    ("date", Pipeline([
        ("parts", DateFeatureTransformer(reference_year=2024)),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ]), DATE_COLUMNS),
    ("categorical", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL_COLUMNS),
])
return Pipeline([
    ("prepare", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )),
])
```

- [ ] **Step 5: Test fit/predict on representative missing values and commit**

### Task 3: Training orchestration, baseline and model artifacts

**Files:**
- Create: `src/spotify_sklearn_pipeline/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `load_dataset`, `filter_training_rows`, `build_model_pipeline`
- Produces: `TrainingResult` dataclass
- Produces: `train_and_evaluate(csv_path, output_dir, random_state=42, test_size=0.2) -> TrainingResult`

- [ ] **Step 1: Write a failing end-to-end training test**

```python
def test_train_and_evaluate_creates_reusable_artifacts(tmp_path):
    csv_path = make_training_csv(tmp_path, rows=40)
    result = train_and_evaluate(csv_path, tmp_path / "artifacts")
    assert result.model_path.is_file()
    assert result.metrics_path.is_file()
    assert set(result.metrics["model"]) == {"mae", "rmse", "r2"}
    loaded = joblib.load(result.model_path)
    assert len(loaded.predict(result.X_test.head(2))) == 2
```

- [ ] **Step 2: Run the focused test and confirm failure**

- [ ] **Step 3: Implement deterministic splitting and model/baseline metrics**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)
pipeline = build_model_pipeline(random_state=random_state)
pipeline.fit(X_train, y_train)
baseline = DummyRegressor(strategy="median").fit(X_train, y_train)

def regression_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
        "r2": r2_score(actual, predicted),
    }
```

- [ ] **Step 4: Serialize `pipeline.joblib` and a JSON-safe `metrics.json`**

- [ ] **Step 5: Run `pytest tests/test_train.py -v`, load the saved model and commit**

### Task 4: CLI and portable evidence image

**Files:**
- Create: `src/spotify_sklearn_pipeline/evidence.py`
- Create: `src/spotify_sklearn_pipeline/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `TrainingResult` from `train.py`
- Produces: `write_execution_evidence(result, output_dir) -> tuple[Path, Path]`
- Produces: console command `spotify-train CSV --output-dir DIR`

- [ ] **Step 1: Write failing CLI and evidence tests**

```python
def test_cli_trains_and_writes_evidence(sample_csv, tmp_path):
    completed = runner.invoke([
        str(sample_csv), "--output-dir", str(tmp_path / "run")
    ])
    assert completed == 0
    assert (tmp_path / "run" / "execution_evidence.json").is_file()
    assert (tmp_path / "run" / "execution_evidence.png").is_file()
```

- [ ] **Step 2: Run the CLI test and confirm failure**

- [ ] **Step 3: Implement argparse parameters and invoke training**

```python
parser.add_argument("csv_path", type=Path)
parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
parser.add_argument("--random-state", type=int, default=42)
result = train_and_evaluate(
    args.csv_path, args.output_dir, random_state=args.random_state
)
write_execution_evidence(result, args.output_dir)
```

- [ ] **Step 4: Render evidence using measured platform and result data**

```python
payload = {
    "status": "SUCCESS",
    "platform": platform.platform(),
    "python": platform.python_version(),
    "sklearn": sklearn.__version__,
    "rows": result.audit["output_rows"],
    "metrics": result.metrics["model"],
}
```

Use matplotlib to draw the payload as a terminal-style card and save it as
`execution_evidence.png`; serialize the same payload to JSON.

- [ ] **Step 5: Run the installed console command against the real CSV and commit**

### Task 5: Executed CRISP-DM notebook and sklearn diagram

**Files:**
- Create: `scripts/build_notebook.py`
- Create: `notebooks/spotify_pipeline_crisp_dm.ipynb`

**Interfaces:**
- Consumes: package data and training interfaces
- Produces: an executed notebook with embedded outputs and sklearn HTML diagram

- [ ] **Step 1: Build the notebook with `nbformat`**

Create concise markdown and code cells in this order:

```text
# Pipeline de Machine Learning con scikit-learn
## tl;dr
## 1. Business Understanding
## 2. Data Understanding
## 3. Preparación y separación
## 4. Pipeline y diagrama
## 5. Resultados
## 6. Takeaways y limitaciones
```

- [ ] **Step 2: Add bounded quality checks**

Include shape, duplicate count, missingness sorted descending, dtypes, target
summary and small visualizations. Do not print all 4,600 rows or all raw values.

- [ ] **Step 3: Display the actual estimator**

```python
from sklearn import set_config
set_config(display="diagram")
pipeline = build_model_pipeline()
pipeline
```

- [ ] **Step 4: Train through the package and compare model versus baseline**

Show MAE, RMSE and R² in a compact dataframe and include a predicted-versus-real
scatter plot.

- [ ] **Step 5: Execute top-to-bottom**

Run:

```bash
python scripts/build_notebook.py
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/spotify_pipeline_crisp_dm.ipynb
```

Confirm the notebook has no error outputs, includes the sklearn diagram MIME
output and contains concrete values in the summary.

- [ ] **Step 6: Commit the executed notebook and builder**

### Task 6: Multiplatform workflow and handoff documentation

**Files:**
- Create: `.github/workflows/pipeline.yml`
- Create: `README.md`
- Create: `.gitignore`

**Interfaces:**
- Consumes: package CLI, tests and real CSV
- Produces: Windows and Ubuntu downloadable evidence artifacts

- [ ] **Step 1: Add a two-runner matrix**

```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest, windows-latest]
    python-version: ["3.11"]
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python-version }}
  - run: python -m pip install -e ".[dev]"
  - run: python -m pytest -q
  - run: spotify-train Most_Streamed_Spotify_Songs_2024.csv --output-dir artifacts
  - uses: actions/upload-artifact@v4
    with:
      name: pipeline-evidence-${{ runner.os }}
      path: artifacts/
```

- [ ] **Step 2: Document local installation and execution**

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
spotify-train Most_Streamed_Spotify_Songs_2024.csv --output-dir artifacts
python -m jupyter lab notebooks/spotify_pipeline_crisp_dm.ipynb
```

- [ ] **Step 3: Explain the CRISP-DM scope and evidence provenance**

State that CI evidence is generated independently on GitHub-hosted Windows and
Ubuntu runners, while `artifacts/execution_evidence.png` records the local run.

- [ ] **Step 4: Run YAML syntax and documentation link checks**

- [ ] **Step 5: Commit workflow and documentation**

### Task 7: Final reproducibility verification

**Files:**
- Modify only files that fail verification.

**Interfaces:**
- Consumes: complete project
- Produces: clean test, build, notebook and CLI verification evidence

- [ ] **Step 1: Install the package editable with development dependencies**

Run `python -m pip install -e ".[dev]"`.

- [ ] **Step 2: Run all automated tests**

Run `python -m pytest -q` and require zero failures.

- [ ] **Step 3: Build distribution artifacts**

Run `python -m build` and inspect that both wheel and source distribution are
created under `dist/`.

- [ ] **Step 4: Rerun the real training command**

Run:

```bash
spotify-train Most_Streamed_Spotify_Songs_2024.csv --output-dir artifacts
```

Require `pipeline.joblib`, `metrics.json`, `execution_evidence.json` and
`execution_evidence.png`.

- [ ] **Step 5: Re-execute the notebook**

Run nbconvert in place and programmatically verify there are no error outputs.

- [ ] **Step 6: Run `git diff --check` and inspect `git status --short`**

- [ ] **Step 7: Commit any verification fixes and report exact paths**
