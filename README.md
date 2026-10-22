# Brown Weather Framework

A basic DATA1030 pipeline for predicting the next day's mean temperature at Brown University (41.82682, −71.40293). This version contains the framework; data collection, EDA, model experiments, and the report remain pending.

## Run

From this project folder:

```bash
conda env create -f environment.yml
conda activate weather1030
python -m pip install -e . --no-deps
jupyter lab src/Weather_Project.ipynb
```

**Run All previews the plan only.** In the notebook, uncomment `collect` to fetch data through the API, then `prepare` to construct samples. Model and analysis examples remain commented.

Environment: Python **3.14.6**; NumPy **2.4.6**, pandas **3.0.3**, scikit-learn **1.9.0**, Matplotlib **3.11.0**, requests **2.34.2**, JupyterLab **4.6.4**, ipykernel **7.3.0**. See [environment.yml](environment.yml) and [requirements.txt](requirements.txt).

## Interfaces

| Interface | Entry points | Behavior |
|---|---|---|
| Data | `run_pipeline(stage="plan", project_root=ROOT)` | Preview; explicit `"collect"` and `"prepare"` stages. |
| Models | `build_models(task="regression", seed=1030)` | Unfitted Ridge and Random Forest pipelines. |
| Analysis | `data_overview(daily)`, `plot_temperature(daily)`, `plot_scores(scores, metric)` | Helpers for supplied data or future experiment scores. |

## Data and prediction task

Collection is configured for **Open-Meteo ERA5, 2001–2025**, using `America/New_York`. Units, categories, API boundaries, and all column definitions are in the [data dictionary](data/README.md). Raw API responses and collection metadata preserve provenance. No dataset is bundled.

For target date **d**, X retains individual observations from:

- **28 recent days:** d−28 through d−1.
- **35 historical positions:** the same month/day in each of five preceding years, with **±3 days** around each anchor.

Each position contains 15 weather fields plus two month/day sine/cosine encodings, using reference year 2000 (366 days). Two target-date encodings give **63 × 17 + 2 = 1,073 raw predictors**, excluding metadata and y. These are four recent weeks plus five separate historical weeks.

February 29 anchors become February 28 in non-leap years. Windows may cross year boundaries and must precede d. Missing context dates exclude a sample; missing predictor values remain missing. Initial years provide historical context.

The only y is **`target_mean_temperature`**. Select X with the generated `feature_columns.json`. Model preprocessing treats WMO codes as categories and converts wind direction to sine/cosine; the transformed dimension can differ from 1,073.

Split by **target date**: training through **2019-12-31**, validation **2020–2022**, test **2023–2025**. This is a rolling one-day retrospective reanalysis experiment. Earlier observed validation/test days can supply later inputs; preprocessing fits on training data, tuning uses validation, and final evaluation uses the reserved test period.

## Repository and remaining work

| Folder | Contents |
|---|---|
| `src/` | Notebook, data/model/analysis modules, offline tests |
| `data/` | Static dictionary; `raw/` and `processed/` start empty |
| `figures/` | Future saved figures |
| `results/` | Future predictions, scores, and models |
| `report/` | Future report PDF |

Explicit `collect` generates the raw cache, `daily.csv`, and collection metadata. Explicit `prepare` generates `samples.csv`, `feature_columns.json`, `data_dictionary.csv`, and preparation metadata under `data/processed/`.

Before midterm submission: collect and check data; produce at least **three explained EDA figures**; complete the required report and references. The notebook lists three pending EDA ideas. This scaffold alone is not a completed midterm submission.

For GitHub, upload this folder's contents, make the repository public, and add all team members as collaborators. Offline framework checks:

```bash
python -m unittest discover -s src/tests -v
```

Weather data: [Open-Meteo / Copernicus ERA5](https://open-meteo.com/en/docs/historical-weather-api), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code: [MIT](LICENSE).
