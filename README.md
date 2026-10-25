# Brown Weather Project

DATA1030 project for **next-day mean-temperature prediction** near Brown University (41.82682, -71.40293). Data collection and preliminary EDA are complete. Ridge and Random Forest pipelines are defined; model training and evaluation remain future work.

[Midterm report](report/Report.pdf) · [Data dictionary](data/README.md)

## Run

From the repository root:

```bash
conda env create -f environment.yml
conda activate weather1030
python -m pip install -e . --no-deps
jupyter lab src/Weather_Project.ipynb
```

The command-line interface offers three explicit stages:

```bash
python -m weather_project --stage plan
python -m weather_project --stage collect
python -m weather_project --stage prepare
```

The default `plan` only previews configuration. `collect` reuses the bundled raw response when its query matches `config.json`; otherwise it requests API data. `prepare` reads local daily data and creates samples and feature documentation without training models. Generated `samples.csv` is not committed; recreate it locally when needed.

Environment: Python **3.14.6**; NumPy **2.4.6**, pandas **3.0.3**, scikit-learn **1.9.0**, Matplotlib **3.11.0**, requests **2.34.2**, JupyterLab **4.6.4**, ipykernel **7.3.0**. See [environment.yml](environment.yml) and [requirements.txt](requirements.txt).

## Data and task

Both collections use Open-Meteo ERA5 for Brown University, with `America/New_York` as the requested timezone. They have different scopes:

| Collection | Dates, inclusive | Records | Weather fields | Location |
| --- | --- | ---: | ---: | --- |
| Modeling | 2001-01-01–2025-12-31 | 9,131 | 15 | `data/raw/`, `data/processed/daily.csv` |
| Exploratory | 2006-01-01–2025-01-01 | 6,941 | 19 | `team_project_bill/data/BROWN-2006-2025.csv` |

The eight original Bill EDA figures are preserved unchanged in `figures/EDA/`. They describe the full exploratory period, including years assigned to validation and testing; they are not training-only EDA.

For target day **d**, the only y is `target_mean_temperature` (°C). X contains 28 preceding days and seven-day windows around the corresponding month/day in each of five previous years: **63 positions**. Each holds 15 weather fields and two calendar encodings; two target-date encodings produce **63 × 17 + 2 = 1,073 raw predictors**, excluding metadata and y. WMO one-hot encoding and circular wind conversion change the transformed dimension.

All source dates precede d. February 29 anchors become February 28 in non-leap years. Complete contexts provide 7,302 eligible target dates, split chronologically:

| Split | Target dates | Samples |
| --- | --- | ---: |
| Training | 2006-01-04–2017-12-31 | 4,380 |
| Validation | 2018-01-01–2021-12-31 | 1,461 |
| Test | 2022-01-01–2025-12-31 | 1,461 |

This is a retrospective reanalysis study; live deployment would require timely weather observations. Future fitting should learn preprocessing from training data and use temporal validation for model and feature choices.

## Repository

- `src/`: notebook, reusable modules, and offline tests.
- `data/`: raw cache, daily table, metadata, and column documentation.
- `figures/EDA/`: eight original explanatory figures.
- `results/`: placeholder for future predictions, scores, and models.
- `report/Report.pdf`: current midterm report.
- `team_project_bill/`: the team's original exploratory collection and scripts.

Offline framework checks: `python -m unittest discover -s src/tests -v`.

Weather data by [Open-Meteo.com](https://open-meteo.com/) and Copernicus ERA5, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data are transformed into project features and figures. Code uses the [MIT License](LICENSE).
