# DATA1030 Weather Project

A basic framework with data, model, and analysis interfaces for seven cities. No datasets or experiment results are included.

## Setup

Run from the project directory:

```bash
conda env create -f environment.yml
conda activate weather1030
python -m pip install -e . --no-deps
jupyter lab Weather_Project.ipynb
```

## How to run

1. Edit `config.json` if you want to change the cities, dates, or thresholds.
2. Run the notebook's **Setup** cell. This only previews the configuration; **Run All does not download data while the remaining calls are commented out**.
3. In **Data interface**, remove the leading `#` from the `collect` call and run that line to fetch and save weather data through the API:

   ```python
   run_pipeline(mode=MODE, stage="collect", project_root=ROOT)
   ```

   After collection finishes, uncomment and run the `prepare` call to create model inputs and next-day targets.
4. Enable other calls only when needed:

   | Stage | When to run | Purpose |
   | --- | --- | --- |
   | `eda` | After `collect` | Explore the daily data |
   | `train` | After `prepare` | Train and evaluate candidate models |
   | `plot` | After `train` | Plot saved model results and EDA |

Each call runs only the selected stage. Later stages do not start automatically. Keep unused calls commented out.

Data are saved in `data/`, model outputs in `results/`, and figures in `figures/`.

To upload to GitHub, extract the ZIP and upload the project folder's contents, including `.gitignore` and the existing folder structure.

## Source and license

Weather data: [Open-Meteo / ERA5](https://open-meteo.com/en/docs/historical-weather-api), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Project code: [MIT](LICENSE).
