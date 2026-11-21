# Weather regression input experiments

The notebooks in this folder compare the model's performance under different data inputs. They use the project’s [daily weather dataset](../team_project_bill/data/BROWN-2006-2025.csv) and the model lineup and hyperparameters from [main.ipynb](../team_project_bill/main.ipynb).

All scores are **mean validation RMSE in °C across five chronological folds**. Lower is better. The temperature-only baseline fits linear regression using recent daily mean temperatures. The other models use 19 weather fields per input day in settings that include weather data.

## 1. How many recent days help? · [01_lag_days.ipynb](01_lag_days.ipynb)

This notebook tests every lookback from **1 to 28 consecutive days** before the forecast date. Each model uses the same set of forecast dates at every lookback length. (The window with the lowest mean validation RMSE for each model is used in Notebook 3).

| Model | Best lookback (days) | Best RMSE | 1 day | 7 days | 28 days |
|---|---:|---:|---:|---:|---:|
| Baseline | 15 | 3.063 | 3.305 | 3.089 | 3.068 |
| Linear regression | 2 | 2.615 | 2.800 | 2.665 | 3.120 |
| Ridge | 2 | 2.614 | 2.799 | 2.657 | 3.050 |
| Lasso | 5 | 2.640 | 2.817 | 2.648 | 2.835 |
| Elastic Net | 5 | 2.639 | 2.818 | 2.652 | 2.887 |
| Polynomial regression | 1 | 3.430 | 3.430 | 6.500 | 4.289 |
| Polynomial Lasso | 5 | 2.616 | 2.738 | 2.619 | 2.682 |
| Polynomial Ridge | 2 | 2.851 | 2.980 | 3.255 | 4.071 |
| KNN | 1 | 3.137 | 3.137 | 3.861 | 4.088 |
| SVR | 4 | 2.632 | 2.808 | 2.661 | 3.005 |
| Random forest | 2 | 2.656 | 2.678 | 2.724 | 2.760 |
| Neural network | **2** | **2.584** | 2.702 | 2.719 | 3.240 |

The **two-day neural network** had the lowest validation RMSE. The 11 weather models reached their lowest errors with **one to five recent days**.

## 2. Do calendar features and prior-year weather help? · [02_calendar_and_history.ipynb](02_calendar_and_history.ipynb)

This notebook compares **seven input combinations**, using a fixed **28-day lookback** wherever recent weather is included. Comparing combinations within a model shows how the validation error changes when an input group is added or removed.

- **Recent** uses the 28 consecutive days before the forecast date.
- **History** uses a seven-day window around the same calendar date in each of the previous five years, for 35 days total.
- **Calendar** represents time of year using sine and cosine. Calendar-only uses the forecast date. Combined settings also encode the dates of the weather inputs.
- **All inputs** combines recent weather, prior-year weather, and calendar features.

The baseline uses the same 28 daily mean temperatures in every column.

| Model | Recent | Calendar | Recent + calendar | History + calendar | History | Recent + history | All inputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.206 | 3.206 | 3.206 | 3.206 | 3.206 | 3.206 | 3.206 |
| Linear regression | 3.840 | 4.153 | 4.268 | 12.282 | 10.938 | 6.517 | 17.582 |
| Ridge | 3.661 | 4.153 | 3.778 | 7.580 | 7.367 | 4.914 | 5.025 |
| Lasso | 3.055 | 4.153 | 3.055 | 5.435 | 5.491 | 3.606 | 3.605 |
| Elastic Net | 3.144 | 4.153 | 3.146 | 5.636 | 5.693 | 3.883 | 3.883 |
| Polynomial regression | 4.454 | 4.169 | 4.477 | 5.108 | 5.221 | 4.326 | 4.360 |
| Polynomial Lasso | **2.814** | 4.165 | 2.817 | 4.641 | 4.799 | 2.904 | 2.904 |
| Polynomial Ridge | 4.298 | 5.525 | 4.332 | 5.067 | 5.180 | 4.201 | 4.255 |
| KNN | 4.241 | 4.632 | 4.200 | 4.581 | 4.894 | 4.502 | 4.383 |
| SVR | 3.279 | 4.172 | 3.334 | 5.240 | 5.426 | 3.650 | 3.687 |
| Random forest | 2.927 | 4.594 | 2.922 | 4.432 | 4.463 | 2.974 | 2.986 |
| Neural network | 3.426 | 4.185 | 3.423 | 4.890 | 5.083 | 3.846 | 3.887 |

**Polynomial Lasso with recent weather alone** had the lowest validation RMSE. Adding prior-year weather to recent weather improved two of the 11 weather models.

## 3. Do those extra inputs help with each model’s best lookback? · [03_model_specific_lookback.ipynb](03_model_specific_lookback.ipynb)

This notebook repeats notebook 2’s seven combinations on **the same forecast dates and validation folds**. Wherever recent weather is included, each model uses its best lookback from Notebook 1. The baseline uses its own 15-day temperature window in every column. For the weather models, calendar-only, history-only, and history-plus-calendar scores carry over from Notebook 2 because those inputs and folds are identical.

| Model | Lookback (days) | Recent | Calendar | Recent + calendar | History + calendar | History | Recent + history | All inputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 15 | 3.192 | 3.192 | 3.192 | 3.192 | 3.192 | 3.192 | 3.192 |
| Linear regression | 2 | 2.690 | 4.153 | 2.697 | 12.282 | 10.938 | 5.965 | 5.748 |
| Ridge | 2 | 2.687 | 4.153 | 2.692 | 7.580 | 7.367 | 5.141 | 5.307 |
| Lasso | 5 | 2.730 | 4.153 | 2.726 | 5.435 | 5.491 | 3.361 | 3.357 |
| Elastic Net | 5 | 2.731 | 4.153 | 2.732 | 5.636 | 5.693 | 3.532 | 3.523 |
| Polynomial regression | 1 | 6.107 | 4.169 | 13.681 | 5.108 | 5.221 | 4.662 | 4.597 |
| Polynomial Lasso | 5 | 2.719 | 4.165 | 2.684 | 4.641 | 4.799 | 2.958 | 2.942 |
| Polynomial Ridge | 2 | 3.031 | 5.525 | 2.863 | 5.067 | 5.180 | 4.380 | 4.354 |
| KNN | 1 | 3.339 | 4.632 | 3.170 | 4.581 | 4.894 | 4.785 | 4.486 |
| SVR | 4 | 2.717 | 4.172 | 2.694 | 5.240 | 5.426 | 3.469 | 3.495 |
| Random forest | 2 | 2.769 | 4.594 | 2.730 | 4.432 | 4.463 | 2.962 | 2.976 |
| Neural network | **2** | 2.654 | 4.185 | **2.595** | 4.890 | 5.083 | 3.650 | 3.611 |

The **neural network with two recent days plus calendar features** had the lowest validation RMSE. Compared with Notebook 2’s fixed 28-day window, the selected lookbacks improved recent-only results for **10 of 11** weather models. Starting from recent weather alone, adding calendar features helped **7 of 11** models, while adding prior-year weather helped **1 of 11**.

Notebook 1 uses an earlier validation period than Notebooks 2 and 3, and the periods overlap. Compare Notebooks 2 and 3 directly because they use the same folds. The tables above report validation scores. The notebooks reserve later dates for holdout testing.
