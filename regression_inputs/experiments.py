"""Shared data, model settings and time-ordered evaluation for the regression notebooks."""
from pathlib import Path
import hashlib
import json
import os
import warnings
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'team_project_bill/data/BROWN-2006-2025.csv'
TARGET = 'temperature_2m_mean'
MODELS = ['Baseline', 'LinearRegression', 'Ridge', 'KNN', 'Lasso', 'ElasticNet',
          'SVR', 'NN', 'RandomForest', 'Poly', 'Poly Ridge', 'Poly Lasso']
LINEAR = {'Baseline', 'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'}
SETTINGS = ['Recent 28', 'Calendar only', 'Recent + calendar', 'History + calendar',
            'History only', 'Recent + history', 'Recent + history + calendar']


def read_weather():
    frame = pd.read_csv(DATA, parse_dates=['date']).set_index('date').sort_index()
    if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max())):
        raise ValueError('Expected one row per consecutive day')
    if frame.shape[1] != 19 or frame.isna().any().any():
        raise ValueError('Expected the complete 19-field Bill dataset')
    frame['weather_code'] = (frame.weather_code >= 51).astype(int)
    # Bill orders temperatures first, then the binary code, then other fields.
    first = [TARGET, 'temperature_2m_max', 'temperature_2m_min', 'weather_code']
    return frame[first + [c for c in frame if c not in first]]


def calendar_pair(index):
    anchor = pd.to_datetime('2000-' + index.strftime('%m-%d'))
    phase = 2 * np.pi * (anchor.dayofyear.to_numpy() - 1) / 366
    return np.column_stack((np.sin(phase), np.cos(phase)))


class Inputs:
    def __init__(self, part, recent_days=28):
        if part not in (1, 2, 3):
            raise ValueError('part must be 1, 2 or 3')
        if not 1 <= recent_days <= 28 or (part != 3 and recent_days != 28):
            raise ValueError('Part 3 accepts lookbacks from 1 to 28 days')
        self.recent_days = recent_days
        self.part = part
        self.weather = read_weather()
        index = self.weather.index
        # Match Bill's holdout start even when additional warm-up removes rows.
        bill_dates = index[7:]
        self.test_start = bill_dates[int(len(bill_dates) * .8)]
        dates = index[28:]
        if part in (2, 3):
            start = index.min() + pd.DateOffset(years=5) + pd.Timedelta(days=3)
            dates = dates[dates >= start]
        self.dates = dates
        self.y = self.weather.loc[dates, TARGET].to_numpy()
        self.settings = list(range(1, 29)) if part == 1 else SETTINGS
        self.recent = [dates - pd.Timedelta(days=d) for d in range(1, 29)]
        self.history = []
        if part in (2, 3):
            for year in range(1, 6):
                # DateOffset maps Feb 29 to Feb 28 in a non-leap anchor year.
                anchor = dates - pd.DateOffset(years=year)
                self.history.extend(anchor + pd.Timedelta(days=d) for d in range(-3, 4))
        for source in self.recent + self.history:
            if not ((source < dates).all() and source.isin(index).all()):
                raise ValueError('Weather window contains unavailable or future dates')

    def matrix(self, setting, baseline=False):
        k = setting if self.part == 1 else self.recent_days
        if baseline:
            # Same comparator for every calendar/history setting.
            return np.column_stack([self.weather.loc[d, TARGET] for d in self.recent[:k]]), []
        with_calendar = self.part in (2, 3) and 'calendar' in setting.lower()
        recent = self.part == 1 or setting in ('Recent 28', 'Recent + calendar', 'Recent + history', 'Recent + history + calendar')
        history = self.part in (2, 3) and setting in ('History only', 'History + calendar', 'Recent + history', 'Recent + history + calendar')
        sources = (self.recent[:k] if recent else []) + (self.history if history else [])
        columns, binary = [], []
        if sources:
            # Feature-major ordering matches Bill for his lag-only designs.
            for name in self.weather:
                for date in sources:
                    if name == 'weather_code':
                        binary.append(len(columns))
                    columns.append(self.weather.loc[date, name].to_numpy())
        if with_calendar:
            for date in [*sources, self.dates]:
                columns.extend(calendar_pair(date).T)
        return np.column_stack(columns), binary

    def splits(self, test=False):
        dev = np.flatnonzero(self.dates < self.test_start)
        if test:
            return [(dev, np.flatnonzero(self.dates >= self.test_start))]
        return list(TimeSeriesSplit(n_splits=5).split(dev))


def estimator(name, seed):
    # Hyperparameters from Bill's fitted cells, retaining full precision.
    factories = {
        'Baseline': LinearRegression, 'LinearRegression': LinearRegression,
        'Ridge': lambda: Ridge(alpha=np.logspace(-6, 4, 50)[9]),
        'Lasso': lambda: Lasso(alpha=np.logspace(-4,-1,20)[10], max_iter=1000000),
        'ElasticNet': lambda: ElasticNet(alpha=np.logspace(-4,-1,10)[4],
                                         l1_ratio=.9, max_iter=10000, tol=1e-3),
        'KNN': lambda: KNeighborsRegressor(n_neighbors=30, weights='distance'),
        'SVR': lambda: SVR(C=1000, gamma=.0001),
        'NN': lambda: MLPRegressor(hidden_layer_sizes=(128,), alpha=10,
                                    max_iter=300, random_state=seed),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=100, max_depth=12,
                                                       max_features=.5, random_state=seed, n_jobs=2)
    }
    return factories[name]()


def predict(name, X, binary, y, train, valid, seed):
    # Identical transformation to Bill's ColumnTransformer: scale continuous
    # columns in this training fold and pass binary weather flags through last.
    binary_set = set(binary)
    numeric = [j for j in range(X.shape[1]) if j not in binary_set]
    scaler = StandardScaler()
    A = scaler.fit_transform(X[np.ix_(train, numeric)])
    B = scaler.transform(X[np.ix_(valid, numeric)])
    if binary:
        A = np.column_stack([A, X[np.ix_(train, binary)]])
        B = np.column_stack([B, X[np.ix_(valid, binary)]])
    if name.startswith('Poly'):
        from quadratic import quadratic_predict
        return quadratic_predict(name, A, y[train], B)
    return estimator(name, seed).fit(A, y[train]).predict(B)


def evaluate(inputs, models, settings=None, test=False):
    """Resume by setting/model/fold/seed. Cache identity covers inputs and code."""
    import sklearn, scipy
    files = [DATA, Path(__file__), HERE / 'quadratic.py']
    signature = hashlib.sha256()
    for file in files:
        signature.update(file.read_bytes())
    signature.update(str((np.__version__, pd.__version__, sklearn.__version__, scipy.__version__)).encode())
    cache = HERE / '.runs' / signature.hexdigest()[:16] / f'part{inputs.part}'
    if inputs.part == 3:
        cache = cache / f'lookback{inputs.recent_days}'
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for setting in (inputs.settings if settings is None else settings):
        if setting not in inputs.settings:
            raise ValueError(f'Unknown input setting: {setting}')
        for model in models:
            if model not in MODELS:
                raise ValueError(f'Unknown model: {model}')
            X = None
            key = 'reference' if inputs.part in (2, 3) and model == 'Baseline' else str(setting)
            for fold, (train, valid) in enumerate(inputs.splits(test), 1):
                seeds = range(5) if test and model in ('NN','RandomForest') else [42]
                for seed in seeds:
                    path = cache / f'{key}_{model}_{test}_{fold}_{seed}.json'
                    if not path.exists():
                        if X is None:
                            X, binary = inputs.matrix(setting, baseline=model == 'Baseline')
                        with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always', ConvergenceWarning)
                            pred = predict(model, X, binary, inputs.y, train, valid, seed)
                        err = inputs.y[valid] - pred
                        record = dict(rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.abs(err).mean()),
                                      fold=fold, seed=seed, n=len(valid), first=str(inputs.dates[valid[0]].date()),
                                      last=str(inputs.dates[valid[-1]].date()),
                                      convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught))
                        if not np.isfinite(record['rmse']):
                            raise ValueError(f'Nonfinite predictions from {model}')
                        tmp = path.with_suffix(f'.{uuid4().hex}.tmp')
                        tmp.write_text(json.dumps(record))
                        os.replace(tmp, path)
                    rows.append(dict(**json.loads(path.read_text()), setting=setting, model=model))
    return pd.DataFrame(rows)


def summarize(scores):
    return scores.groupby(['model', 'setting'], sort=False).agg(
        cv_rmse=('rmse','mean'), fold_sd=('rmse','std'), mae=('mae','mean'),
        folds=('fold','nunique'), convergence_warning=('convergence_warning','any')).reset_index()


def best_lookbacks(scores):
    """Select the shortest window among any ties in Notebook 1 validation."""
    lag_scores = scores.loc[scores.part.eq(1)].copy()
    lag_scores['setting'] = lag_scores.setting.astype(int)
    summary = summarize(lag_scores)
    counts = lag_scores.groupby(['model', 'setting']).size()
    expected = pd.MultiIndex.from_product([MODELS, range(1, 29)])
    if (not counts.reindex(expected).eq(5).all()
            or not summary.folds.eq(5).all()
            or not np.isfinite(summary.cv_rmse).all()):
        raise ValueError('Expected five validation folds for each Notebook 1 model and lookback')
    ranked = summary.sort_values(['cv_rmse', 'setting'])
    return ranked.drop_duplicates('model').set_index('model').setting.reindex(MODELS)
