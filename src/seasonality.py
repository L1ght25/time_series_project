import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

import config


def compute_seasonality_strength(values, period=None):
    if period is None:
        period = config.SEASON_PERIOD
    if len(values) < 2 * period + 1:
        return 0.0
    try:
        result = STL(values, period=period, robust=True).fit()
        var_r = np.var(result.resid)
        var_sr = np.var(result.seasonal + result.resid)
        if var_sr < 1e-10:
            return 0.0
        return max(0.0, 1.0 - var_r / var_sr)
    except Exception:
        return 0.0


def classify_seasonality(strength, bins=None):
    if bins is None:
        bins = config.SEASONALITY_BINS
    for cat, (lo, hi) in bins.items():
        if lo <= strength < hi:
            return cat
    return 'strong'


def compute_acf_values(values, nlags=None):
    if nlags is None:
        nlags = min(config.SEASON_PERIOD * 3, len(values) // 2 - 1)
    try:
        return acf(values, nlags=nlags, fft=True)
    except Exception:
        return np.zeros(nlags + 1)


def analyze_all_series(series_dict):
    info = {}
    for uid, s in tqdm(series_dict.items(), desc='Seasonality (STL)'):
        st = compute_seasonality_strength(s['train_values'])
        info[uid] = {'strength': st, 'category': classify_seasonality(st)}
    return info


def analyze_from_df(train_df):
    info = {}
    grouped = dict(list(train_df.sort_values('ds').groupby('unique_id')))
    for uid, grp in tqdm(grouped.items(), desc='Seasonality (STL)'):
        vals = grp['y'].values.astype(float)
        st = compute_seasonality_strength(vals)
        info[uid] = {'strength': st, 'category': classify_seasonality(st)}
    return info


def sample_series_ids(seasonality_info, n_series=None, seed=None):
    if n_series is None:
        n_series = config.N_SERIES
    if seed is None:
        seed = config.RANDOM_SEED
    rng = np.random.RandomState(seed)

    by_cat = {}
    for uid, info in seasonality_info.items():
        by_cat.setdefault(info['category'], []).append(uid)

    per_cat = max(10, n_series // max(len(by_cat), 1))
    selected = []
    for _, uids in by_cat.items():
        n = min(per_cat, len(uids))
        selected.extend(rng.choice(uids, size=n, replace=False).tolist())

    if len(selected) > n_series:
        selected = rng.choice(selected, size=n_series, replace=False).tolist()
    return selected
