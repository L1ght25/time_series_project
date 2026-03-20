import numpy as np
import pandas as pd

import config


def _all_lags(feature_config):
    lags = set()
    if feature_config.get('use_lags'):
        lags.update(range(1, config.MAX_REGULAR_LAGS + 1))
    if feature_config.get('use_seasonal_lags'):
        lags.update(config.SEASONAL_LAGS)
    return sorted(lags)


def generate_step_features(values, dates, target_idx, feature_config):
    features = {}
    for lag in _all_lags(feature_config):
        idx = target_idx - lag
        features[f'lag_{lag}'] = values[idx] if idx >= 0 else np.nan

    if feature_config.get('use_calendar'):
        dt = pd.Timestamp(dates[target_idx])
        features['month'] = dt.month
        features['quarter'] = dt.quarter

    if feature_config.get('use_fourier'):
        dt = pd.Timestamp(dates[target_idx])
        m = dt.month
        for k in range(1, config.N_FOURIER_TERMS + 1):
            features[f'sin_{k}'] = np.sin(2 * np.pi * k * m / config.SEASON_PERIOD)
            features[f'cos_{k}'] = np.cos(2 * np.pi * k * m / config.SEASON_PERIOD)
    return features


def create_tabular_dataset(series_dict, feature_config, selected_ids=None):
    if selected_ids is None:
        selected_ids = list(series_dict.keys())

    rows, targets = [], []
    for uid in selected_ids:
        vals = series_dict[uid]['normalized_train']
        dates = series_dict[uid]['train_dates']
        for t in range(config.MIN_HISTORY, len(vals)):
            rows.append(generate_step_features(vals, dates, t, feature_config))
            targets.append(vals[t])

    return pd.DataFrame(rows), np.array(targets)


def get_feature_names(feature_config):
    names = [f'lag_{l}' for l in _all_lags(feature_config)]
    if feature_config.get('use_calendar'):
        names += ['month', 'quarter']
    if feature_config.get('use_fourier'):
        for k in range(1, config.N_FOURIER_TERMS + 1):
            names += [f'sin_{k}', f'cos_{k}']
    return names
