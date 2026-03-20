import numpy as np

import config


def smape(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    denom = np.abs(a) + np.abs(p)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(200.0 * np.mean(np.abs(a[mask] - p[mask]) / denom[mask]))


def mase(actual, predicted, train_values, season_period=None):
    if season_period is None:
        season_period = config.SEASON_PERIOD
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    tv = np.asarray(train_values, float)
    scale = np.mean(np.abs(tv[season_period:] - tv[:-season_period]))
    if scale < 1e-10:
        return np.nan
    return float(np.mean(np.abs(a - p)) / scale)


def compute_metrics_per_series(actual, predicted, train_values):
    return {
        'sMAPE': smape(actual, predicted),
        'MASE': mase(actual, predicted, train_values),
    }


def compute_metrics_per_horizon(actual, predicted, train_values):
    tv = np.asarray(train_values, float)
    scale = np.mean(np.abs(tv[config.SEASON_PERIOD:] - tv[:-config.SEASON_PERIOD]))
    out = []
    for h in range(len(actual)):
        d = abs(actual[h]) + abs(predicted[h])
        s = 200.0 * abs(actual[h] - predicted[h]) / d if d > 0 else 0.0
        m = abs(actual[h] - predicted[h]) / scale if scale > 1e-10 else np.nan
        out.append({'step': h + 1, 'sMAPE': s, 'MASE': m})
    return out
