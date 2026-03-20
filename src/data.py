import os
import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4

import config


def load_m4_monthly():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    result = M4.load(directory=config.DATA_DIR, group='Monthly')
    Y_df = result[0]
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    Y_df['y'] = Y_df['y'].astype(float)
    return Y_df


def filter_series(df, min_length=None):
    if min_length is None:
        min_length = config.MIN_SERIES_LENGTH
    lengths = df.groupby('unique_id').size()
    valid_ids = lengths[lengths >= min_length].index
    return df[df['unique_id'].isin(valid_ids)].copy()


def split_train_test(df, horizon=None):
    if horizon is None:
        horizon = config.FORECAST_HORIZON
    df = df.sort_values(['unique_id', 'ds'])
    reverse_rank = df.groupby('unique_id').cumcount(ascending=False)
    return df[reverse_rank >= horizon].copy(), df[reverse_rank < horizon].copy()


def prepare_series_dict(train_df, test_df):
    train_df = train_df.sort_values(['unique_id', 'ds'])
    test_df = test_df.sort_values(['unique_id', 'ds'])

    train_grouped = dict(list(train_df.groupby('unique_id')))
    test_grouped = dict(list(test_df.groupby('unique_id')))

    series_dict = {}
    for uid, train in train_grouped.items():
        test = test_grouped[uid]
        train_values = train['y'].values
        mean, std = train_values.mean(), train_values.std()
        if std < 1e-8:
            std = 1.0
        series_dict[uid] = {
            'train_values': train_values,
            'train_dates': train['ds'].values,
            'test_values': test['y'].values,
            'test_dates': test['ds'].values,
            'mean': mean,
            'std': std,
            'normalized_train': (train_values - mean) / std,
        }
    return series_dict
