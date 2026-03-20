import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from tqdm import tqdm

import config
from src.features import generate_step_features, create_tabular_dataset


def train_catboost(X_train, y_train, params=None):
    if params is None:
        params = config.CATBOOST_PARAMS
    cat_features = [c for c in X_train.columns if c in ('month', 'quarter')]
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, cat_features=cat_features or None)
    return model


def recursive_predict(model, series_info, feature_config, horizon=None):
    if horizon is None:
        horizon = config.FORECAST_HORIZON

    values = list(series_info['normalized_train'])
    train_dates = list(pd.to_datetime(series_info['train_dates']))
    future_dates = [train_dates[-1] + pd.DateOffset(months=i + 1) for i in range(horizon)]
    all_dates = np.array(train_dates + future_dates, dtype='datetime64[ns]')

    preds_norm = []
    for h in range(horizon):
        t_idx = len(series_info['normalized_train']) + h
        feat = generate_step_features(np.array(values), all_dates, t_idx, feature_config)
        pred = model.predict(pd.DataFrame([feat]))[0]
        values.append(pred)
        preds_norm.append(pred)

    return np.array(preds_norm) * series_info['std'] + series_info['mean']


def run_catboost_experiment(series_dict, feature_config, selected_ids, horizon=None):
    if horizon is None:
        horizon = config.FORECAST_HORIZON

    X_train, y_train = create_tabular_dataset(series_dict, feature_config, selected_ids)
    model = train_catboost(X_train, y_train)

    predictions = {}
    for uid in tqdm(selected_ids, desc='Predicting', leave=False):
        predictions[uid] = recursive_predict(model, series_dict[uid], feature_config, horizon)

    return predictions, model
