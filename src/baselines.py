import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoTheta, AutoETS

import config

def run_baselines(train_df, horizon=None, season_length=None):
    if horizon is None:
        horizon = config.FORECAST_HORIZON
    if season_length is None:
        season_length = config.SEASON_PERIOD

    models = [
        Naive(),
        SeasonalNaive(season_length=season_length),
        AutoTheta(season_length=season_length),
        AutoETS(season_length=season_length),
    ]

    sf = StatsForecast(models=models, freq='MS', n_jobs=-1)
    forecast_df = sf.forecast(df=train_df, h=horizon)
    return forecast_df