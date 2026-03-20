import os


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

RANDOM_SEED = 42
N_SERIES = 150
SEASON_PERIOD = 12
FORECAST_HORIZON = 18
MIN_SERIES_LENGTH = 72

MAX_REGULAR_LAGS = 12
SEASONAL_LAGS = [12, 24, 36]
MIN_HISTORY = 36
N_FOURIER_TERMS = 3

SEASONALITY_BINS = {
    'weak': (0.0, 0.4),
    'medium': (0.4, 0.7),
    'strong': (0.7, 1.0),
}

FEATURE_SETS = {
    'lags_only': {
        'use_lags': True, 'use_seasonal_lags': False,
        'use_calendar': False, 'use_fourier': False,
    },
    'lags_seasonal': {
        'use_lags': True, 'use_seasonal_lags': True,
        'use_calendar': False, 'use_fourier': False,
    },
    'lags_calendar': {
        'use_lags': True, 'use_seasonal_lags': False,
        'use_calendar': True, 'use_fourier': False,
    },
    'lags_fourier': {
        'use_lags': True, 'use_seasonal_lags': False,
        'use_calendar': False, 'use_fourier': True,
    },
    'lags_seasonal_calendar': {
        'use_lags': True, 'use_seasonal_lags': True,
        'use_calendar': True, 'use_fourier': False,
    },
    'lags_seasonal_fourier': {
        'use_lags': True, 'use_seasonal_lags': True,
        'use_calendar': False, 'use_fourier': True,
    },
}

CATBOOST_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'RMSE',
    'random_seed': RANDOM_SEED,
    'verbose': 0,
}
