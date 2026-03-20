import os
import pickle
import logging

import numpy as np
import pandas as pd

import config
from src.data import load_m4_monthly, filter_series, split_train_test, prepare_series_dict
from src.seasonality import analyze_from_df, sample_series_ids
from src.baselines import run_baselines
from src.models import run_catboost_experiment
from src.metrics import compute_metrics_per_series, compute_metrics_per_horizon


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

PRE_SAMPLE_SIZE = 1500


def _save_metrics(all_preds, sdict, ids, season_info):
    rows = []
    for method, preds in all_preds.items():
        sm, ma = [], []
        for uid in ids:
            m = compute_metrics_per_series(
                sdict[uid]['test_values'], preds[uid], sdict[uid]['train_values'])
            sm.append(m['sMAPE']); ma.append(m['MASE'])
        rows.append({
            'method': method,
            'sMAPE_mean': np.nanmean(sm), 'sMAPE_median': np.nanmedian(sm),
            'MASE_mean': np.nanmean(ma), 'MASE_median': np.nanmedian(ma),
        })
    df = pd.DataFrame(rows).sort_values('sMAPE_mean')
    df.to_csv(os.path.join(config.RESULTS_DIR, 'metrics_overall.csv'), index=False)
    log.info("Overall metrics:\n%s", df.to_string(index=False))

    rows = []
    for method, preds in all_preds.items():
        for cat in config.SEASONALITY_BINS:
            cat_ids = [u for u in ids if season_info[u]['category'] == cat]
            if not cat_ids:
                continue
            sm = [compute_metrics_per_series(
                sdict[u]['test_values'], preds[u], sdict[u]['train_values'])['sMAPE']
                for u in cat_ids]
            ma = [compute_metrics_per_series(
                sdict[u]['test_values'], preds[u], sdict[u]['train_values'])['MASE']
                for u in cat_ids]
            rows.append({
                'method': method, 'seasonality': cat, 'n_series': len(cat_ids),
                'sMAPE_mean': np.nanmean(sm), 'MASE_mean': np.nanmean(ma),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(config.RESULTS_DIR, 'metrics_by_seasonality.csv'), index=False)

    rows = []
    for method, preds in all_preds.items():
        for h in range(config.FORECAST_HORIZON):
            sm, ma = [], []
            for uid in ids:
                hm = compute_metrics_per_horizon(
                    sdict[uid]['test_values'], preds[uid], sdict[uid]['train_values'])
                sm.append(hm[h]['sMAPE']); ma.append(hm[h]['MASE'])
            rows.append({
                'method': method, 'horizon_step': h + 1,
                'sMAPE_mean': np.nanmean(sm), 'MASE_mean': np.nanmean(ma),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(config.RESULTS_DIR, 'metrics_by_horizon.csv'), index=False)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    log.info("Loading M4 Monthly dataset...")
    df = load_m4_monthly()
    log.info("Loaded %d series, %d rows", df['unique_id'].nunique(), len(df))

    df = filter_series(df)
    log.info("After length filter: %d series", df['unique_id'].nunique())

    log.info("Splitting train / test (vectorized)...")
    train_df, test_df = split_train_test(df)
    log.info("Train rows: %d, test rows: %d", len(train_df), len(test_df))

    all_ids = train_df['unique_id'].unique()
    rng = np.random.RandomState(config.RANDOM_SEED)
    pre_ids = rng.choice(all_ids, size=min(PRE_SAMPLE_SIZE, len(all_ids)), replace=False)
    log.info("Pre-sampled %d series for seasonality analysis", len(pre_ids))

    train_pre = train_df[train_df['unique_id'].isin(pre_ids)]
    season_info = analyze_from_df(train_pre)

    selected_ids = sample_series_ids(season_info)
    log.info("Selected %d series", len(selected_ids))
    cats = pd.Series([season_info[u]['category'] for u in selected_ids])
    log.info("Distribution:\n%s", cats.value_counts().to_string())

    pd.DataFrame([
        {'unique_id': u, 'strength': season_info[u]['strength'],
         'category': season_info[u]['category']}
        for u in selected_ids
    ]).to_csv(os.path.join(config.RESULTS_DIR, 'seasonality_info.csv'), index=False)

    train_sel = train_df[train_df['unique_id'].isin(selected_ids)]
    test_sel = test_df[test_df['unique_id'].isin(selected_ids)]
    selected_dict = prepare_series_dict(train_sel, test_sel)
    log.info("Series dict ready (%d series)", len(selected_dict))

    log.info("Running baselines...")
    bl_fc = run_baselines(train_sel).reset_index()
    baseline_names = ['Naive', 'SeasonalNaive', 'AutoTheta', 'AutoETS']
    all_preds = {}
    for bname in baseline_names:
        preds = {}
        for uid in selected_ids:
            preds[uid] = bl_fc.loc[bl_fc['unique_id'] == uid, bname].values
        all_preds[bname] = preds

    for feat_name, feat_cfg in config.FEATURE_SETS.items():
        log.info("CatBoost [%s] ...", feat_name)
        preds, _ = run_catboost_experiment(selected_dict, feat_cfg, selected_ids)
        all_preds[f'CB_{feat_name}'] = preds

    log.info("Computing metrics...")
    _save_metrics(all_preds, selected_dict, selected_ids, season_info)

    with open(os.path.join(config.RESULTS_DIR, 'predictions.pkl'), 'wb') as f:
        pickle.dump(all_preds, f)
    with open(os.path.join(config.RESULTS_DIR, 'test_actuals.pkl'), 'wb') as f:
        pickle.dump({u: selected_dict[u]['test_values'] for u in selected_ids}, f)
    with open(os.path.join(config.RESULTS_DIR, 'train_data.pkl'), 'wb') as f:
        pickle.dump({u: selected_dict[u]['train_values'] for u in selected_ids}, f)

    log.info("Done! Results saved to %s", config.RESULTS_DIR)


if __name__ == '__main__':
    main()
