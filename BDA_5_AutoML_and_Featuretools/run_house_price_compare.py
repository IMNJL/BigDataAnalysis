"""
Experiment: House price prediction comparison
- Compares models trained on:
  A) original features
  B) original + manual engineered features
  C) original + Featuretools deep features
  D) original + AutoML-selected features (FLAML feature selection/AutoML)

Outputs:
- outputs/house_compare_results.json
- outputs/house_compare_details.csv (per-fold metrics)

Usage (from project root .venv):
python3 "BDA_5_AutoML_and_Featuretools/Topic5 [light version]/run_house_price_compare.py" --data "california_housing.csv" --out outputs

Notes:
- This script is lightweight and aims to be reproducible on CPU. For full AutoML runs, increase time budgets.
"""
import os
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    import featuretools as ft
except Exception:
    ft = None

try:
    from flaml import AutoML
except Exception:
    AutoML = None


def load_data(path):
    df = pd.read_csv(path)
    return df


def manual_features(df):
    df2 = df.copy()
    # Example manual features for California housing
    # rooms_per_household, bedrooms_per_room, population_per_household
    if 'total_rooms' in df2.columns and 'households' in df2.columns:
        df2['rooms_per_household'] = df2['total_rooms'] / (df2['households'] + 1e-6)
    if 'total_bedrooms' in df2.columns and 'total_rooms' in df2.columns:
        df2['bedrooms_per_room'] = df2['total_bedrooms'] / (df2['total_rooms'] + 1e-6)
    if 'population' in df2.columns and 'households' in df2.columns:
        df2['population_per_household'] = df2['population'] / (df2['households'] + 1e-6)
    return df2


def featuretools_features(df, target_col):
    if ft is None:
        raise RuntimeError('featuretools is not installed; install with pip install featuretools')
    df_copy = df.reset_index().rename(columns={'index': 'idx'})
    es = ft.EntitySet('housing')
    es = es.add_dataframe(dataframe_name='main', dataframe=df_copy, index='idx')
    # run deep feature synthesis with primitive defaults
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='main', max_depth=2)
    # feature_matrix contains the generated features; drop target if present
    if target_col in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=[target_col])
    # align index
    feature_matrix = feature_matrix.reset_index(drop=True)
    return feature_matrix


def evaluate_model(X, y, model, cv=3):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    metrics = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        metrics.append({'mse': float(mse), 'rmse': float(rmse), 'r2': float(r2)})
    return metrics


def summarize(metrics):
    arr = pd.DataFrame(metrics)
    return {'mse_mean': arr['mse'].mean(), 'rmse_mean': arr['rmse'].mean(), 'r2_mean': arr['r2'].mean()}


def run_all(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(data_path)
    # Expect target 'median_house_value' or 'target'
    if 'median_house_value' in df.columns:
        target_col = 'median_house_value'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        # assume last column
        target_col = df.columns[-1]

    y = df[target_col]
    X_orig = df.drop(columns=[target_col])

    results = {}

    # A) Original features
    model_rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    print('Evaluating original features...')
    metrics_a = evaluate_model(X_orig, y, model_rf, cv=3)
    results['original'] = {'per_fold': metrics_a, 'summary': summarize(metrics_a)}

    # B) Original + manual engineered features
    print('Generating manual features...')
    df_manual = manual_features(df)
    X_manual = df_manual.drop(columns=[target_col])
    metrics_b = evaluate_model(X_manual, y, model_rf, cv=3)
    results['manual'] = {'per_fold': metrics_b, 'summary': summarize(metrics_b)}

    # C) Original + Featuretools features (may be slow)
    if ft is not None:
        try:
            print('Generating featuretools features (this may take some time)...')
            ft_feats = featuretools_features(df.drop(columns=[target_col]), target_col)
            # featuretools returns new features only; align and concat
            X_ft = pd.concat([X_orig.reset_index(drop=True), ft_feats.reset_index(drop=True)], axis=1).fillna(0)
            metrics_c = evaluate_model(X_ft, y.reset_index(drop=True), model_rf, cv=3)
            results['featuretools'] = {'per_fold': metrics_c, 'summary': summarize(metrics_c)}
        except Exception as e:
            results['featuretools'] = {'error': str(e)}
    else:
        results['featuretools'] = {'error': 'featuretools not installed'}

    # D) AutoML features / AutoML model (FLAML) - run a light AutoML to find a good model
    if AutoML is not None:
        try:
            print('Running FLAML AutoML (light budget)...')
            automl = AutoML()
            X_train, X_test, y_train, y_test = train_test_split(X_orig, y, test_size=0.2, random_state=42)
            automl_settings = {
                "time_budget": 30,  # seconds, small budget
                "metric": 'rmse',
                "task": 'regression',
                "log_file_name": os.path.join(out_dir, 'flaml.log')
            }
            automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
            pred = automl.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, pred)
            results['autofit'] = {'mse': float(mse), 'rmse': float(rmse), 'r2': float(r2), 'best_config': automl.best_config}
        except Exception as e:
            results['autofit'] = {'error': str(e)}
    else:
        results['autofit'] = {'error': 'FLAML not installed'}

    # save
    ts = int(time.time())
    out_json = os.path.join(out_dir, f'house_compare_results_{ts}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Wrote results to', out_json)
    # also write CSV per fold
    records = []
    for strategy, val in results.items():
        if isinstance(val, dict) and 'per_fold' in val:
            for i, fold in enumerate(val['per_fold']):
                records.append({'strategy': strategy, 'fold': i, **fold})
    if records:
        df_rec = pd.DataFrame(records)
        csv_out = os.path.join(out_dir, f'house_compare_details_{ts}.csv')
        df_rec.to_csv(csv_out, index=False)
        print('Wrote per-fold metrics to', csv_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='california_housing.csv')
    parser.add_argument('--out', default='BDA_5_AutoML_and_Featuretools/Topic5 [light version]/outputs')
    args = parser.parse_args()
    run_all(args.data, args.out)
