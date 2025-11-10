"""
Experiment 1: Compare Linear Regression and Neural Network Regression
on synthetic dataset and California Housing dataset.
"""
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from utils import save_json, Timer
import os


def synthetic_dataset(n=1000, noise=0.2, random_state=0):
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, size=(n, 1))
    y = np.sin(X).ravel() + noise * rng.randn(n)
    return X, y


def run_on_dataset(X, y, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    results = {'dataset': name}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear regression
    with Timer() as t:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['linear'] = {
        'mse': float(mean_squared_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'train_time': t.dt,
    }

    # Neural network regression (MLP)
    with Timer() as t:
        mlp = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500, random_state=0)
        mlp.fit(X_train, y_train)
    y_pred2 = mlp.predict(X_test)
    results['mlp'] = {
        'mse': float(mean_squared_error(y_test, y_pred2)),
        'r2': float(r2_score(y_test, y_pred2)),
        'train_time': t.dt,
    }

    # plot
    xs = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
    try:
        y_lr = lr.predict(xs)
        y_mlp = mlp.predict(xs)
        plt.figure()
        plt.scatter(X_test, y_test, alpha=0.3, label='test')
        plt.plot(xs, y_lr, label='Linear', color='C1')
        plt.plot(xs, y_mlp, label='MLP', color='C2')
        plt.legend()
        plt.title(f'Prediction curves - {name}')
        plot_path = os.path.join(outdir, 'prediction_plot.png')
        plt.savefig(plot_path)
        results['plot'] = plot_path
        plt.close()
    except Exception:
        pass

    save_json(os.path.join(outdir, 'results_exp1.json'), results)
    return results


def main(outdir):
    # synthetic
    Xs, ys = synthetic_dataset(n=1000, noise=0.2)
    res_syn = run_on_dataset(Xs, ys, os.path.join(outdir, 'synthetic'), 'synthetic_sin')

    # California housing
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    # standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    res_cal = run_on_dataset(Xs, y, os.path.join(outdir, 'california'), 'california_housing')

    allres = {'synthetic': res_syn, 'california': res_cal}
    save_json(os.path.join(outdir, 'results_exp1_overview.json'), allres)
    print('Wrote results to', outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', default='outputs/exp1')
    args = parser.parse_args()
    main(args.output_dir)
