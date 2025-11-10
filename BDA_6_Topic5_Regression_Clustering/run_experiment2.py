"""
Experiment 2: KMeans vs DeepCluster-like (autoencoder + kmeans) on MNIST and Fashion-MNIST.
This implementation uses a lightweight autoencoder (MLP) as feature extractor to simulate DeepCluster.
"""
import argparse
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import normalized_mutual_info_score
from utils import save_json, Timer
import matplotlib.pyplot as plt


def load_mnist(name='mnist_784', n_samples=2000):
    print('Loading', name)
    data = fetch_openml(name, version=1, as_frame=False)
    X = data.data.astype(np.float32) / 255.0
    y = data.target.astype(int)
    if n_samples is not None and n_samples < X.shape[0]:
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        return X[idx], y[idx]
    return X, y


def kmeans_baseline(X, y, outdir, n_clusters=10):
    os.makedirs(outdir, exist_ok=True)
    with Timer() as t:
        k = KMeans(n_clusters=n_clusters, random_state=0)
        preds = k.fit_predict(X)
    score = normalized_mutual_info_score(y, preds)
    save_json(os.path.join(outdir, 'kmeans_results.json'), {'nmi': float(score), 'time': t.dt})
    return score


def deepcluster_like(X, y, outdir, n_clusters=10, n_iter=2, code_size=64):
    os.makedirs(outdir, exist_ok=True)
    # train autoencoder (MLP) to reconstruct and use bottleneck as features
    # encoder is first half of layers
    X_in = X
    for it in range(n_iter):
        print('DeepCluster-like iteration', it+1)
        # autoencoder
        ae = MLPRegressor(hidden_layer_sizes=(256, code_size), max_iter=50, random_state=it)
        ae.fit(X_in, X_in)
        # extract code (output of hidden layer) - MLPRegressor does not expose hidden layer outputs easily
        # so we approximate by using the transform from a separate MLP with same weights is complex; instead,
        # we use the predictions of the bottleneck by taking the activations of the second layer via a small hack:
        try:
            # sklearn MLP stores coefs_ and intercepts_
            W1, W2 = ae.coefs_[0], ae.coefs_[1]
            b1, b2 = ae.intercepts_[0], ae.intercepts_[1]
            # compute hidden activation after first layer, then after second
            h1 = np.tanh(np.dot(X_in, W1) + b1)
            code = np.tanh(np.dot(h1, W2) + b2)
        except Exception:
            # fallback: use AE predictions as features
            code = ae.predict(X_in)
        # cluster codes
        k = KMeans(n_clusters=n_clusters, random_state=it)
        preds = k.fit_predict(code)
        score = normalized_mutual_info_score(y, preds)
        print('NMI at iter', it+1, score)
        # for next iteration, we can pseudo-label and optionally re-train; here we just continue
    save_json(os.path.join(outdir, 'deepcluster_results.json'), {'nmi': float(score)})
    return score


def main(outdir, dataset):
    os.makedirs(outdir, exist_ok=True)
    if dataset == 'mnist':
        X, y = load_mnist('mnist_784', n_samples=3000)
    else:
        X, y = load_mnist('Fashion-MNIST', n_samples=3000)

    res_k = kmeans_baseline(X, y, os.path.join(outdir, 'kmeans'))
    res_d = deepcluster_like(X, y, os.path.join(outdir, 'deepcluster'))
    allres = {'kmeans_nmi': res_k, 'deepcluster_nmi': res_d}
    save_json(os.path.join(outdir, 'results_exp2.json'), allres)
    print('Wrote exp2 results to', outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', default='outputs/exp2')
    parser.add_argument('--dataset', '-d', default='mnist')
    args = parser.parse_args()
    main(args.output_dir, args.dataset)
