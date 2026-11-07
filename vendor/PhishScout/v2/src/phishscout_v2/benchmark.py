#!/usr/bin/env python3
"""PhishScout v2 - robustness and determinism benchmark.

Reproduces the v1-style robustness checks plus a 10-seed determinism check
using the selected v2 configuration. Writes benchmark_results.json to models/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score

from .data import load_splits

OUT = Path(__file__).resolve().parents[2] / "models"
N_SEEDS = 10


def _config():
    info = json.loads((OUT / "model_info.json").read_text())
    p = info["params"]
    return dict(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
        learning_rate=p.get("learning_rate", 0.1),
    )


def _fit(cfg, X, y, seed):
    m = lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **cfg)
    m.fit(X, y)
    return m


def determinism(Xtr, ytr, Xte, yte):
    cfg = _config()
    f1s, aucs = [], []
    for s in range(N_SEEDS):
        m = _fit(cfg, Xtr, ytr, s)
        p = m.predict_proba(Xte)[:, 1]
        f1s.append(f1_score(yte, p > 0.5))
        aucs.append(roc_auc_score(yte, p))
    return {
        "n_seeds": N_SEEDS,
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
    }


def _corrupt_extreme(X, frac=0.05, mag=1e6, rng=None):
    rng = rng or np.random.default_rng(0)
    Xc = X.copy()
    n, d = Xc.shape
    mask = rng.random((n, d)) < frac
    Xc[mask] = mag
    return Xc


def _corrupt_gaussian(X, sigma=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    Xc = X.copy().astype(np.float64)
    Xc += rng.normal(0, sigma, Xc.shape)
    np.maximum(Xc, 0, out=Xc)
    return Xc.astype(np.float32)


def robustness(cfg, Xtr, ytr, Xte, yte):
    m = _fit(cfg, Xtr, ytr, 42)
    base = f1_score(yte, m.predict_proba(Xte)[:, 1] > 0.5)
    rng = np.random.default_rng(0)
    f_ext = f1_score(yte, m.predict(_corrupt_extreme(Xte, rng=rng)) > 0.5)
    rng = np.random.default_rng(1)
    f_gau = f1_score(yte, m.predict(_corrupt_gaussian(Xte, rng=rng)) > 0.5)
    return {
        "clean_f1": round(float(base), 4),
        "extreme_5pct_f1": round(float(f_ext), 4),
        "gaussian_sigma05_f1": round(float(f_gau), 4),
    }


def main():
    splits = load_splits()
    Xtr, ytr, _ = splits["train"]
    Xte, yte, _ = splits["test"]
    cfg = _config()
    print("benchmark config:", cfg)
    det = determinism(Xtr, ytr, Xte, yte)
    rob = robustness(cfg, Xtr, ytr, Xte, yte)
    print("determinism:", det)
    print("robustness :", rob)
    (OUT / "benchmark_results.json").write_text(json.dumps(
        {"config": cfg, "determinism": det, "robustness": rob}, indent=2))
    print(f"-> {OUT / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
