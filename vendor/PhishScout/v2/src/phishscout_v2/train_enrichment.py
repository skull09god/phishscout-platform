#!/usr/bin/env python3
"""Train the PhishScout v2 enrichment meta-model.

Fetches a sample of URLs from the PhishTrap validation/test splits, extracts
page/TLS signals (via PageTlsEnricher), and trains a small logistic regression
that maps [core_probability, page_signals] -> phishing. Only URLs that were
actually fetched (no fetch error) are used. Saves enrichment_model.json; the
EnrichedScorer then uses it automatically.

This step is OPTIONAL and network-dependent; it does not change the on-device
core model or its ONNX file.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .data import load_splits
from .enrichment import EnrichedScorer

OUT = Path(__file__).resolve().parents[2] / "models"
N_META_TRAIN = 1200
N_META_EVAL = 1200
WORKERS = 48
TIMEOUT = 8.0


def _build_rows(scorer, urls, labels):
    rows = []
    feats = json.loads((OUT / "features.json").read_text())
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut = {ex.submit(scorer.enricher.signals_for, u): (i, u) for i, u in enumerate(urls)}
        for f in as_completed(fut):
            i, u = fut[f]
            sig = f.result()
            if sig.get("fetch_error"):
                continue
            core_p = scorer.core.predict_url(u)["probability"]
            vec = scorer._signal_vector(sig)
            rows.append(([core_p] + vec, int(labels[i])))
    return rows


def main():
    splits = load_splits()
    scorer = EnrichedScorer()
    scorer.enricher.timeout = TIMEOUT

    _, yva, uva = splits["validation"]
    _, yte, ute = splits["test"]

    print(f"fetching {N_META_TRAIN} validation URLs for meta-training ...")
    train_rows = _build_rows(scorer, uva[:N_META_TRAIN], yva[:N_META_TRAIN])
    print(f"  usable (fetched) rows: {len(train_rows)}")
    if len(train_rows) < 50:
        print("  too few fetched rows; aborting (network likely restricted).")
        return

    Xtr = np.array([r[0] for r in train_rows], dtype=np.float64)
    ytr = np.array([r[1] for r in train_rows], dtype=np.int64)

    print(f"fetching {N_META_EVAL} test URLs for meta-eval ...")
    eval_rows = _build_rows(scorer, ute[:N_META_EVAL], yte[:N_META_EVAL])

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    model.fit(Xtr_s, ytr)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def combined_prob(vec):
        x = scaler.transform(np.asarray(vec, dtype=np.float64).reshape(1, -1))
        z = model.intercept_[0] + model.coef_[0] @ x[0]
        return sigmoid(z)

    # 5-fold cross-validated estimate on the fetched training sample (scaled).
    cv = cross_val_score(model, Xtr_s, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                        scoring="f1")
    print(f"  5-fold CV F1: {cv.mean():.4f} +/- {cv.std():.4f}")

    # Compare core-only vs enriched on the separately fetched eval set.
    core_ps, enr_ps, ys = [], [], []
    for vec, y in eval_rows:
        core_ps.append(vec[0])
        enr_ps.append(combined_prob(vec))
        ys.append(y)
    core_ps, enr_ps, ys = map(np.array, (core_ps, enr_ps, ys))

    t = scorer.core.threshold
    core_f1 = f1_score(ys, core_ps > t)
    core_rec = recall_score(ys, core_ps > t)
    core_prec = precision_score(ys, core_ps > t)
    enr_f1 = f1_score(ys, enr_ps > t)
    enr_rec = recall_score(ys, enr_ps > t)
    enr_prec = precision_score(ys, enr_ps > t)
    enr_auc = roc_auc_score(ys, enr_ps)

    print("\n  meta-model trained on fetched sample")
    print(f"  core-only : F1={core_f1:.4f} rec={core_rec:.4f} prec={core_prec:.4f}")
    print(f"  enriched  : F1={enr_f1:.4f} rec={enr_rec:.4f} prec={enr_prec:.4f} AUC={enr_auc:.4f}")

    meta = {
        "method": "logistic",
        "features": ["core_probability"] + scorer.SIGNAL_FEATURES,
        "w": [float(x) for x in model.coef_[0]],
        "b": float(model.intercept_[0]),
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "n_train_rows": len(train_rows),
        "cv_f1_mean": round(float(cv.mean()), 4),
        "cv_f1_std": round(float(cv.std()), 4),
        "eval_core": {"f1": round(float(core_f1), 4), "recall": round(float(core_rec), 4),
                      "precision": round(float(core_prec), 4)},
        "eval_enriched": {"f1": round(float(enr_f1), 4), "recall": round(float(enr_rec), 4),
                          "precision": round(float(enr_prec), 4), "auc": round(float(enr_auc), 4)},
    }
    (OUT / "enrichment_model.json").write_text(json.dumps(meta, indent=2))
    print(f"  -> {OUT / 'enrichment_model.json'}")


if __name__ == "__main__":
    main()
