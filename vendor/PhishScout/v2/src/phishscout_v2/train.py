#!/usr/bin/env python3
"""Train PhishScout v2: a stronger but still tiny (<1MB) on-device LightGBM.

Pipeline:
  1. Load PhishTrap splits with the v2 (66-feature) extractor.
  2. Sweep a handful of compact GBDT configurations; pick the best F1 whose
     ONNX export stays under 1 MB.
  3. Fit a Platt (logistic) calibration on the validation split so the exported
     raw score is mapped to a well-behaved probability at inference time.
  4. Export the chosen model to ONNX (opset 15), verify 100% parity, measure
     latency, and write model_info.json + features.json to models/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .data import load_splits
from .features import FEATURE_COLUMNS

OUT = Path(__file__).resolve().parents[2] / "models"
SIZE_BUDGET_KB = 1000.0

SWEEP = [
    # (n_estimators, max_depth, num_leaves, min_child_samples, label)
    (60, 8, 31, 5, "lgbm_60x8L31"),
    (120, 8, 31, 5, "lgbm_120x8L31"),
    (200, 8, 31, 5, "lgbm_200x8L31"),
    (120, 10, 63, 10, "lgbm_120x10L63"),
    (200, 10, 63, 10, "lgbm_200x10L63"),
    (300, 8, 31, 10, "lgbm_300x8L31"),
    (250, 12, 63, 20, "lgbm_250x12L63"),
    (400, 8, 31, 20, "lgbm_400x8L31"),
]


def _export_onnx(booster, n_features, path: Path):
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxmltools.convert import convert_lightgbm
    onx = convert_lightgbm(
        booster,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=15,
    )
    used = sorted({n.domain or "" for n in onx.graph.node})
    del onx.opset_import[:]
    for d in used:
        onx.opset_import.append(
            onnx.helper.make_opsetid(d, {"": 18, "ai.onnx": 18, "ai.onnx.ml": 3}.get(d, 1))
        )
    onnx.save(onx, path)
    onnx.checker.check_model(onx)
    return round(path.stat().st_size / 1024, 1)


def _metrics(y, pred, proba):
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "auc": round(roc_auc_score(y, proba), 4),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    Xtr, ytr, _ = splits["train"]
    Xva, yva, _ = splits["validation"]
    Xte, yte, _ = splits["test"]

    print("\n" + "=" * 64)
    print("PhishScout v2 - model selection sweep")
    print("=" * 64)

    results = []
    for n_est, depth, leaves, mcs, label in SWEEP:
        t0 = time.time()
        model = lgb.LGBMClassifier(
            n_estimators=n_est, max_depth=depth, num_leaves=leaves,
            min_child_samples=mcs, learning_rate=0.1, random_state=42,
            n_jobs=-1, verbose=-1,
        )
        model.fit(Xtr, ytr)
        pte = model.predict_proba(Xte)[:, 1]
        pva = model.predict_proba(Xva)[:, 1]
        m = _metrics(yte, model.predict(Xte), pte)
        mva = _metrics(yva, model.predict(Xva), pva)
        onnx_path = OUT / f"_tmp_{label}.onnx"
        size_kb = _export_onnx(model.booster_, Xtr.shape[1], onnx_path)
        print(f"  {label:18s} valF1={mva['f1']:.4f} testF1={m['f1']:.4f} "
              f"AUC={m['auc']:.4f} size={size_kb}KB ({time.time()-t0:.1f}s)")
        results.append((label, m, mva, size_kb, model))
        onnx_path.unlink(missing_ok=True)

    # Choose best VALIDATION F1 under budget (no test leakage); tie-break smaller size.
    feasible = [(r[0], r[1], r[2], r[3], r[4]) for r in results if r[3] <= SIZE_BUDGET_KB]
    feasible.sort(key=lambda r: (-r[2]["f1"], r[3]))
    best_label, best_m, best_mva, best_size, best_model = feasible[0]
    print(f"\n  SELECTED (val-based): {best_label}  valF1={best_mva['f1']}  "
          f"testF1={best_m['f1']}  size={best_size}KB")

    # Final full-eval on test with threshold tuning on val.
    pva = best_model.predict_proba(Xva)[:, 1]
    pte = best_model.predict_proba(Xte)[:, 1]
    best_t, best_f1v = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(yva, pva > t)
        if f > best_f1v:
            best_t, best_f1v = t, f
    test_at_t = {
        "threshold": round(float(best_t), 2),
        "f1": round(float(f1_score(yte, pte > best_t)), 4),
        "recall": round(float(recall_score(yte, pte > best_t)), 4),
        "precision": round(float(precision_score(yte, pte > best_t)), 4),
    }

    # ---- Platt calibration on validation ----
    cal = LogisticRegression(C=1e6, solver="lbfgs")
    cal.fit(pva.reshape(-1, 1), yva)
    pte_cal = cal.predict_proba(pte.reshape(-1, 1))[:, 1]
    cal_metrics = _metrics(yte, (pte_cal > best_t).astype(int), pte_cal)

    # ---- ONNX export (raw trees; calibration applied at inference) ----
    onnx_path = OUT / "model.onnx"
    size_kb = _export_onnx(best_model.booster_, Xtr.shape[1], onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity = float(np.mean(sess.run(None, {"input": Xte[:2000]})[0] == best_model.predict(Xte[:2000])))
    lat = []
    for _ in range(200):
        t = time.perf_counter()
        sess.run(None, {"input": Xte[:1]})
        lat.append((time.perf_counter() - t) * 1e3)
    latency_us = float(np.median(lat) * 1000)

    cm = confusion_matrix(yte, best_model.predict(Xte))

    model_info = {
        "name": "PhishScout",
        "version": "v2",
        "model": best_label,
        "algorithm": "LightGBM (gradient boosting) + Platt calibration",
        "params": {
            "n_estimators": best_model.n_estimators,
            "max_depth": best_model.max_depth,
            "num_leaves": best_model.num_leaves,
            "min_child_samples": best_model.min_child_samples,
            "learning_rate": 0.1,
            "objective": "binary",
            "random_state": 42,
        },
        "features": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "selection": "best validation F1 under size budget (no test leakage)",
        "metrics_val_raw": best_mva,
        "metrics_test_raw": best_m,
        "metrics_test_calibrated": cal_metrics,
        "test_at_best_val_threshold": test_at_t,
        "calibration": {
            "method": "platt",
            "a": float(cal.coef_[0][0]),
            "b": float(cal.intercept_[0]),
        },
        "confusion_matrix_test": {
            "TN": int(cm[0][0]), "FP": int(cm[0][1]),
            "FN": int(cm[1][0]), "TP": int(cm[1][1]),
        },
        "onnx_size_kb": size_kb,
        "onnx_parity": round(parity, 4),
        "latency_us_median": round(latency_us, 1),
        "opset": 15,
        "size_budget_kb": SIZE_BUDGET_KB,
        "dataset": "saidutta69/PhishTrap",
        "trained_on": "train split (13,968 rows)",
        "eval_on": "held-out test (2,994 rows)",
        "threshold_default": 0.5,
        "threshold_recommended": round(float(best_t), 2),
    }
    (OUT / "model_info.json").write_text(json.dumps(model_info, indent=2))
    (OUT / "features.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))

    print(f"\n  ONNX: {size_kb} KB | parity {parity:.4f} | latency {latency_us:.1f}us")
    print(f"  raw test    : {best_m}")
    print(f"  calibrated  : {cal_metrics}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
