#!/usr/bin/env python3
"""Benchmark multiple tiny phishing-detection models on PhishTrap.

Each candidate: 5-fold CV on train, held-out test eval, ONNX export,
model size, ONNX parity, and single-URL inference latency via onnxruntime.
All training is CPU-only (14K x 16 rows - seconds per model).

Candidates: naive bayes, logistic regression, decision tree,
random forest, extra trees, adaboost, gradient boosting,
histogram boosting, lightgbm, xgboost, svm, knn.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

warnings.filterwarnings("ignore")

REPO_ID = "saidutta69/PhishTrap"
FEATURE_COLUMNS = [
    "url_length", "hyphen_count", "digit_count", "subdomain_count",
    "trusted_tld", "protocol_exists", "special_char_count", "entropy",
    "path_depth", "domain_length", "is_domain_ip", "has_at_symbol",
    "has_double_slash_redirect", "tld_length", "query_param_count", "path_length",
]
OUTPUT = Path("/tmp/phishtrap_bench")
OUTPUT.mkdir(parents=True, exist_ok=True)
N_LATENCY = 200


def make_candidates():
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
        GradientBoostingClassifier, HistGradientBoostingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    rf = RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=5,
                                max_features="sqrt", random_state=42, n_jobs=-1)
    import lightgbm as lgb
    import xgboost as xgb
    candidates = {
        "gaussian_nb": ("naive bayes", GaussianNB()),
        "logistic_regression": ("logistic regression", Pipeline([("sc", StandardScaler()),
                                 ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])),
        "decision_tree": ("decision tree", DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)),
        "random_forest": ("random forest 10x6", rf),
        "extra_trees": ("extra trees 10x6", ExtraTreesClassifier(n_estimators=10, max_depth=6, min_samples_leaf=5,
                                                                 max_features="sqrt", random_state=42, n_jobs=-1)),
        "adaboost": ("adaboost 30x3", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=5),
                                                         n_estimators=30, random_state=42)),
        "gradient_boosting": ("gboost 30x3", GradientBoostingClassifier(n_estimators=30, max_depth=3,
                                                                        min_samples_leaf=5, random_state=42)),
        "hist_gb": ("hist gboost 30x15", HistGradientBoostingClassifier(max_leaf_nodes=15,
                                                                        min_samples_leaf=5, max_iter=30, random_state=42)),
        "lightgbm": ("lightgbm 30x6", lgb.LGBMClassifier(
            n_estimators=30, max_depth=6, num_leaves=15, min_child_samples=5,
            random_state=42, verbose=-1, n_jobs=-1)),
        "xgboost": ("xgboost 30x4", xgb.XGBClassifier(
            n_estimators=30, max_depth=4, min_child_weight=1, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss")),
        "svm": ("svm rbf", Pipeline([("sc", StandardScaler()),
                                     ("svm", SVC(C=1.0, gamma="scale", probability=True, random_state=42))])),
        "knn": ("knn k=5", Pipeline([("sc", StandardScaler()),
                                     ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])),
    }
    return candidates


def download_dataset():
    from huggingface_hub import hf_hub_download
    splits = {}
    for name in ("train", "val", "test"):
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"data/{name}.csv")
        splits[name] = pd.read_csv(path)
    return splits


def repair_opset(onx):
    used = sorted({n.domain or "" for n in onx.graph.node})
    versions = {"": 18, "ai.onnx": 18, "ai.onnx.ml": 3, "com.microsoft": 1}
    del onx.opset_import[:]
    for d in used:
        onx.opset_import.append(onnx.helper.make_opsetid(d, versions.get(d, 1)))


def convert_with_fallback(model):
    last = None
    opts_sets = [
        {id(model): {"zipmap": False, "nocl": True}},
        {id(model): {"zipmap": False}},
        None,
    ]
    for opts in opts_sets:
        try:
            return convert_sklearn(model, initial_types=initial_types,
                                   target_opset=18, options=opts)
        except Exception as e:
            last = e
    raise last


def convert_boosting(model, name):
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    src = model.booster_ if hasattr(model, "booster_") else model
    fn = onnxmltools.convert_xgboost if name == "xgboost" else onnxmltools.convert_lightgbm
    return fn(src, initial_types=[("input", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
              target_opset=15)


def onnx_export(model, name):
    if name in ("lightgbm", "xgboost"):
        onx = convert_boosting(model, name)
    else:
        onx = convert_with_fallback(model)
    repair_opset(onx)
    path = OUTPUT / f"{name}.onnx"
    onnx.save(onx, path)
    onnx.checker.check_model(onx)
    return path


def main():
    global initial_types
    initial_types = [("input", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    print("=" * 72)
    print("PhishTrap model benchmark - 12 candidates, CPU only")
    print("=" * 72)

    splits = download_dataset()
    train, val, test = splits["train"], splits["val"], splits["test"]
    X_train = train[FEATURE_COLUMNS].astype(np.float32).values
    y_train = train["label"].values
    X_val = val[FEATURE_COLUMNS].astype(np.float32).values
    y_val = val["label"].values
    X_test = test[FEATURE_COLUMNS].astype(np.float32).values
    y_test = test["label"].values
    print(f"  train {len(X_train)} | val {len(X_val)} | test {len(X_test)}")
    print(f"  label 1 (phishing) share: train={y_train.mean():.3f} test={y_test.mean():.3f}\n")

    candidates = make_candidates()
    results = []
    for name, (label, model) in candidates.items():
        t0 = time.time()
        row = {"name": name, "family": label}
        try:
            model.fit(X_train, y_train)
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
            row["cv_f1_mean"] = round(float(cv.mean()), 4)
            row["cv_f1_std"] = round(float(cv.std()), 4)

            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            row.update({
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "precision": round(precision_score(y_test, preds), 4),
                "recall": round(recall_score(y_test, preds), 4),
                "f1": round(f1_score(y_test, preds), 4),
                "auc": round(roc_auc_score(y_test, proba), 4),
            })

            # best-F1 threshold tuned on val, applied to test
            val_proba = model.predict_proba(X_val)[:, 1]
            best_t, best_f1 = 0.5, -1
            for t in np.arange(0.05, 0.96, 0.05):
                f = f1_score(y_val, val_proba > t)
                if f > best_f1:
                    best_t, best_f1 = t, f
            row["best_thr"] = round(float(best_t), 2)
            row["f1_at_best_thr"] = round(f1_score(y_test, proba > best_t), 4)

            try:
                path = onnx_export(model, name)
                row["size_kb"] = round(path.stat().st_size / 1024, 1)

                sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                sample = X_test[:1000]
                parity = np.mean(sess.run(None, {"input": sample})[0] == model.predict(sample))
                row["onnx_parity"] = round(float(parity), 4)

                lat = []
                for _ in range(N_LATENCY):
                    t = time.perf_counter()
                    sess.run(None, {"input": X_test[:1]})
                    lat.append((time.perf_counter() - t) * 1e3)
                row["latency_ms"] = round(float(np.median(lat)), 3)
            except Exception as e:
                row["export_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            row["train_s"] = round(time.time() - t0, 2)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:80]}"

        results.append(row)
        done = "OK " if "error" not in row else "ERR"
        print(f"  [{done}] {name:22s} f1={row.get('f1', '-'):<7} acc={row.get('accuracy', '-'):<7} "
              f"size={row.get('size_kb', '-'):<7}KB lat={row.get('latency_ms', '-')}ms "
              f"cv={row.get('cv_f1_mean', '-')} train={row.get('train_s', '-')}s"
              + (f"  ERR={row['error']}" if "error" in row else ""))

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {OUTPUT}/results.json")


if __name__ == "__main__":
    main()
