#!/usr/bin/env python3
"""Train a tiny Random Forest phishing detector on PhishTrap, export to ONNX.

Design (Approach 2): Tiny Random Forest
- 10 trees, max_depth=6 -> ~30 KB ONNX model
- Runs fully on-device (browser extension, ONNX Runtime Web)
- No GPU needed: 20K x 16 features trains in seconds on CPU
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnx
from onnxruntime import InferenceSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import os
HF_TOKEN = os.environ.get("HF_TOKEN") or None
REPO_ID = "saidutta69/PhishTrap"
MODEL_REPO = "saidutta69/PhishTrap-Model"
FEATURE_COLUMNS = [
    "url_length", "hyphen_count", "digit_count", "subdomain_count",
    "trusted_tld", "protocol_exists", "special_char_count", "entropy",
    "path_depth", "domain_length", "is_domain_ip", "has_at_symbol",
    "has_double_slash_redirect", "tld_length", "query_param_count", "path_length",
]

OUTPUT_DIR = Path("/tmp/phishtrap_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():
    from huggingface_hub import hf_hub_download
    splits = {}
    for name in ("train", "val", "test"):
        path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"data/{name}.csv", token=HF_TOKEN,
        )
        splits[name] = pd.read_csv(path)
        print(f"  {name}: {len(splits[name]):,} rows")
    return splits


def main():
    print("=" * 60)
    print("PhishTrap Tiny Model Trainer (RandomForest, CPU-only)")
    print("=" * 60)

    print("\n[1/5] Downloading PhishTrap from Hugging Face...")
    splits = download_dataset()
    train, val, test = splits["train"], splits["val"], splits["test"]

    X_train = train[FEATURE_COLUMNS].astype(np.float32)
    y_train = train["label"].values
    X_val = val[FEATURE_COLUMNS].astype(np.float32)
    y_val = val["label"].values
    X_test = test[FEATURE_COLUMNS].astype(np.float32)
    y_test = test["label"].values

    print("\n[2/5] Training Tiny RandomForest (10 trees, depth 6)...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=6,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.2f}s")

    print("\n[3/5] Evaluating...")
    metrics = {}
    for name, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
        preds = clf.predict(X)
        proba = clf.predict_proba(X)[:, 1]
        m = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "f1": f1_score(y, preds),
            "auc": roc_auc_score(y, proba),
            "n_samples": int(len(y)),
        }
        metrics[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
        print(f"  {name:5s}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

    cm = confusion_matrix(y_test, clf.predict(X_test))
    print(f"  Confusion matrix (test):\n{cm}")

    print("\n[4/5] Exporting to ONNX...")
    initial_types = [("input", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    onx = convert_sklearn(
        clf,
        initial_types=initial_types,
        target_opset=18,
        options={id(clf): {"zipmap": False, "nocl": True}},
    )
    onnx_path = OUTPUT_DIR / "model.onnx"
    onnx.save(onx, onnx_path)
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  Saved {onnx_path} ({size_kb:.1f} KB)")

    print("\n[5/5] Verifying ONNX parity with sklearn...")
    sess = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    n = min(2000, len(X_test))
    sample = X_test.iloc[:n].values
    ort_preds = sess.run(None, {"input": sample})[0]
    sk_preds = clf.predict(sample)
    parity = float(np.mean(ort_preds == sk_preds))
    print(f"  ONNX == sklearn on {n} samples: {parity:.4f}")

    model_info = {
        "model": "tiny-random-forest-v1",
        "algorithm": "RandomForestClassifier",
        "params": {
            "n_estimators": 10, "max_depth": 6, "min_samples_leaf": 5,
            "max_features": "sqrt", "random_state": 42,
        },
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "onnx_size_kb": round(size_kb, 1),
        "onnx_parity": round(parity, 4),
        "opset": 18,
        "dataset": REPO_ID,
        "trained_on": "train split",
    }
    with open(OUTPUT_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"\n  model_info.json written. ONNX size: {size_kb:.1f} KB")

    print("\n" + "=" * 60)
    print("DONE - model ready at /tmp/phishtrap_model/model.onnx")
    print("=" * 60)


if __name__ == "__main__":
    main()
