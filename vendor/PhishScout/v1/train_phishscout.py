#!/usr/bin/env python3
"""Train the final PhishScout model: LightGBM 60x8L31 on the full 35-feature set.

Writes to /tmp/phishtrap_model/:
  model.onnx, model_info.json, features.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import lightgbm as lgb
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

REPO_ID = "saidutta69/PhishTrap"
OUT = Path("/tmp/phishtrap_model")
OUT.mkdir(parents=True, exist_ok=True)

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "secure", "security", "account",
    "update", "confirm", "banking", "password", "webscr", "delivery", "invoice",
    "wallet", "authenticate", "unlock", "recover", "suspended", "limited", "alert",
    "dispute", "refund", "billing", "bonus", "prize", "lottery", "free",
]
BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google",
    "gmail", "amazon", "netflix", "facebook", "instagram", "whatsapp", "chase",
    "wellsfargo", "citibank", "bankofamerica", "hsbc", "barclays", "amex",
    "americanexpress", "visa", "mastercard", "coinbase", "binance", "steam",
    "spotify", "dropbox", "dhl", "fedex", "usps", "ups", "payoneer",
]
SENSITIVE_EXTENSIONS = (".exe", ".scr", ".apk", ".zip", ".jar", ".bat", ".cmd", ".js", ".docm")


def compute_entropy(s):
    from collections import Counter
    if not s:
        return 0.0
    probs = [c / len(s) for c in Counter(s).values()]
    return -sum(p * np.log2(p) for p in probs)


def extract_features(url_raw):
    import re
    from urllib.parse import urlparse
    url = str(url_raw).lower().strip()
    has_proto = url.startswith("http")
    full = url if has_proto else "http://" + url
    parsed = urlparse(full)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    netloc = parsed.netloc

    f = {}
    f["url_length"] = len(full)
    f["hyphen_count"] = url.count("-")
    f["digit_count"] = sum(1 for c in url if c.isdigit())
    f["subdomain_count"] = max(0, len(host.split(".")) - 2)
    f["protocol_exists"] = int(bool(re.match(r"^https?://", str(url_raw).lower().strip())))
    f["special_char_count"] = sum(1 for c in url if c in "@-_.,;:#~!$&'()*+/:=?")
    f["entropy"] = compute_entropy(full)
    f["path_depth"] = len([p for p in path.split("/") if p])
    f["domain_length"] = len(host)
    f["is_domain_ip"] = int(bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host)))
    f["has_at_symbol"] = int("@" in url)
    f["has_double_slash_redirect"] = int(url.count("//") > 1)
    try:
        from tld import get_tld
        from phishtrap_pipeline.config import TRUSTED_TLDS
        tld_info = get_tld(full, fail_silently=True)
        f["tld_length"] = len(tld_info) if tld_info else 0
        f["trusted_tld"] = int(bool(tld_info) and f".{tld_info}" in TRUSTED_TLDS)
    except Exception:
        f["tld_length"] = 0
        f["trusted_tld"] = 0
    f["query_param_count"] = len(query.split("&")) if query else 0
    f["path_length"] = len(path)

    f["keyword_count"] = sum(url.count(k) for k in SUSPICIOUS_KEYWORDS)
    host_brands = sum(1 for b in BRANDS if b in host)
    path_brands = sum(1 for b in BRANDS if b in path)
    f["brand_in_host"] = int(host_brands > 0)
    f["brand_in_path"] = int(path_brands > 0)
    f["brand_count"] = host_brands + path_brands
    f["has_punycode"] = int("xn--" in host)
    f["has_nonstandard_port"] = int(re.search(r":(?!80|443)(\d+)", netloc) is not None)
    f["percent_encoding_count"] = url.count("%")
    f["has_hex_ip"] = int(bool(re.search(r"0x[0-9a-f]{1,8}", url)))
    f["uppercase_ratio"] = round(sum(1 for c in url_raw if c.isupper()) / max(1, len(url_raw)), 4)
    f["www_count"] = url.count("www.")
    runs = re.findall(r"\d+", url)
    f["max_digit_run"] = max((len(r) for r in runs), default=0)
    labels = host.split(".")
    f["max_label_len"] = max((len(l) for l in labels), default=0)
    f["host_label_count"] = len(labels)
    f["has_domain_in_path"] = int(bool(re.search(r"/[a-z0-9-]+\.[a-z]{2,}/", path)))
    f["query_length"] = len(query)
    f["equals_count"] = query.count("=")
    f["sensitive_extension"] = int(url.endswith(SENSITIVE_EXTENSIONS))
    f["has_multiple_tlds"] = int(len(re.findall(r"\.[a-z]{2,}/", url)) > 1)
    f["consecutive_punct"] = int(bool(re.search(r"[@_.:;,]{3,}", url)))
    return f


FEATURE_COLUMNS = list(extract_features("http://example.com/a?b=c").keys())


def load(name):
    p = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"data/{name}.csv")
    df = pd.read_csv(p)
    feats = pd.DataFrame([extract_features(u) for u in df["url"]])
    return feats[FEATURE_COLUMNS].astype(np.float32).values, df["label"].values, df


def main():
    print("=" * 60)
    print("PhishScout - final model build")
    print("=" * 60)
    Xtr, ytr, _ = load("train")
    Xva, yva, _ = load("val")
    Xte, yte, dfte = load("test")
    print(f"  train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} | features {len(FEATURE_COLUMNS)}")

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=60, max_depth=8, num_leaves=31, min_child_samples=5,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    print(f"  trained in {time.time()-t0:.1f}s")

    metrics = {}
    for name, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        p = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        metrics[name] = {
            "accuracy": round(accuracy_score(y, p), 4),
            "precision": round(precision_score(y, p), 4),
            "recall": round(recall_score(y, p), 4),
            "f1": round(f1_score(y, p), 4),
            "auc": round(roc_auc_score(y, proba), 4),
        }
        print(f"  {name:5s}: {metrics[name]}")

    cm = confusion_matrix(yte, model.predict(Xte))
    print(f"  confusion (test):\n{cm}")

    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    onx = onnxmltools.convert_lightgbm(
        model.booster_,
        initial_types=[("input", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
        target_opset=15,
    )
    used = sorted({n.domain or "" for n in onx.graph.node})
    del onx.opset_import[:]
    for d in used:
        onx.opset_import.append(onnx.helper.make_opsetid(d, {"": 18, "ai.onnx": 18, "ai.onnx.ml": 3}.get(d, 1)))
    onnx_path = OUT / "model.onnx"
    onnx.save(onx, onnx_path)
    onnx.checker.check_model(onx)
    size_kb = round(onnx_path.stat().st_size / 1024, 1)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = Xte[:2000]
    parity = float(np.mean(sess.run(None, {"input": sample})[0] == model.predict(sample)))
    lat = []
    for _ in range(200):
        t = time.perf_counter()
        sess.run(None, {"input": Xte[:1]})
        lat.append((time.perf_counter() - t) * 1e3)
    latency_us = float(np.median(lat) * 1000)

    best_t = 0.5
    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    best_f1 = -1
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(yva, pva > t)
        if f > best_f1:
            best_t, best_f1 = t, f
    metrics["test_at_best_val_threshold"] = {
        "threshold": round(float(best_t), 2),
        "f1": round(float(f1_score(yte, pte > best_t)), 4),
    }

    model_info = {
        "name": "PhishScout",
        "version": "v1",
        "model": "lightgbm-60x8L31-full35",
        "algorithm": "LightGBM (gradient boosting)",
        "params": {
            "n_estimators": 60, "max_depth": 8, "num_leaves": 31,
            "min_child_samples": 5, "learning_rate": 0.1,
            "objective": "binary", "random_state": 42,
        },
        "features": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "metrics": metrics,
        "confusion_matrix_test": {
            "TN": int(cm[0][0]), "FP": int(cm[0][1]),
            "FN": int(cm[1][0]), "TP": int(cm[1][1]),
        },
        "onnx_size_kb": size_kb,
        "onnx_parity": round(parity, 4),
        "latency_us_median": round(latency_us, 1),
        "opset": 15,
        "dataset": REPO_ID,
        "trained_on": "train split (13,969 rows)",
        "eval_on": "held-out test (2,993 rows)",
        "threshold_default": 0.5,
        "threshold_recommended": round(float(best_t), 2),
    }
    with open(OUT / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    with open(OUT / "features.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"\n  ONNX: {size_kb} KB | parity {parity:.4f} | latency {latency_us:.1f}us")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
