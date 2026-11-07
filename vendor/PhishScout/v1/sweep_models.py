#!/usr/bin/env python3
"""Pareto-frontier sweep: quality vs ONNX size.

Expanded feature set (~35 features) + model family x capacity grid.
For each config: train on train split, eval on test, export ONNX, record size.
Goal: find the elbow of the F1-vs-size curve, not just the smallest model.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

REPO_ID = "saidutta69/PhishTrap"
OUT = Path("/tmp/phishtrap_sweep")
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
    if not s:
        return 0.0
    from collections import Counter
    probs = [c / len(s) for c in Counter(s).values()]
    return -sum(p * np.log2(p) for p in probs)


def expanded_features(url_raw):
    url = str(url_raw).lower().strip()
    has_proto = url.startswith("http")
    full = url if has_proto else "http://" + url
    from urllib.parse import urlparse
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

    # --- expanded features ---
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


EXPANDED_COLUMNS = list(expanded_features("http://example.com/a?b=c").keys())
BASE_COLUMNS = [c for c in EXPANDED_COLUMNS[:17]]


def load_splits():
    splits = {}
    for name in ("train", "val", "test"):
        p = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"data/{name}.csv")
        df = pd.read_csv(p)
        feats = pd.DataFrame([expanded_features(u) for u in df["url"]])
        df = pd.concat([df[["url", "label"]], feats], axis=1)
        splits[name] = df
    return splits


def onnx_export(model, name):
    import onnxmltools
    from skl2onnx.common.data_types import FloatTensorType as SkFT
    from onnxmltools.convert.common.data_types import FloatTensorType as OmFT
    if name.startswith("lgbm"):
        onx = onnxmltools.convert_lightgbm(model.booster_,
                                           initial_types=[("input", OmFT([None, model.n_features_in_]))],
                                           target_opset=15)
    elif name.startswith("xgb"):
        onx = onnxmltools.convert_xgboost(model,
                                          initial_types=[("input", OmFT([None, model.n_features_in_]))],
                                          target_opset=15)
    else:
        onx = convert_sklearn(model, initial_types=[("input", SkFT([None, model.n_features_in_]))],
                              target_opset=18)
    used = sorted({n.domain or "" for n in onx.graph.node})
    del onx.opset_import[:]
    for d in used:
        onx.opset_import.append(onnx.helper.make_opsetid(d, {"": 18, "ai.onnx": 18, "ai.onnx.ml": 3}.get(d, 1)))
    path = OUT / f"{name}.onnx"
    onnx.save(onx, path)
    return path


def main():
    print("=" * 72)
    print("Pareto sweep: quality vs ONNX size (expanded features)")
    print("=" * 72)
    splits = load_splits()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"  train {len(tr)} | val {len(va)} | test {len(te)} | features {len(EXPANDED_COLUMNS)}")

    configs = {}
    for base in ("base16", "full"):
        cols = BASE_COLUMNS if base == "base16" else EXPANDED_COLUMNS
        cfg = {
            f"dt_d{base}": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
            f"rf_10x6_{base}": RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1),
            f"rf_20x8_{base}": RandomForestClassifier(n_estimators=20, max_depth=8, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1),
            f"rf_40x10_{base}": RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1),
            f"rf_100x12_{base}": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1),
            f"et_30x10_{base}": ExtraTreesClassifier(n_estimators=30, max_depth=10, min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1),
            f"gb_50x4_{base}": GradientBoostingClassifier(n_estimators=50, max_depth=4, min_samples_leaf=5, random_state=42),
            f"gb_100x5_{base}": GradientBoostingClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42),
            f"ada_50x3_{base}": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=5), n_estimators=50, random_state=42),
            f"ada_150x3_{base}": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=5), n_estimators=150, random_state=42),
        }
        configs.update(cfg)

    import lightgbm as lgb
    import xgboost as xgb
    for base in ("base16", "full"):
        for est, depth, leaves in ((30, 6, 15), (60, 8, 31), (100, 10, 63), (200, 12, 127)):
            configs[f"lgbm_{est}x{depth}L{leaves}_{base}"] = lgb.LGBMClassifier(
                n_estimators=est, max_depth=depth, num_leaves=leaves, min_child_samples=5,
                random_state=42, verbose=-1, n_jobs=-1)
        for est, depth in ((30, 4), (60, 6), (100, 8)):
            configs[f"xgb_{est}x{depth}_{base}"] = xgb.XGBClassifier(
                n_estimators=est, max_depth=depth, min_child_weight=1, learning_rate=0.1,
                random_state=42, n_jobs=-1, eval_metric="logloss")

    results = []
    for name, model in configs.items():
        is_base16 = name.endswith("base16")
        cols = BASE_COLUMNS if is_base16 else EXPANDED_COLUMNS
        Xtr = tr[cols].astype(np.float32).values
        Xte = te[cols].astype(np.float32).values
        ytr, yte = tr["label"].values, te["label"].values
        t0 = time.time()
        try:
            model.fit(Xtr, ytr)
            p = model.predict(Xte)
            proba = model.predict_proba(Xte)[:, 1]
            path = onnx_export(model, name.replace("/", "_"))
            size = round(path.stat().st_size / 1024, 1)
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            parity = float(np.mean(sess.run(None, {"input": Xte[:1000]})[0] == p[:1000]))
            r = {
                "name": name, "feature_set": "base16" if is_base16 else "full",
                "f1": round(f1_score(yte, p), 4), "acc": round(accuracy_score(yte, p), 4),
                "prec": round(precision_score(yte, p), 4), "rec": round(recall_score(yte, p), 4),
                "auc": round(roc_auc_score(yte, proba), 4), "size_kb": size,
                "onnx_parity": round(parity, 4), "train_s": round(time.time() - t0, 1),
            }
            results.append(r)
            print(f"  {r['name']:30s} feat={r['feature_set']:6s} f1={r['f1']} acc={r['acc']} "
                  f"auc={r['auc']} size={size:>7}KB parity={r['onnx_parity']}")
        except Exception as e:
            print(f"  {name:30s} FAILED {type(e).__name__}: {str(e)[:60]}")

    with open(OUT / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsweep -> {OUT}/sweep_results.json")


if __name__ == "__main__":
    main()
