#!/usr/bin/env python3
"""Deep head-to-head benchmark: decision tree d8 vs LightGBM 40x8L31.

1. Seed variance (10 seeds): F1 / acc / AUC mean + std on held-out test
2. McNemar test on paired predictions (seed 42)
3. Paired bootstrap 95% CI on the F1 difference
4. PR curve + optimal threshold (F1-optimal on val, applied to test)
5. Misclassification profiles (FP/FN feature distributions)
6. Robustness: Gaussian feature noise, random corruption
7. Hard subset: re-labeled (compromised) legit domains
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

REPO_ID = "saidutta69/PhishTrap"
FEATURE_COLUMNS = [
    "url_length", "hyphen_count", "digit_count", "subdomain_count",
    "trusted_tld", "protocol_exists", "special_char_count", "entropy",
    "path_depth", "domain_length", "is_domain_ip", "has_at_symbol",
    "has_double_slash_redirect", "tld_length", "query_param_count", "path_length",
]
OUT = Path("/tmp/phishtrap_bench2")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def make_models(seed):
    tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed)
    lgbm = lgb.LGBMClassifier(n_estimators=40, max_depth=8, num_leaves=31,
                              min_child_samples=5, random_state=seed,
                              verbose=-1, n_jobs=-1)
    return tree, lgbm


def load():
    splits = {}
    for name in ("train", "val", "test"):
        p = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"data/{name}.csv")
        df = pd.read_csv(p)
        splits[name] = (
            df[FEATURE_COLUMNS].astype(np.float32).values,
            df["label"].values,
            df,
        )
    return splits


def main():
    splits = load()
    Xtr, ytr, dftr = splits["train"]
    Xva, yva, dfva = splits["val"]
    Xte, yte, dfte = splits["test"]
    print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}")

    # --- 1. seed variance ---
    print("\n[1] Seed variance (10 seeds, held-out test)")
    seed_rows = {"tree": [], "lgbm": []}
    for seed in SEEDS:
        tree, lgbm = make_models(seed)
        tree.fit(Xtr, ytr)
        lgbm.fit(Xtr, ytr)
        for name, m in (("tree", tree), ("lgbm", lgbm)):
            p = m.predict(Xte)
            proba = m.predict_proba(Xte)[:, 1]
            seed_rows[name].append({
                "seed": seed, "f1": f1_score(yte, p), "acc": accuracy_score(yte, p),
                "prec": precision_score(yte, p), "rec": recall_score(yte, p),
                "auc": roc_auc_score(yte, proba),
            })
    for name, rows in seed_rows.items():
        for metric in ("f1", "acc", "prec", "rec", "auc"):
            vals = [r[metric] for r in rows]
            print(f"  {name:5s} {metric:5s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    with open(OUT / "seed_variance.json", "w") as f:
        json.dump(seed_rows, f, indent=2)

    # --- 2 + 3. paired comparison at seed 42 ---
    print("\n[2] Paired comparison (seed 42)")
    tree, lgbm = make_models(42)
    tree.fit(Xtr, ytr)
    lgbm.fit(Xtr, ytr)
    pt = tree.predict(Xte)
    pl = lgbm.predict(Xte)

    # McNemar on disagreements
    d = {}
    for i in range(len(yte)):
        key = (pt[i] == yte[i], pl[i] == yte[i])
        d[key] = d.get(key, 0) + 1
    b = d.get((False, True), 0)  # lgbm better
    c = d.get((True, False), 0)  # tree better
    chi = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0
    from scipy.stats import chi2 as chi2_dist
    pval = 1 - chi2_dist.cdf(chi, 1)
    print(f"  McNemar: lgbm-only-right={b}, tree-only-right={c}, chi2={chi:.2f}, p={pval:.4f}")
    print(f"  {'significant' if pval < 0.05 else 'NOT significant'} at alpha=0.05")

    # paired bootstrap on F1 diff
    rng = np.random.default_rng(42)
    n_boot = 2000
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yte), len(yte))
        f1_t = f1_score(yte[idx], pt[idx])
        f1_l = f1_score(yte[idx], pl[idx])
        diffs.append(f1_l - f1_t)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  bootstrap 95% CI of F1(lgbm - tree): [{lo:.4f}, {hi:.4f}] "
          f"({'lgbm better' if lo > 0 else 'tree better' if hi < 0 else 'inconclusive'})")

    # --- 4. PR curve + optimal threshold ---
    print("\n[3] Threshold analysis (F1-optimal on val)")
    for name, m in (("tree", tree), ("lgbm", lgbm)):
        pva = m.predict_proba(Xva)[:, 1]
        pte = m.predict_proba(Xte)[:, 1]
        best_t, best_f1 = 0.5, -1
        for t in np.arange(0.05, 0.96, 0.01):
            f = f1_score(yva, pva > t)
            if f > best_f1:
                best_t, best_f1 = t, f
        f1_at = f1_score(yte, pte > best_t)
        rec_at = recall_score(yte, pte > best_t)
        prec_at = precision_score(yte, pte > best_t)
        print(f"  {name:5s}: default-0.5 F1={f1_score(yte, pl if name=='lgbm' else pt):.4f} | "
              f"best t={best_t:.2f} (val F1 {best_f1:.4f}) -> test F1={f1_at:.4f} "
              f"prec={prec_at:.4f} rec={rec_at:.4f}")

    # --- 5. misclassification profiles ---
    print("\n[4] Misclassification analysis (seed 42, tree d8)")
    for name, p in (("tree", pt), ("lgbm", pl)):
        tp = (p == 1) & (yte == 1)
        fp = (p == 1) & (yte == 0)
        fn = (p == 0) & (yte == 1)
        tn = (p == 0) & (yte == 0)
        print(f"  {name:5s}: TP={tp.sum()} FP={fp.sum()} FN={fn.sum()} TN={tn.sum()}")
        print(f"         FP URL-length mean={dfte['url_length'][fp].mean():.0f} "
              f"vs TN={dfte['url_length'][tn].mean():.0f} | "
              f"entropy FP={dfte['entropy'][fp].mean():.2f} vs TN={dfte['entropy'][tn].mean():.2f}")
        print(f"         FN trusted_tld frac={dfte['trusted_tld'][fn].mean():.2f} "
              f"vs TP={dfte['trusted_tld'][tp].mean():.2f} | "
              f"FN is_ip frac={dfte['is_domain_ip'][fn].mean():.2f} vs TP={dfte['is_domain_ip'][tp].mean():.2f}")
        print(f"         FN subdomain_count={dfte['subdomain_count'][fn].mean():.2f} "
              f"vs TP={dfte['subdomain_count'][tp].mean():.2f}")

    # --- 6. robustness ---
    print("\n[5] Robustness (seed 42, test set)")
    rng = np.random.default_rng(0)
    for noise_sigma in (0.05, 0.15, 0.5):
        for name, m in (("tree", tree), ("lgbm", lgbm)):
            Xn = Xte + rng.normal(0, noise_sigma, Xte.shape)
            p = m.predict(Xn)
            drop = f1_score(yte, pt if name == "tree" else pl) - f1_score(yte, p)
            print(f"  gaussian sigma={noise_sigma}: {name:5s} F1 drop {drop:.4f}")

    # corruption: set random column to extreme garbage
    for name, m in (("tree", tree), ("lgbm", lgbm)):
        Xc = Xte.copy()
        mask = rng.random(Xc.shape) < 0.05
        Xc[mask] = 1e9
        p = m.predict(Xc)
        print(f"  5% extreme-value corruption: {name:5s} F1 {f1_score(yte, p):.4f} "
              f"(clean {f1_score(yte, pt if name=='tree' else pl):.4f})")

    # --- 7. hard subset: re-labeled (compromised legit) ---
    print("\n[6] Hard subset: re-labeled domains (compromised legit)")
    legit_urls = set(dftr["url"][dftr["label"] == 0].str.lower())
    relabeled = dfte[dfte["url"].str.lower().isin(legit_urls)]
    if len(relabeled):
        Xh = relabeled[FEATURE_COLUMNS].astype(np.float32).values
        yh = relabeled["label"].values
        for name, m in (("tree", tree), ("lgbm", lgbm)):
            p = m.predict(Xh)
            print(f"  {name:5s}: n={len(yh)} acc={accuracy_score(yh, p):.4f} "
                  f"recall(phishing)={recall_score(yh, p):.4f}")
    else:
        print("  none found in test split")

    print(f"\nreport artifacts -> {OUT}")


if __name__ == "__main__":
    main()
