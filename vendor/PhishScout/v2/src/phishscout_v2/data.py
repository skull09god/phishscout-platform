#!/usr/bin/env python3
"""Load the PhishTrap dataset (train/validation/test splits) and build v2 features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import load_dataset

from .features import extract_features, FEATURE_COLUMNS

REPO_ID = "saidutta69/PhishTrap"


def _features_df(urls: list[str]) -> np.ndarray:
    rows = [extract_features(u) for u in urls]
    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLUMNS].astype(np.float32).values


def load_splits(cache_dir: str | None = None):
    ds = load_dataset(REPO_ID, cache_dir=cache_dir)
    out = {}
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        urls = df["url"].tolist()
        X = _features_df(urls)
        y = df["label"].astype(np.int64).values
        out[split] = (X, y, urls)
        print(f"  {split:10s}: {len(X):6d} rows | pos={int(y.sum())} neg={len(y)-int(y.sum())} | features={X.shape[1]}")
    return out


if __name__ == "__main__":
    splits = load_splits()
    print("feature count:", len(FEATURE_COLUMNS))
