#!/usr/bin/env python3
"""PhishScout v2 - ONNX inference loader.

Loads models/model.onnx together with its metadata (features.json,
model_info.json), applies the Platt calibration stored at training time, and
exposes a URL -> probability/label API. The core model is fully on-device and
offline; this loader has no network dependencies.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import extract_features, FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class PhishScoutV2:
    def __init__(self, models_dir: str | Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.features = json.loads((self.models_dir / "features.json").read_text())
        self.info = json.loads((self.models_dir / "model_info.json").read_text())
        self.session = ort.InferenceSession(
            str(self.models_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        cal = self.info.get("calibration", {})
        self._a = float(cal.get("a", 1.0))
        self._b = float(cal.get("b", 0.0))
        self.threshold = float(self.info.get("threshold_recommended", 0.5))

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self._a * raw + self._b)))

    def predict_features(self, features: np.ndarray, threshold: float | None = None) -> dict:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = np.asarray(features, dtype=np.float32)
        outputs = self.session.run(None, {"input": features})
        p = outputs[1]  # LightGBM probability (ZipMap or 2-col array)
        if isinstance(p, list):
            proba_lgbm = np.array([d.get(1, 0.0) for d in p], dtype=np.float64)
        else:
            proba_lgbm = np.asarray(p, dtype=np.float64)[:, 1]
        proba = self._calibrate(proba_lgbm)
        t = self.threshold if threshold is None else threshold
        label = (proba > t).astype(int)
        return {
            "probability": float(proba[0]),
            "label": int(label[0]),
            "is_phishing": bool(label[0] == 1),
            "probability_lightgbm": float(proba_lgbm[0]),
            "threshold": float(t),
        }

    def predict_url(self, url: str, threshold: float | None = None) -> dict:
        feats = extract_features(url)
        ordered = np.array([[feats[c] for c in self.features]], dtype=np.float32)
        return self.predict_features(ordered, threshold)


if __name__ == "__main__":
    m = PhishScoutV2()
    for u in [
        "http://paypa1-verify.ngrok.io/login?token=x",
        "https://www.google.com",
        "http://3.136.61.40/login",
        "http://facebоок-account.com",  # homoglyph
    ]:
        print(f"{u:45s} -> {m.predict_url(u)}")
