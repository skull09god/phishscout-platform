# v2 - PhishScout v2 source package

## Purpose
Python package implementing PhishScout v2: URL feature extraction, dataset
loading, model training/export, ONNX inference, and optional page/TLS enrichment.

## Ownership
- Owner: PhishScout v2 project.

## Local Contracts
- `src/phishscout_v2/features.py` - canonical v2 feature extractor. `extract_features(url)` returns an ordered `dict`; `FEATURE_COLUMNS` is the stable, ordered list that must match the ONNX model's input exactly. Adding/removing/reordering a feature is a breaking change: retrain, re-export, and update `features.json`.
- `src/phishscout_v2/data.py` - loads `saidutta69/PhishTrap` (train/validation/test) and builds feature matrices. Never mutate the upstream dataset; splits are fixed.
- `src/phishscout_v2/train.py` - sweeps compact LightGBM configs, selects best VALIDATION F1 under the 1 MB budget (no test leakage), fits Platt calibration on validation, exports ONNX, and writes `models/`. Raw tree scores are exported; calibration (a, b) is applied at inference, not baked into ONNX.
- `src/phishscout_v2/onnx_model.py` - loads `models/model.onnx`, applies Platt calibration, returns probability + label. Single entry point for on-device inference.
- `src/phishscout_v2/enrichment.py` - OPTIONAL, network-dependent second stage. `PageTlsEnricher` fetches the page/TLS and extracts signals; `EnrichedScorer` combines core + page signals via a scaled logistic meta-model (default, trained on a live-fetched cross-validated sample) or a transparent heuristic fallback. Must degrade gracefully: fetch failure -> fall back to core.
- `src/phishscout_v2/train_enrichment.py` - (optional, network) fetches a live sample, trains the scaled enrichment meta-model, writes `models/enrichment_model.json`.
- `src/phishscout_v2/benchmark.py` - 10-seed determinism + perturbation robustness; writes `models/benchmark_results.json`.
- `src/phishscout_v2/upload_model.py` - pushes `models/` + card to HuggingFace.

## Work Guidance
- Keep the on-device core strictly URL-only and dependency-light.
- Feature engineering must stay interpretable and computed from the URL string only (no network, no page fetch) for core features.
- Benchmark every candidate against v1 (F1 0.8626, AUC 0.9369, 131.5 KB) and report size, parity, latency.
- `v1/` is read-only reference; do not edit.

## Verification
- `cd v2 && python -m src.phishscout_v2.train` - trains, exports ONNX, asserts parity (must be 1.0000) and writes `models/`.
- After export, confirm `models/model.onnx` size < 1000 KB and ONNX parity == 1.0.

## Child DOX Index
- `models/AGENTS.md` - trained artifacts contract.
