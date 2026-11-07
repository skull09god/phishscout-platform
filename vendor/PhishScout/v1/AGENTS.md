# v1 - PhishScout v1 training and benchmark code

## Purpose
Original PhishScout v1 training/benchmark scripts. Kept as the minimal-footprint
reference and for backwards-compatible comparison.

## Ownership
- Owner: PhishScout project. Read-only reference; do not edit.

## Local Contracts
- `train_phishscout.py` - trains the 35-feature LightGBM, exports ONNX + metadata.
- `sweep_models.py` - 34-config Pareto sweep (quality vs ONNX size).
- `benchmark_models.py` - 12-model baseline benchmark.
- `deep_benchmark.py` - seed variance, McNemar, bootstrap CI, robustness.
- `train_tiny_model.py` - earlier tiny random-forest experiment (historical).
- `upload_model.py` - pushes v1 artifacts to HuggingFace.
- `benchmark_results.md`, `benchmark.png` - v1 benchmark outputs.

## Work Guidance
- v1 is read-only reference. Reproduce with `python train_phishscout.py` from this directory.
- Do not modify v1 code; v2 lives in `../v2/`.

## Verification
- `python train_phishscout.py` trains and exports ONNX to `/tmp/phishtrap_model/`.

## Child DOX Index
- No child AGENTS.md files.
