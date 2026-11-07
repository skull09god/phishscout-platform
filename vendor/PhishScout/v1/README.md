# PhishScout — Training & Benchmark Code

Open-source training and benchmarking code for [PhishScout](https://huggingface.co/saidutta69/PhishScout),
the 131.5 KB on-device phishing URL detector.

## What's here

| File | Purpose |
|---|---|
| `train_phishscout.py` | Trains the final PhishScout model (LightGBM 60x8L31, 35 features), exports ONNX, writes `model_info.json` |
| `benchmark_models.py` | 12-model baseline benchmark (CV + test metrics + ONNX export + latency) |
| `sweep_models.py` | 34-config Pareto sweep (quality vs ONNX size, 16 vs 35 features) |
| `deep_benchmark.py` | Head-to-head validation: seed variance, McNemar, bootstrap CI, robustness |
| `train_tiny_model.py` | Earlier tiny random-forest experiment (kept for history) |
| `upload_model.py` | Pushes model artifacts to Hugging Face (reads `HF_TOKEN` from env) |
| `benchmark_results.md` | Full benchmark report |
| `benchmark.png` | Size-vs-quality chart |

## Requirements

Python 3.11+, CPU only (no GPU needed):

```bash
pip install scikit-learn onnx skl2onnx onnxruntime lightgbm xgboost \
            onnxmltools pandas numpy requests huggingface_hub tld pyarrow scipy
```

## Reproduce the winner

```bash
python train_phishscout.py          # trains, evaluates, exports /tmp/phishtrap_model/model.onnx
```

Expected test metrics (PhishTrap build `20260731T085613Z`, seed 42):
accuracy 0.8697, F1 0.8626, AUC 0.9369, 131.5 KB ONNX.

Training data comes from [saidutta69/PhishTrap](https://huggingface.co/datasets/saidutta69/PhishTrap)
(the auto-refreshing dataset — see the PhishTrap repo for its pipeline).

## Upload to Hugging Face

```bash
# Set HF_TOKEN in your environment (e.g. via `huggingface-cli login`)
python upload_model.py
```

## License

MIT — see the [model card](https://huggingface.co/saidutta69/PhishScout) for usage, limitations, and benchmark data.
