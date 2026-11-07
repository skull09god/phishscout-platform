# PhishScout

Tiny, on-device phishing-URL detectors by Sai Dutta Abhishek Dash. This repo
holds two generations of the model, kept in separate folders so researchers can
reproduce or compare either.

## v2 (recommended)

A more powerful, still-tiny on-device detector: **579 URL features** (66 structured + 512 stateless char n-grams + shortener), a compact LightGBM exported to an **896 KB ONNX** (< 1 MB budget), **F1 0.8966 / ROC AUC 0.9615** on the held-out PhishTrap test split, fully deterministic, with an optional **page/TLS enrichment stage** (meta-model default) for higher recall when the device is online.

- Code: [`v2/`](v2/)
- Model card: [`v2/README.md`](v2/README.md)
- Reproduce: `cd v2 && python -m src.phishscout_v2.train`

## v1

The original 35-feature, 131.5 KB model (F1 0.8626, AUC 0.9369). Kept as the
minimal-footprint option and for backwards-compatible comparison.

- Code: [`v1/`](v1/)

## Layout

```
PhishScout/
  v1/   # v1 training/benchmark scripts + card
  v2/   # v2 package (src/phishscout_v2), models/, card
  README.md, LICENSE
```

Both generations train on [saidutta69/PhishTrap](https://huggingface.co/datasets/saidutta69/PhishTrap),
auto-refreshed every 6 hours from OpenPhish, Phishing.Database, PhishStats, and Tranco.
MIT licensed.

## Planned improvements / TODO

- **Retrain on latest PhishTrap snapshot** — v2 was trained on the dataset as of this release; the feed refreshes every 6h, so retrain periodically for current campaigns.
- **Browser/JS port of the 512 char-n-gram features** — the card claims JS-portability via FNV-1a, but no JS feature extractor is shipped yet; needed for pure in-browser inference.
- **Compact v2 variant** — a smaller-config model (< 200 KB) for users who need the absolute smallest footprint.
- **CI / automated tests** — currently only run-scripts; add regression + parity checks in CI.
- **Workspace DOX docs** — `models/AGENTS.md` and `src/phishscout_v2/AGENTS.md` still carry pre-n-gram numbers (669 KB / 300×8L31) from earlier drafts; cosmetic update needed.
- **Enrichment meta-model** — currently trained on a live-fetched sample (546 rows, 5-fold CV F1 ~0.88); grow the fetched sample and add more page/TLS signals as needed.
