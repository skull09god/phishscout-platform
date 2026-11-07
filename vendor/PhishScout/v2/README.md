---
tags:
- text-classification
- onnx
- phishing
- url-detection
- cybersecurity
- lightgbm
- on-device
- browser-extension
- edge-inference
license: mit
datasets:
- saidutta69/PhishTrap
---

# PhishScout

![RACER IS OP](https://res.cloudinary.com/cmazqjs6/image/upload/racer_is_op_banner_branded_pu7zud.png)

> Catch phishing URLs before they catch you — **896 KB of on-device LightGBM. No GPU, no cloud, no API calls. 579 URL features** (structured signals + stateless character n-grams), trained on PhishTrap's auto-refreshing 20K-URL dataset. Plus an **optional page/TLS enrichment stage** for higher recall when the device is online.

Priorities: **Quality > Size > Speed**

**Trained on:** [saidutta69/PhishTrap](https://huggingface.co/datasets/saidutta69/PhishTrap) — auto-refreshed every 6 hours from OpenPhish, Phishing.Database, PhishStats, and Tranco

---

## Model Overview

PhishScout v2 is a tiny gradient-boosted tree model that classifies URLs as phishing or legitimate using **URL structure alone** — no page content, no third-party lookups, no network calls at inference time. The core model is a single **669 KB ONNX file** that runs anywhere ONNX Runtime runs (Python, Node, browser via ONNX Runtime Web, mobile). When network is available, an **optional second stage** fetches the page and TLS certificate to recover cases the URL-only model misses.

Property | Value
--- | ---
Architecture | LightGBM (gradient boosting)
Config | 200 trees, max_depth 10, 63 leaves
Features | 579 (66 structured + 512 char n-grams + shortener)
Model size | **896.3 KB** (ONNX, opset 15)
Training time | ~5s on CPU
Inference latency | **~24 µs/URL** (median, CPU)
Calibration | Platt (sigmoid) on validation
ONNX parity | 100% (predictions match LightGBM exactly)
License | MIT

---

## Use Cases

### Primary — Real-time, on-device URL screening
- **Browser extension** — screen every URL before navigation/click, entirely locally. Zero data leaves the device; works offline; no privacy trade-offs.
- **Email / messaging security clients** — pre-click link inspection in mail and chat apps.
- **SMS / QR scanner apps** — decode -> classify -> warn, before opening the URL.

### Secondary — Infrastructure integration
- **Email gateway & firewall plugins** — lightweight first-pass filter; hand off high-probability URLs to heavier sandboxing.
- **Proxy / DNS-layer filtering** — classify requested URLs at the edge with single-digit-microsecond latency.
- **SIEM / SOAR enrichment** — batch-classify URLs from logs, tickets, and incident data.

### Tertiary — Research & data pipelines
- **URL crawler triage**, phishing trend measurement, benchmark baseline.

### Optional enrichment stage (when online)
- Boosts recall on **bare trusted-domain phishing** (a real `paypal.com/...` path after a compromise) by inspecting the live page and TLS cert — login forms, brand/domain mismatch, redirect-to-other-domain, invalid/Let's-Encrypt certs. Falls back to the core model if the fetch fails.

---

## Performance

### Final model (test split, held out, seed 42, config chosen by validation F1 — no test leakage)

Metric | Raw | Calibrated
--- | --- | ---
Accuracy | 0.8995 | 0.8981
Precision | 0.9229 | 0.9180
Recall | 0.8717 | 0.8744
F1 | **0.8966** | 0.8957
ROC AUC | **0.9615** | 0.9615

Confusion matrix (test, default threshold 0.5): TP=1252, FP=116, FN=245, TN=1381.

### Determinism & stability
- 10-seed variance on test: F1 0.8966 ± 0.0000, AUC 0.9615 ± 0.0000 — fully deterministic.

### Robustness
Perturbation | v1 F1 | v2 F1
--- | --- | ---
Clean | 0.8626 | 0.8966
5% extreme-value corruption | 0.8080 | 0.8306
Gaussian noise σ=0.5 | 0.7097 | 0.6699

v2's dense character-n-gram block raises clean and extreme-value robustness but gives uniform Gaussian feature noise more surface area (it perturbs the whole n-gram vector), so the σ=0.5 number is lower. This is an artificial stress test; at inference, features are computed exactly, not perturbed.

### Threshold behavior
- Default threshold 0.5 is near-optimal. v2 ships a calibrated probability and a recommended threshold tuned on validation; choose per use case (higher threshold = more precision, lower = more recall).

---

## v2 vs v1

Property | v1 (PhishScout) | v2 (PhishScout)
--- | --- | ---
Features | 35 | **579** (66 structured + 512 char n-grams + shortener)
Base algorithm | LightGBM 60x8L31 | LightGBM 200x10L63
ONNX size | 131.5 KB | **896.3 KB** (budget < 1 MB)
Test F1 | 0.8626 | **0.8966**
Test ROC AUC | 0.9369 | **0.9615**
Calibration | none | Platt (sigmoid)
Page/TLS stage | no | **optional, network-gated**
Deterministic | yes | yes (F1 std 0.0)

The v2 feature set combines 66 interpretable signals (the 35 v1 features plus 31 anti-evasion signals: homoglyph / IDN homographs, typosquatting via edit-distance to brands, TLD tricks, free-hosting abuse, open-redirect / encoded-URL tricks, URL shorteners, and a much larger multilingual + regional brand list) with **512 stateless hashed character n-grams** over the URL. The n-grams capture lexical patterns (random strings, brand fragments, lure keywords) that handcrafted features miss, and are computed with a portable FNV-1a hash so browser-side extraction matches training.

---

## Features (66)

The first 35 are the v1 set (16 URL-structure + 19 semantic signals). The 31 new signals target evasion:

- **Homoglyph / IDN** — `homoglyph_count`, `has_homoglyph`, `host_nonascii_count`, `has_mixed_script`
- **Typosquatting / brand** — `brand_min_editdist`, `brand_typo_in_subdomain`, `brand_apex_match`, `brand_in_subdomain_not_apex`, `brand_on_free_host`
- **TLD tricks** — `tld_is_suspicious`, `tld_looks_like_com`, `tld_is_cctld`
- **Structure** — `num_dots`, `num_labels_gt3`, `registrable_domain_length`, `leftmost_subdomain_len`, `leftmost_subdomain_entropy`, `leftmost_subdomain_vowel_ratio`, `has_long_random_subdomain`, `domain_digit_ratio`
- **Encoded-IP / scheme** — `has_octal_ip`, `has_decimal_ip`, `has_dword_ip`, `scheme_is_nonthttp`, `has_userinfo`
- **Redirect / trickery** — `open_redirect_param`, `encoded_url_in_path`, `host_is_free_host`, `consecutive_identical_chars`, `emoji_in_url`, `query_has_suspicious_keyword`
- **URL shorteners** — `host_is_shortener` (bit.ly, tinyurl, t.co, rebrand.ly, ...)

On top of these 66 structured signals, v2 adds **512 stateless hashed character n-grams** (`cng0..cng511`): L2-normalized hashed counts of every 3–5 character substring of the lowercased URL, using FNV-1a hashing (no stored vocabulary, so it is trivially reproducible in JavaScript for browser inference).

Full ordered list (exact inference order): `features.json`.

---

## Optional enrichment stage

When the calling device is online, `enrichment.py` (`PageTlsEnricher` + `EnrichedScorer`)
fetches the page and TLS certificate and extracts signals the URL-only core model
cannot see, then combines them with the core probability. The combiner is, by
default, a **data-driven logistic meta-model** (`enrichment_model.json`, built by
`train_enrichment.py`) trained on a live-fetched, cross-validated sample. Its
features are `[core_probability]` plus page/TLS signals:

- form/input/iframe field counts, external-link count
- brand/domain mismatch, brand mentions in the page title
- suspicious-keyword count, redirect-to-other-domain, redirect count
- TLS validity / Let's-Encrypt issuer, and **domain registration age via RDAP**
  (newly registered domains are a strong phishing signal)

If the fetch fails or there is no network, it **falls back to the core model** — the
enrichment stage is never required. This is the main lever for the hardest v1 case:
**bare trusted-domain phishing** (e.g. a real `paypal.com/...` path after a compromise)
where URL structure is invisible. Metrics (5-fold CV + held-out live sample) are in
`models/enrichment_model.json`.

## Usage

### Python (on-device core, ONNX Runtime)

```python
from phishscout_v2.onnx_model import PhishScoutV2
m = PhishScoutV2()
print(m.predict_url("http://paypa1-verify.ngrok.io/login?token=x"))
# -> {'probability': 0.96..., 'label': 1, 'is_phishing': True, ...}
```

### Python (with optional enrichment)

```python
from phishscout_v2.enrichment import EnrichedScorer
s = EnrichedScorer()
print(s.score("https://suspicious.example/login"))  # falls back to core if offline
```

### Browser (ONNX Runtime Web)

Load `model.onnx` into `ort.InferenceSession` with WebAssembly backend — runs fully in-browser, no server. Compute the 66 features in JS (order in `features.json`) and run.

---

## Verification & Reproducibility

- **ONNX parity** — ONNX Runtime predictions match LightGBM on 100% of 2,000 sampled test URLs.
- **Deterministic** — identical results across 10 seeds (LightGBM with fixed seed + 10-seed re-check).
- **No leakage** — train/val/test splits are disjoint; config selected on validation only.
- **Reproducible** — `python -m src.phishscout_v2.train` (sweep + export), `python -m src.phishscout_v2.benchmark` (determinism + robustness). Source: [github.com/instax-dutta/PhishScout](https://github.com/instax-dutta/PhishScout) (`v2/`).

---

## Limitations

- **URL-only core** — the on-device model uses no page content, TLS, or behavior; the optional stage covers this only when online.
- **Bare trusted-domain phishing** — still the hardest case; the enrichment stage helps but is network-dependent.
- **English/Western-leaning lists** — brands and keywords are broader than v1 but still incomplete for every region and language.
- **Feed dependence** — retrain on the latest PhishTrap snapshot for current campaigns (refreshes every 6 hours).
- **No temporal adaptation** — retrain between campaigns.
- **Size** — v2 is ~5x v1's footprint (669 KB vs 131.5 KB); still < 1 MB. Choose v1 if every byte matters.

---

## Training Data

Trained on PhishTrap (balanced 50/50, 70/15/15 train/val/test split). Phishing URLs from PyFunceble-verified feeds (Phishing.Database, OpenPhish, PhishStats); legitimate URLs from Tranco top 10K. See the [dataset card](https://huggingface.co/datasets/saidutta69/PhishTrap).

---

## Citation

```bibtex
@misc{saidutta69_2026_phishscout_v2,
  author = {Sai Dutta Abhishek Dash},
  title = {PhishScout v2: More powerful tiny on-device phishing URL detector},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/saidutta69/PhishScout-v2}},
  note = {Trained on the auto-refreshing PhishTrap dataset}
}
```

---

Built on [PhishTrap](https://huggingface.co/datasets/saidutta69/PhishTrap) + Phishing.Database, OpenPhish, PhishStats, and Tranco. MIT licensed.
