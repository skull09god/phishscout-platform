#!/usr/bin/env python3
"""Upload the PhishScout v2 artifacts to Hugging Face.

Pushes everything in models/ (model.onnx, model_info.json, features.json,
benchmark_results.json, enrichment_model.json if present) plus the model card
as README.md.

This is a publish step and is NOT run automatically. Set HF_HUB_TOKEN (or run
`huggingface-cli login`) and call with the target repo:

    python -m phishscout_v2.upload_model --repo saidutta69/PhishScout-v2

Set --dry-run to validate the local artifacts without uploading.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

MODELS = Path(__file__).resolve().parents[2] / "models"
CARD = Path(__file__).resolve().parent / "model_card.md"
DEFAULT_REPO = "saidutta69/PhishScout-v2"

ARTIFACTS = [
    "model.onnx",
    "model_info.json",
    "features.json",
    "benchmark_results.json",
    "enrichment_model.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    missing = [a for a in ARTIFACTS if not (MODELS / a).exists()]
    if missing and "enrichment_model.json" in missing:
        missing.remove("enrichment_model.json")  # optional
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    print("will upload from", MODELS)
    for a in ARTIFACTS:
        p = MODELS / a
        if p.exists():
            print(f"  {a:24s} {p.stat().st_size/1024:8.1f} KB")
    if CARD.exists():
        print(f"  {'README.md (model card)':24s} {CARD.stat().st_size/1024:8.1f} KB")

    if args.dry_run:
        print("\n[dry-run] no upload performed.")
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    for a in ARTIFACTS:
        p = MODELS / a
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=a,
                            repo_id=args.repo, repo_type="model")
    if CARD.exists():
        api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model")
    print(f"\nuploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
