#!/usr/bin/env python3
"""Push the final PhishScout model to Hugging Face.

Usage:
    # Set HF_TOKEN in your environment (e.g. via `huggingface-cli login`)
    python upload_model.py

Uploads model.onnx, model_info.json, features.json, README.md, banner.png to
saidutta69/PhishScout. Tags the release v1.
"""
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file

MODEL_REPO = "saidutta69/PhishScout"
SRC = Path("/tmp/phishtrap_model")

if __name__ == "__main__":
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")

    create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False, token=token)

    files = ["model.onnx", "model_info.json", "features.json", "README.md", "banner.png"]
    for f in files:
        path = SRC / f
        if not path.exists():
            print(f"  skipping missing {f}")
            continue
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f,
            repo_id=MODEL_REPO,
            repo_type="model",
            token=token,
            commit_message=f"upload {f}",
        )
        print(f"  uploaded {f} ({path.stat().st_size/1024:.1f} KB)")

    api.create_tag(repo_id=MODEL_REPO, repo_type="model", tag="v1", token=token)
    print(f"  tagged v1")
    print(f"model live at https://huggingface.co/{MODEL_REPO}")
