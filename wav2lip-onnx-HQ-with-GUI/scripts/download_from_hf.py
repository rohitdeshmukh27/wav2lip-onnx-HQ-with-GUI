"""
Download files directly from Hugging Face Hub into repo model folders.

Example (PowerShell):
  python .\scripts\download_from_hf.py --repo_id username/wav2lip-onnx-model --filename wav2lip_256.onnx --dest checkpoints

If the model repo is private, set the environment variable:
  $env:HUGGINGFACE_HUB_TOKEN = "<your_token>"
or pass --token <your_token>

This uses `huggingface_hub.hf_hub_download` which caches downloads. The script will move the downloaded file to the requested dest folder.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from huggingface_hub import hf_hub_download


def download_from_hf(repo_id: str, filename: str, dest_folder: str, token: str | None = None) -> str:
    # hf_hub_download will return a path in the HF cache. We'll copy it to dest_folder.
    os.makedirs(dest_folder, exist_ok=True)
    print(f"Downloading {filename} from {repo_id} into {dest_folder}")
    local_cached_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    dest_path = os.path.join(dest_folder, os.path.basename(filename))
    if os.path.exists(dest_path):
        print(f"File already exists at destination: {dest_path} (skipping copy)")
        return dest_path
    shutil.copy(local_cached_path, dest_path)
    print(f"Saved to: {dest_path}")
    return dest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download model file(s) from Hugging Face Hub into a repo folder")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo id, e.g. username/repo-name")
    parser.add_argument("--filename", action="append", required=True, help="Filename inside the HF repo to download. Repeat to download multiple files.")
    parser.add_argument("--dest", default="checkpoints", help="Destination folder inside the repo")
    parser.add_argument("--token", help="Hugging Face token (optional). If not provided, HF client will check HUGGINGFACE_HUB_TOKEN env var or local cache.")

    args = parser.parse_args(argv)
    token = args.token
    repo_id = args.repo_id
    dest = os.path.abspath(args.dest)

    for fname in args.filename:
        try:
            download_from_hf(repo_id, fname, dest, token)
        except Exception as e:
            print(f"Failed to download {fname} from {repo_id}: {e}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
