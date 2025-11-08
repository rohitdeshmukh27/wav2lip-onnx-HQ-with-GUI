"""
Small helper to download large model files into the repository's model folders.

Usage (PowerShell):
  python .\scripts\download_models.py --dest checkpoints --url <MODEL_URL>
  python .\scripts\download_models.py --dest wav2lip_onnx_models --url <MODEL_URL>

This script is intentionally simple; replace <MODEL_URL> with the real URL for each model.
You can add multiple --url entries to download several files.

Note: If you plan to commit these large files to GitHub, use Git LFS (see README or comments below).
"""

from __future__ import annotations
import argparse
import os
import sys
from urllib.request import urlretrieve
from urllib.error import URLError


def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size <= 0:
        percent = "?"
    else:
        percent = f"{downloaded * 100 / total_size:.1f}%"
    end = '\r' if downloaded < total_size else '\n'
    print(f"Downloaded {downloaded}/{total_size} bytes ({percent})", end=end)


def download(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"File already exists, skipping: {dest_path}")
        return
    try:
        print(f"Downloading {url} -> {dest_path}")
        urlretrieve(url, dest_path, _reporthook)
    except URLError as e:
        print(f"Failed to download {url}: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download model file(s) into repo model folder")
    parser.add_argument("--dest", required=True, help="Destination folder inside the repo (e.g. checkpoints or wav2lip_onnx_models)")
    parser.add_argument("--url", action="append", required=True, help="Model file URL to download. Repeat for multiple files.")
    parser.add_argument("--name", action="append", help="Optional filename(s) to save as. If not provided, filename is taken from URL.")

    args = parser.parse_args(argv)
    dest_folder = os.path.abspath(args.dest)
    urls = args.url
    names = args.name or []

    for i, url in enumerate(urls):
        if i < len(names) and names[i]:
            fname = names[i]
        else:
            fname = os.path.basename(url.split("?")[0]) or f"model_{i}"
        dest_path = os.path.join(dest_folder, fname)
        download(url, dest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
