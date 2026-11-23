"""
VeriMath — Model Download Utility
==================================
Downloads the Qwen2.5-1.5B-Instruct GGUF (Q4_K_M quantized, ~980 MB) model
for local edge LLM inference on Raspberry Pi 4 (4 GB RAM).
Default target: models/qwen2.5-1.5b-instruct-q4_k_m.gguf or LLM_MODEL_PATH.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

QWEN_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
QWEN_3B_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)
PHI3_URL = (
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/"
    "Phi-3-mini-4k-instruct-q4.gguf"
)

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"File already exists at: {dest_path}")
        return

    print(f"Downloading model from:\n  {url}\nDestination:\n  {dest_path}")
    print("Download may take a few minutes depending on internet speed...")

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        mb_downloaded = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        sys.stdout.write(f"\rProgress: {percent}% [{mb_downloaded:.1f} MB / {total_mb:.1f} MB]")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
        print("\nDownload complete! Model ready.")
    except Exception as e:
        print(f"\nFailed to download model: {e}")
        if dest_path.exists():
            dest_path.unlink()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download VeriMath local GGUF model")
    parser.add_argument("--3b", dest="use_3b", action="store_true", help="Download Qwen2.5-3B-Instruct Q4_K_M GGUF (~2.0 GB, ideal for 8 GB Pi 4)")
    parser.add_argument("--phi3", action="store_true", help="Download legacy Phi-3 Mini 4k GGUF (~2.2 GB)")
    args = parser.parse_args()

    if args.phi3:
        url = PHI3_URL
        default_filename = "phi-3-mini-4k-instruct-q4.gguf"
    elif args.use_3b:
        url = QWEN_3B_URL
        default_filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
    else:
        url = QWEN_URL
        default_filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    target_env = os.getenv("LLM_MODEL_PATH")
    if target_env:
        target_path = Path(target_env)
    else:
        target_path = Path(__file__).parent.parent / "models" / default_filename

    download_file(url, target_path)

if __name__ == "__main__":
    main()
