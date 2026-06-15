"""Download the primary symbol dataset (Digitize-PID, YOLO format) from Hugging Face.

Usage:
    python -m src.pidetect.data.download
"""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "hamzas/digitize-pid-yolo"
DEST = Path("data/digitize-pid-yolo")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {DEST} ...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(DEST))
    print("Done. Expect images/ + labels/ in YOLO format (train/val 4:1).")
    print("Next: python -m src.pidetect.data.inspect")


if __name__ == "__main__":
    main()
