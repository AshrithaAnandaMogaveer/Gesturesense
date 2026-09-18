"""
download_dataset.py
───────────────────
Downloads the Indian Sign Language (ISL) dataset from Kaggle using kagglehub
and prints the local path so train_isl_model.py can use it.

Usage
-----
  python download_dataset.py

Requirements
------------
  pip install kagglehub

Kaggle credentials
------------------
Either set the environment variables:
  KAGGLE_USERNAME=<your_username>
  KAGGLE_KEY=<your_api_key>

Or place your kaggle.json at  ~/.kaggle/kaggle.json  (Linux/macOS)
or  C:\\Users\\<user>\\.kaggle\\kaggle.json  (Windows).

Dataset
-------
  mrigaankjaswal/indian-sign-language-to-characters-dataset
  Contains one sub-folder per ISL character/digit (A-Z, 0-9 and some
  special signs), each holding JPEG/PNG images of hand signs in front
  of a plain background.
"""

import os
import sys

try:
    import kagglehub
except ImportError:
    sys.exit(
        "[ERROR] kagglehub is not installed.\n"
        "Run:  pip install kagglehub\n"
        "Then re-run this script."
    )


DATASET_SLUG = "mrigaankjaswal/indian-sign-language-to-characters-dataset"


def download() -> str:
    """Download the dataset and return the local directory path."""
    print(f"[INFO] Downloading dataset: {DATASET_SLUG}")
    print("[INFO] This may take a few minutes on first run (cached afterward).")

    path = kagglehub.dataset_download(DATASET_SLUG)

    print(f"\n[OK]  Dataset ready at: {path}\n")

    # Verify we actually got image sub-folders
    try:
        entries = os.listdir(path)
        class_dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
        if class_dirs:
            print(f"[INFO] Found {len(class_dirs)} class folders: {sorted(class_dirs)}")
        else:
            # Some Kaggle datasets have one extra nesting level
            sub = os.path.join(path, entries[0]) if entries else path
            if os.path.isdir(sub):
                class_dirs = [
                    e for e in os.listdir(sub)
                    if os.path.isdir(os.path.join(sub, e))
                ]
                print(f"[INFO] Found {len(class_dirs)} class folders inside '{entries[0]}'")
            else:
                print("[WARN] Could not find class sub-folders automatically.")
    except Exception as exc:
        print(f"[WARN] Could not inspect dataset structure: {exc}")

    return path


if __name__ == "__main__":
    dataset_path = download()
    # Write the path to a small text file so train_isl_model.py can read it
    marker = os.path.join(os.path.dirname(__file__), ".dataset_path")
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(dataset_path)
    print(f"[INFO] Path saved to {marker}")
    print("[INFO] You can now run:  python train_isl_model.py")
