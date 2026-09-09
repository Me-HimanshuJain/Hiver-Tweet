"""
00_download_data.py
===================
Programmatically downloads real data to avoid relying on hardcoded or synthetic
placeholders.

Downloads:
1. Customer Support on Twitter (Kaggle) via the `kaggle` CLI
   Requires ~/.kaggle/kaggle.json credentials.
2. Banking77 (PolyAI) via Hugging Face `datasets` library.

Outputs:
  data/twcs/twcs.csv
  Prints row counts and sample rows for verification.
"""

import os
import sys
import subprocess
import pandas as pd
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TWCS_DIR = os.path.join(DATA_DIR, "twcs")
TWCS_CSV = os.path.join(TWCS_DIR, "twcs", "twcs.csv")

os.makedirs(TWCS_DIR, exist_ok=True)

def download_kaggle():
    print("-- 1. Downloading Customer Support on Twitter (Kaggle) --")
    if os.path.exists(TWCS_CSV):
        print(f"Dataset already exists at {TWCS_CSV}. Skipping download.")
    else:
        print("Running `kaggle datasets download`...")
        try:
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "thoughtvector/customer-support-on-twitter",
                "-p", TWCS_DIR, "--unzip"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR downloading Kaggle dataset: {e}")
            print("Please ensure kaggle.json is configured in ~/.kaggle/kaggle.json")
            sys.exit(1)
        except FileNotFoundError:
            print("ERROR: kaggle CLI not found. Run `pip install kaggle`.")
            sys.exit(1)
        
    df = pd.read_csv(TWCS_CSV)
    print(f"\n[OK] Kaggle Dataset verified.")
    print(f"Total Rows: {len(df):,}")
    print("Sample rows:")
    print(df[["tweet_id", "author_id", "inbound", "text"]].head(3).to_string())
    print("\n")


def download_banking77():
    print("-- 2. Downloading Banking77 (Hugging Face datasets) --")
    print("Loading PolyAI/banking77...")
    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    
    print(f"\n[OK] Banking77 Dataset verified.")
    train_count = len(ds['train'])
    test_count = len(ds['test'])
    print(f"Total Rows: {train_count + test_count:,} (Train: {train_count:,}, Test: {test_count:,})")
    
    print("Sample rows (Train set):")
    sample_df = pd.DataFrame(ds['train'][:3])
    print(sample_df.to_string())
    print("\n")

if __name__ == "__main__":
    download_kaggle()
    download_banking77()
