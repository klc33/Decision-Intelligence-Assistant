#!/usr/bin/env python
"""
Data Preparation Script – Phase 1

This script:
1. Loads the raw Twitter customer support dataset.
2. Filters to inbound customer tweets only.
3. Cleans text (drops missing values, strips whitespace, removes very short messages).
4. Samples exactly 500,000 rows.
5. Applies the labeling function to create a 'priority' column.
6. Saves the cleaned and labeled dataset to data/processed/.

Usage:
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

# Add project root to sys.path to import training module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from training.labeling import create_priority_labels

# Configuration
TARGET_ROWS = 500_000
RAW_DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed/tickets_with_labels.csv")

def main():
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("🔍 Loading raw data...")
    if not RAW_DATA_PATH.exists():
        print(f"❌ Raw data not found at {RAW_DATA_PATH}")
        print("   Please download twcs.csv from Kaggle and place it in data/raw/")
        sys.exit(1)

    # Read only necessary columns; oversample to account for filtering
    # Using nrows to limit memory usage (full file is ~3GB)
    df = pd.read_csv(
        RAW_DATA_PATH,
        usecols=['text', 'inbound'],
        nrows=TARGET_ROWS * 2  # Read extra rows because we'll filter some out
    )
    print(f"   Loaded {len(df):,} rows")

    # Keep only inbound customer tweets (inbound == True)
    if 'inbound' in df.columns:
        df = df[df['inbound'] == True].copy()
        print(f"   After inbound filter: {len(df):,} rows")
    else:
        print("   ⚠️ 'inbound' column not found; keeping all rows")

    # Drop rows with missing text
    df = df.dropna(subset=['text'])

    # Clean text: strip whitespace
    df['text'] = df['text'].str.strip()

    # Remove very short messages (likely noise or placeholders)
    df = df[df['text'].str.len() >= 10]

    print(f"   After cleaning: {len(df):,} rows")

    # Sample exactly TARGET_ROWS if we have enough; otherwise use all available
    if len(df) > TARGET_ROWS:
        df = df.sample(n=TARGET_ROWS, random_state=42)
        print(f"   Sampled down to {TARGET_ROWS:,} rows")
    else:
        print(f"   ⚠️ Only {len(df):,} rows available, using all")

    # Apply labeling function
    print("🏷️  Applying priority labels...")
    df['priority'] = df['text'].apply(create_priority_labels)

    # Keep only necessary columns and reset index
    df = df[['text', 'priority']].reset_index(drop=True)
    # Add a simple sequential ID
    df.insert(0, 'id', df.index)

    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df):,} labeled tickets to {OUTPUT_PATH}")

    # Show label distribution
    print("\n📊 Priority distribution:")
    dist = df['priority'].value_counts(normalize=True) * 100
    for label, percent in dist.items():
        print(f"   {label}: {percent:.1f}%")

if __name__ == "__main__":
    main()