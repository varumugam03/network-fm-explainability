from pathlib import Path

import pandas as pd

DATA_ROOT = Path("data/cic-ids2018/processed")
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
INPUT_DATA_PATH = DATA_ROOT / "cleaned.csv"
OUTPUT_DATA_PATH = DATA_ROOT / "explained.csv"

print(f"Loading data: {INPUT_DATA_PATH}...")
df = pd.read_csv(INPUT_DATA_PATH, low_memory=False)
df["explanation"] = None

print(f"Merging checkpoints from {CHECKPOINT_DIR}...")
for checkpoint_path in sorted(CHECKPOINT_DIR.glob("rank*.csv")):
    checkpoint_df = pd.read_csv(checkpoint_path, index_col=0)
    for idx, row in checkpoint_df.iterrows():
        df.loc[idx, "explanation"] = row["explanation"]
    print(f"  - Loaded {len(checkpoint_df)} explanations from {checkpoint_path.name}")

df.to_csv(OUTPUT_DATA_PATH, index=False)
print(f"Saved to {OUTPUT_DATA_PATH}")
print(f"  - Total rows: {len(df)}")
print(f"  - Rows with explanations: {df['explanation'].notna().sum()}")
