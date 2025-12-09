from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path("data/cic-ids2018/")

# Move all CSV files into a raw subdirectory
raw_dir = data_dir / "raw"
raw_dir.mkdir(exist_ok=True)

for file in data_dir.glob("*.csv"):
    print(f"Moving {file} to {raw_dir}...")
    file.rename(raw_dir / file.name)

# Load all CSV files from the raw directory
all_files = sorted(raw_dir.glob("*.csv"))
all_dfs = []
for filename in all_files:
    print(f"Loading {filename}...")
    all_dfs.append(pd.read_csv(filename, low_memory=False))

df: pd.DataFrame = pd.concat(all_dfs, ignore_index=True)

identifier_cols = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
df = df.drop(columns=identifier_cols)

# Remove zero-variance columns
nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
if len(cols_to_drop) > 0:
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped zero-variance columns: {list(cols_to_drop)}")

# Convert feature columns to numerics
feature_cols = [c for c in df.columns if c not in ["Label"]]
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

# Remove rows with inf/-inf/NaN values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Fix label typos and remove rows with missing labels
df["Fine Label"] = df["Label"].replace({"Infilteration": "Infiltration", "SQL Injection": np.nan, "Label": np.nan})
df = df.dropna(subset=["Fine Label"])

# Aggregate labels into broader attack categories
label_mapping = {
    "Benign": "Benign",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "DDOS attack-HOIC": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "Bot": "Bot",
    "Brute Force -Web": "Brute Force",
    "Brute Force -XSS": "Brute Force",
    "Infiltration": "Infiltration",
}

df["Label"] = df["Fine Label"].map(label_mapping)
df.dropna(subset=["Label"], inplace=True)

processed_dir = data_dir / "processed"
processed_dir.mkdir(exist_ok=True)
df.to_csv(processed_dir / "cleaned.csv", index=False)
