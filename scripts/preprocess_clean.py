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
print(f"Removing identifier columns: {identifier_cols}...")
df.drop(columns=identifier_cols, inplace=True)

# Remove zero-variance columns
nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
if len(cols_to_drop) > 0:
    print(f"Removing zero-variance columns: {list(cols_to_drop)}...")
    df.drop(columns=cols_to_drop, inplace=True)

# Convert feature columns to numerics
feature_cols = [c for c in df.columns if c not in ["Label"]]
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

# One-hot encode categorical features
df["Protocol"] = df["Protocol"].astype(int)
df = pd.get_dummies(df, columns=["Protocol"], dtype=int)
df

# Remove rows with inf/-inf/NaN values
print("Removing rows with inf/-inf/NaN values...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Fix label typos and remove rows with missing labels
print(f"Cleaning labels...")
df["Fine Label"] = df["Label"].replace({"Infilteration": "Infiltration", "Label": np.nan})
df.dropna(subset=["Fine Label"], inplace=True)

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
    "Infiltration": "Infiltration",
    "Bot": "Bot",
}

df["Label"] = df["Fine Label"].map(label_mapping)
df.dropna(subset=["Label"], inplace=True)

# Split into a train and validation set
train_indices, val_indices = train_test_split(df.index, test_size=0.2, random_state=42)
df["Split"] = "train"
df.loc[val_indices, "Split"] = "val"

processed_dir = data_dir / "processed"
processed_dir.mkdir(exist_ok=True)
print(f"Saving cleaned data to {processed_dir / 'cleaned.csv'}...")
df.to_csv(processed_dir / "cleaned.csv", index=False)
