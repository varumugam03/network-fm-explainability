import os
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, pipeline

# Configuration
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
DATA_ROOT = Path("data/cic-ids2018/processed/")
INPUT_DATA_PATH = DATA_ROOT / "cleaned.csv"
OUTPUT_DATA_PATH = DATA_ROOT / "explained.csv"
BATCH_SIZE = 32
TOTAL_SAMPLES = 1000
SAVE_EVERY = 25
SEED = 42

# Distributed settings (SLURM)
RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / f"rank{RANK}.csv"

if RANK == 0:
    print(f"World Size: {WORLD_SIZE}")

DESCRIPTIONS = {
    "Benign": "Normal activity in a typical corporate network, such as file transfers, browsing, or background services.",
    "DoS attacks-Slowloris": "Slowloris exhausts a server's resources by holding many connections open with partial HTTP requests using minimal bandwidth.",
    "DoS attacks-Hulk": "Hulk generates large volumes of HTTP requests with random parameters to overwhelm the server.",
    "DoS attacks-GoldenEye": "GoldenEye sends malformed HTTP requests at high frequency to degrade server responsiveness.",
    "DoS attacks-SlowHTTPTest": "SlowHTTPTest sends legitimate-looking HTTP traffic very slowly to keep server connections open indefinitely.",
    "DDOS attack-HOIC": "HOIC floods web servers with HTTP requests from multiple machines using booster scripts to randomize targets.",
    "DDoS attacks-LOIC-HTTP": "LOIC-HTTP overwhelms web services with multi-threaded GET/POST requests in distributed fashion.",
    "DDOS attack-LOIC-UDP": "LOIC-UDP floods targets with high-rate UDP packets from multiple machines.",
    "Bot": "A Zeus or Ares botnet periodically exfiltrates data or performs keylogging and remote commands on an infected machine.",
    "Infiltration": "A compromised internal host scanned internal network resources after being exploited through a malicious document.",
}

# Load data
print(f"[{RANK}] Loading data: {INPUT_DATA_PATH}...")
df = pd.read_csv(INPUT_DATA_PATH, low_memory=False)
protocol_cols = [col for col in df.columns if col.startswith("Protocol_")]

# Sample indices
print(f"[{RANK}] Sampling {TOTAL_SAMPLES} rows (seed={SEED}) using stratified sampling...")
labels = df["Label"].unique()
per_label = TOTAL_SAMPLES // len(labels)

sampled_parts = []
for lbl in labels:
    part = df[df["Label"] == lbl].sample(
        n=min(per_label, len(df[df["Label"] == lbl])),
        random_state=SEED,
    )
    sampled_parts.append(part)

sampled_df = pd.concat(sampled_parts)
all_sampled_indices = sampled_df.index.to_numpy()

# Split indices across ranks
sampled_indices = np.array_split(all_sampled_indices, WORLD_SIZE)[RANK]
print(f"[{RANK}] Processing {len(sampled_indices)}/{len(all_sampled_indices)} samples")

# Load checkpoint if exists
results = {}
completed_indices = set()
if CHECKPOINT_PATH.exists():
    print(f"[{RANK}] Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint_df = pd.read_csv(CHECKPOINT_PATH, index_col=0)
    results = checkpoint_df["explanation"].to_dict()
    completed_indices = set(checkpoint_df.index.tolist())
    print(f"[{RANK}] Resuming with {len(completed_indices)} completed samples")

# Filter remaining
remaining_indices = [idx for idx in sampled_indices if idx not in completed_indices]
print(f"[{RANK}] Remaining samples: {len(remaining_indices)}")

if len(remaining_indices) == 0:
    print(f"[{RANK}] All samples already processed!")
else:
    # Load model
    print(f"[{RANK}] Loading model: {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        device=device,
        dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "flash_attention_2"},
    )

    # Generate explanations
    print(f"[{RANK}] Generating explanations...")
    batch_count = 0
    for i in tqdm(range(0, len(remaining_indices), BATCH_SIZE)):
        batch_indices = remaining_indices[i : i + BATCH_SIZE]
        batch_prompts = []

        for idx in batch_indices:
            row = df.loc[idx]
            label = row["Label"]
            fine_label = row["Fine Label"]
            desc = DESCRIPTIONS.get(fine_label, "No description available.")
            prompt = dedent(
                f"""
                You are a cybersecurity analyst. Your task is to explain why the following network flow from the CIC-IDS2018 dataset is labeled as a {label} attack.

                ### Ground Truth
                - Label: {label}
                - Description: {desc}

                ### Flow Features
                - Dst Port: {row['Dst Port']}
                - Protocol: {row[protocol_cols].idxmax().split("_")[1]}
                - Flow Duration: {row['Flow Duration']}

                - Tot Fwd Pkts: {row['Tot Fwd Pkts']}
                - Tot Bwd Pkts: {row['Tot Bwd Pkts']}
                - TotLen Fwd Pkts: {row['TotLen Fwd Pkts']}
                - TotLen Bwd Pkts: {row['TotLen Bwd Pkts']}
                - Down/Up Ratio: {row['Down/Up Ratio']}

                - Fwd Pkt Len Max: {row['Fwd Pkt Len Max']}
                - Fwd Pkt Len Min: {row['Fwd Pkt Len Min']}
                - Fwd Pkt Len Mean: {row['Fwd Pkt Len Mean']}
                - Fwd Pkt Len Std: {row['Fwd Pkt Len Std']}
                - Bwd Pkt Len Max: {row['Bwd Pkt Len Max']}
                - Bwd Pkt Len Min: {row['Bwd Pkt Len Min']}
                - Bwd Pkt Len Mean: {row['Bwd Pkt Len Mean']}
                - Bwd Pkt Len Std: {row['Bwd Pkt Len Std']}

                - Pkt Len Max: {row['Pkt Len Max']}
                - Pkt Len Min: {row['Pkt Len Min']}
                - Pkt Len Mean: {row['Pkt Len Mean']}
                - Pkt Len Std: {row['Pkt Len Std']}
                - Pkt Size Avg: {row['Pkt Size Avg']}

                - Flow Byts/s: {row['Flow Byts/s']}
                - Flow Pkts/s: {row['Flow Pkts/s']}
                - Fwd Pkts/s: {row['Fwd Pkts/s']}
                - Bwd Pkts/s: {row['Bwd Pkts/s']}

                - Flow IAT Mean: {row['Flow IAT Mean']}
                - Flow IAT Max: {row['Flow IAT Max']}
                - Flow IAT Min: {row['Flow IAT Min']}
                - Fwd IAT Mean: {row['Fwd IAT Mean']}
                - Fwd IAT Max: {row['Fwd IAT Max']}
                - Fwd IAT Min: {row['Fwd IAT Min']}
                - Bwd IAT Mean: {row['Bwd IAT Mean']}
                - Bwd IAT Max: {row['Bwd IAT Max']}
                - Bwd IAT Min: {row['Bwd IAT Min']}

                - SYN Flag Cnt: {row['SYN Flag Cnt']}
                - ACK Flag Cnt: {row['ACK Flag Cnt']}
                - RST Flag Cnt: {row['RST Flag Cnt']}
                - PSH Flag Cnt: {row['PSH Flag Cnt']}

                - Fwd Header Len: {row['Fwd Header Len']}
                - Bwd Header Len: {row['Bwd Header Len']}

                - Init Fwd Win Byts: {row['Init Fwd Win Byts']}
                - Init Bwd Win Byts: {row['Init Bwd Win Byts']}
                - Fwd Act Data Pkts: {row['Fwd Act Data Pkts']}

                - Active Mean: {row['Active Mean']}
                - Idle Mean: {row['Idle Mean']}

                ### INSTRUCTIONS
                Using the flow features, construct a concise (2-3 sentence) explanation of why this flow is consistent with a {label} attack. 
                Do NOT include any of the ground truth information in the explanation itself.

                Focus on *qualitative interpretation* of the features rather than repeating numerical values. 
                Describe behaviors using comparative terms such as “low,” “high,” “balanced,” “normal,” “burst-like,” or “intermittent,” and explain how these patterns align with the expected behavior of a {label} attack.
                Highlight only the most indicative features and explain the overall traffic pattern they suggest.

                Respond ONLY with the explanation, without any additional commentary, preamble, notes, or clarifications.
            """
            )
            batch_prompts.append([{"role": "user", "content": prompt}])

        # Generate
        outputs = pipe(
            batch_prompts,
            batch_size=BATCH_SIZE,
            max_new_tokens=1024,
            do_sample=True,
            truncation=True,
            return_full_text=False,
        )

        for idx, out in zip(batch_indices, outputs):
            results[idx] = out[0]["generated_text"].strip()

        batch_count += 1

        # Save checkpoint
        if batch_count % SAVE_EVERY == 0:
            results_df = pd.DataFrame.from_dict(results, orient="index", columns=["explanation"])
            results_df.to_csv(CHECKPOINT_PATH)
            print(f"[{RANK}] Checkpoint saved: {len(results_df)} explanations")

    # Final checkpoint
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["explanation"])
    results_df.to_csv(CHECKPOINT_PATH)
    print(f"[{RANK}] Checkpoint saved: {len(results_df)} explanations")

print(f"[{RANK}] Finished. Checkpoints saved to {CHECKPOINT_DIR}")
