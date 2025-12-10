from pathlib import Path
from textwrap import dedent

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, pipeline

# Configuration
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
DATA_ROOT = Path("data/cic-ids2018/processed/")
INPUT_DATA_PATH = DATA_ROOT / "cleaned.csv"
OUTPUT_DATA_PATH = DATA_ROOT / "explained.csv"
CHECKPOINT_PATH = DATA_ROOT / "explained_checkpoint.csv"
BATCH_SIZE = 64
TOTAL_SAMPLES = 256
SAVE_EVERY = 2
SEED = 42

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
print(f"Loading data: {INPUT_DATA_PATH}...")
df = pd.read_csv(INPUT_DATA_PATH, low_memory=False)

# Sample indices
print(f"Sampling {TOTAL_SAMPLES} rows (seed={SEED}) using stratified sampling...")
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
sampled_indices = sampled_df.index.to_numpy()
print(f"Total samples to process: {len(sampled_indices)}")

# Load checkpoint if exists
results = {}
completed_indices = set()
if CHECKPOINT_PATH.exists():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint_df = pd.read_csv(CHECKPOINT_PATH, index_col=0)
    results = checkpoint_df["explanation"].to_dict()
    completed_indices = set(checkpoint_df.index.tolist())
    print(f"Resuming with {len(completed_indices)} completed samples")

# Filter remaining
remaining_indices = [idx for idx in sampled_indices if idx not in completed_indices]
print(f"Remaining samples: {len(remaining_indices)}")

if len(remaining_indices) == 0:
    print("All samples already processed!")
else:
    # Load model
    print(f"Loading model: {MODEL_ID}...")
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
    print("Generating explanations...")
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
                - Flow Duration: {row['Flow Duration']}
                - Tot Fwd Pkts: {row['Tot Fwd Pkts']}
                - TotLen Fwd Pkts: {row['TotLen Fwd Pkts']}
                - Fwd Pkt Len Max: {row['Fwd Pkt Len Max']}
                - Fwd Pkt Len Mean: {row['Fwd Pkt Len Mean']}
                - Fwd IAT Tot: {row['Fwd IAT Tot']}
                - Fwd IAT Mean: {row['Fwd IAT Mean']}
                - Fwd IAT Max: {row['Fwd IAT Max']}
                - Fwd IAT Min: {row['Fwd IAT Min']}
                - Flow IAT Min: {row['Flow IAT Min']}
                - Flow IAT Max: {row['Flow IAT Max']}
                - Flow IAT Mean: {row['Flow IAT Mean']}
                - Fwd Seg Size Min: {row['Fwd Seg Size Min']}
                - Fwd Seg Size Avg: {row['Fwd Seg Size Avg']}
                - Flow Pkts/s: {row['Flow Pkts/s']}
                - Fwd Pkts/s: {row['Fwd Pkts/s']}
                - Bwd Pkts/s: {row['Bwd Pkts/s']}
                - Fwd Header Len: {row['Fwd Header Len']}
                - Init Fwd Win Byts: {row['Init Fwd Win Byts']}
                - Init Bwd Win Byts: {row['Init Bwd Win Byts']}
                - Pkt Len Max: {row['Pkt Len Max']}
                - Subflow Fwd Byts: {row['Subflow Fwd Byts']}
                - Subflow Fwd Pkts: {row['Subflow Fwd Pkts']}

                ### INSTRUCTIONS
                Using the flow features construct a concise (2-3 sentences) explanation of why this flow is consistent with a {label} attack. Do NOT include any of the ground truth information in the explanation itself, using it only for added context. Make sure your response includes which features specifically lead you to believe that it is the said attack, focusing on only the most indicitave features. Respond ONLY with the explanation, without any additional commentary or preamble.
                """
            )
            batch_prompts.append(prompt)

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
            print(f"Checkpoint saved: {len(results_df)} explanations")

    # Final checkpoint
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["explanation"])
    results_df.to_csv(CHECKPOINT_PATH)
    print(f"Checkpoint saved: {len(results_df)} explanations")

# Join to original dataframe and save
print("Saving final output...")
df["explanation"] = None
for idx, explanation in results.items():
    df.loc[idx, "explanation"] = explanation

df.to_csv(OUTPUT_DATA_PATH, index=False)
print(f"Saved to {OUTPUT_DATA_PATH}")
print(f"  - Total rows: {len(df)}")
print(f"  - Rows with explanations: {df['explanation'].notna().sum()}")
