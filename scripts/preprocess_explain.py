from pathlib import Path
from textwrap import dedent

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, pipeline

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
DATA_ROOT = Path("data/cic-ids2018/processed/")
INPUT_DATA_PATH = DATA_ROOT / "cleaned.csv"
OUTPUT_DATA_PATH = DATA_ROOT / "explained.csv"
BATCH_SIZE = 64


def generate_prompt(row):
    attack_descriptions = {
        "DoS attacks-Slowloris": "Slowloris exhausts a server's resources by holding many connections open with partial HTTP requests using minimal bandwidth.",
        "DoS attacks-Hulk": "Hulk generates large volumes of HTTP requests with random parameters to overwhelm the server.",
        "DoS attacks-GoldenEye": "GoldenEye sends malformed HTTP requests at high frequency to degrade server responsiveness.",
        "DoS attacks-SlowHTTPTest": "SlowHTTPTest sends legitimate-looking HTTP traffic very slowly to keep server connections open indefinitely.",
        "DDOS attack-HOIC": "HOIC floods web servers with HTTP requests from multiple machines using booster scripts to randomize targets.",
        "DDoS attacks-LOIC-HTTP": "LOIC-HTTP overwhelms web services with multi-threaded GET/POST requests in distributed fashion.",
        "DDOS attack-LOIC-UDP": "LOIC-UDP floods targets with high-rate UDP packets from multiple machines.",
        "Bot": "This flow is from a machine infected with Zeus or Ares botnet, which periodically exfiltrate data or perform keylogging and remote commands.",
        "Brute Force -Web": "The attacker tried different username/password combinations on a web login interface in an automated fashion.",
        "Brute Force -XSS": "This attack injected malicious JavaScript to exploit cross-site scripting vulnerabilities.",
        "Infiltration": "A compromised internal host scanned internal network resources after being exploited through a malicious document.",
        "Benign": "This flow represents normal activity in a typical corporate network, such as file transfers, browsing, or background services.",
    }

    label = row["Label"]
    fine_label = row["Fine Label"]
    desc = attack_descriptions.get(fine_label, "No description available.")

    prompt = dedent(
        f"""You are a cybersecurity analyst. Your task is to explain why the following network flow is labeled as a {fine_label} attack. This flow is part of the CIC-IDS2018 dataset, which includes realistic attack scenarios and benign behavior.

        ### Ground Truth
        - Label (High-level): {label}
        - Label (Specific Variant): {fine_label}

        ### Background on Attack
        {desc}

        ### INSTRUCTIONS
        Only use the flow features and values below to construct your explanation. Do NOT reference any external knowledge in your explanation. Justify the label using only flow-level behavioral characteristics.

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

        ### Your Task
        Based on these features alone, write a BRIEF explanation of why this flow is consistent with a {label} attack. Make sure your response includes which features specifically lead you to believe that it is the said attack, focusing on only the most indicitave features. Respond ONLY with the explanation, without any additional commentary or preamble.
        """
    )

    return prompt


print("Loading cleaned data...")
df = pd.read_csv(INPUT_DATA_PATH, low_memory=False).sample(n=1000).reset_index(drop=True)

# Filter to attacks only
print(f"Original shape: {df.shape}")
attack_df = df[df["Label"] != "Benign"].copy()
print(f"Attack samples only: {attack_df.shape}")

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

# Generation Loop
print(f"\nStarting generation for {len(attack_df)} samples...")
explanations = []

# Prepare all prompts
print("Preparing prompts...")
prompts = [generate_prompt(row) for _, row in attack_df.iterrows()]

print("Generating explanations...")
for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch_prompts = prompts[i : i + BATCH_SIZE]

    # Generate
    outputs = pipe(
        batch_prompts,
        batch_size=BATCH_SIZE,
        max_new_tokens=1000,
        do_sample=True,
        truncation=True,
        return_full_text=False,
    )

    for out in outputs:
        explanation = out[0]["generated_text"].strip()
        explanations.append(explanation)

print(explanations)
