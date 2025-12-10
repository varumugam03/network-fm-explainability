from pathlib import Path

import torch
import transformers
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ExplainDataSet
from models import NetworkExplainer
from utils.misc import set_random_seed

# --- Config ---
DATA_ROOT = Path("data/cic-ids2018/processed/")
# Update this filename to whatever your partner gives you 
# (e.g. if they just add a col to 'cleaned.csv' or save a new 'merged.csv')
DATA_FILE = DATA_ROOT / "explained.csv" 

MLP_CHECKPOINT = "weights/cic-ids2018/best.pt"
LLM_ID = "Qwen/Qwen2.5-32B-Instruct" 

EPOCHS = 3
BATCH_SIZE = 4 
LR = 2e-4
GRAD_ACCUM = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(42)

def main():
    wandb.init(project="network-explainability", name="lora-finetune")

    # 1. Dataset
    print(f"Initializing Dataset from {DATA_FILE}...")
    
    # Updated: Just passing the single file path now
    train_dataset = ExplainDataSet(data_path=DATA_FILE)
    
    def collate_fn(batch):
        mlp_feats = torch.stack([b['mlp_features'] for b in batch])
        prompts = [b['text_prompt'] for b in batch]
        grounds = [b['ground_truth'] for b in batch]
        return mlp_feats, prompts, grounds

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )

    # 2. Model
    # Important: The 'input_dim' here must match the number of numeric columns 
    # produced by ExplainDataSet (which drops Label, Fine Label, and explanation).
    # It's safer to grab this dynamically from the dataset if possible, or hardcode if known.
    input_dim = train_dataset.numeric_features.shape[1]
    
    mlp_config = {
        "input_dim": input_dim,
        "hidden_dims": [768, 768],
        "output_dim": 5, # Number of IDS classes
        "token_dim": 5120 # MATCHES QWEN 32B (Use 4096 for 7B models)
    }
    
    print(f"Loading Explainer with MLP Input Dim: {input_dim}")
    
    model = NetworkExplainer(
        mlp_checkpoint=MLP_CHECKPOINT,
        llm_model_id=LLM_ID,
        mlp_config=mlp_config,
        device=DEVICE
    )

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 4. Training Loop
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for step, (mlp_feats, prompts, grounds) in enumerate(progress_bar):
            mlp_feats = mlp_feats.to(DEVICE)
            
            loss = model(mlp_feats, prompts, grounds)
            
            loss = loss / GRAD_ACCUM
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"train_loss": loss.item() * GRAD_ACCUM})
            
            total_loss += loss.item() * GRAD_ACCUM
            progress_bar.set_postfix({"loss": loss.item() * GRAD_ACCUM})
        
        # Save Adapter
        save_path = f"weights/explainer/epoch_{epoch}"
        model.llm.save_pretrained(save_path)
        print(f"Epoch {epoch} complete. Adapter saved to {save_path}")

if __name__ == "__main__":
    main()