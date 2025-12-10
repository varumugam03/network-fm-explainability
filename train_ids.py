import json
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from metrics import IDSEvaluator

import wandb
from datasets import IDSDataSet
from models import MLP
from utils.misc import set_random_seed
from utils.processor import train


LLM_EMBEDDING_DIM = 5120 # Qwen2.5-32B hidden size. Use 4096 for Llama-7B/Qwen-7B.

DATA_ROOT = Path("data/cic-ids2018/processed/")
RUN_NAME = "cic-ids2018"
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
NUM_WORKERS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(42)

# Create the datasets & dataloaders
train_dataset = IDSDataSet(DATA_ROOT / "train.csv")
val_dataset = IDSDataSet(DATA_ROOT / "test.csv")

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

# Create the model & optimizer
model = MLP(
    input_dim=train_dataset.num_features, 
    hidden_dims=[768, 768], 
    output_dim=train_dataset.num_classes,
    token_dim=LLM_EMBEDDING_DIM # <--- Added this
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Define the criterion and metric evaluator
criterion = torch.nn.CrossEntropyLoss()

evaluator = IDSEvaluator()

# Initialize logging
wandb.init(dir="./logs", project="intrusion-detection", name=RUN_NAME)

train(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    evaluator=evaluator,
    primary_metric="f1",
    train_data=train_loader,
    val_data=val_loader,
    epochs=EPOCHS,
    output_dir="weights",
    run_name=RUN_NAME,
    device=DEVICE,
)
