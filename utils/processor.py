from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from metrics import Evaluator
from models import Model
from utils.misc import send_to_device

Criterion = Callable[[Tensor, Tensor], Tensor]


def train(
    model: Model,
    optimizer: optim.Optimizer,
    criterion: Criterion,
    train_data: DataLoader,
    val_data: DataLoader,
    epochs: int,
    output_dir: str,
    run_name: str,
    evaluator: Evaluator = None,
    primary_metric: str = None,
    device: str = "cpu",
) -> None:
    """
    Train a model.

    :param model: Model to train.
    :param optimizer: Optimizer.
    :param criterion: Loss function.
    :param train_data: Training data.
    :param val_data: Validation data.
    :param epochs: Number of epochs to train for.
    :param output_dir: Parent directory to save the weights to.
    :param run_name: Name of the run.
    :param evaluator: Evaluator to use.
    :param primary_metric: Primary metric to use for model selection.
    :param device: Device to use.
    """

    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if we're using metrics or loss for model selection
    use_metrics = evaluator is not None and primary_metric is not None
    best_score = float("-inf") if use_metrics else float("inf")

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, criterion, train_data, epoch, device)

        val_loss, val_metrics = evaluate(model, criterion, val_data, epoch, evaluator, device)

        # Log validation results
        log_data = {"val": {"loss": val_loss}}
        if val_metrics is not None:
            log_data["val"]["metric"] = val_metrics
        wandb.log(log_data, step=wandb.run.step)

        # Save the latest model
        torch.save(model.state_dict(), output_dir / "last.pt")

        # Determine if this is the best model so far
        current_score = val_metrics[primary_metric] if use_metrics else val_loss
        is_best = current_score >= best_score if use_metrics else current_score <= best_score

        if is_best:
            best_score = current_score
            torch.save(model.state_dict(), output_dir / "best.pt")


def train_one_epoch(
    model: Model,
    optimizer: optim.Optimizer,
    criterion: Criterion,
    data: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    """
    Train a model for one epoch.

    :param model: Model to train.
    :param optimizer: Optimizer.
    :param criterion: Loss function.
    :param data: Training data.
    :param epoch: Current epoch.
    :param device: Device to train on.
    """

    # Set the model to training mode
    model.train()

    for features, targets in tqdm(data, desc=f"Training (Epoch {epoch})", dynamic_ncols=True):
        # Zero the gradients
        optimizer.zero_grad()

        # Send the batch to the training device
        features, targets = send_to_device(features, device), send_to_device(targets, device)

        # Forward pass
        if isinstance(features, (Tuple, List)):
            predictions = model(*features)
        else:
            predictions = model(features)

        # Compute the loss
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log the training loss
        wandb.log({"train": {"loss": loss.item()}}, step=wandb.run.step + len(features))


@torch.no_grad()
def evaluate(
    model: Model,
    criterion: Criterion,
    data: DataLoader,
    epoch: int,
    evaluator: Evaluator = None,
    device: str = "cpu",
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate a model.

    :param model: Model to evaluate.
    :param criterion: Loss function.
    :param data: Data to evaluate.
    :param epoch: Current epoch.
    :param evaluator: Evaluator to use.
    :param device: Device to evaluate on.
    :return: Average loss and average metrics.
    """
    # Set the model to evaluation mode
    model.eval()

    # Keep track of the running loss
    loss = 0.0
    if evaluator is not None:
        evaluator.reset()

    for features, targets in tqdm(data, desc=f"Validation (Epoch {epoch})", dynamic_ncols=True):
        # Send the batch to the evaluation device
        features, targets = send_to_device(features, device), send_to_device(targets, device)

        # Forward pass
        if isinstance(features, (Tuple, List)):
            predictions = model(*features)
        else:
            predictions = model(features)

        # Compute the loss
        batch_loss = criterion(predictions, targets)

        # Update the running loss & evaluator
        loss += batch_loss.item()
        if evaluator is not None:
            evaluator.update(predictions, targets)

    # Calculate the average loss and metrics
    loss = loss / len(data)
    metrics = evaluator.compute() if evaluator is not None else None

    return loss, metrics
