from torch import Tensor
from typing import Dict
from .evaluator import Evaluator
import torch


class IDSEvaluator(Evaluator):
    """
    Intrusion Detection System evaluator for classification tasks.

    Calculates accuracy, precision, recall, and F1-score.
    """

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        self.predicted_scores = []
        self.predicted_labels = []
        self.target_labels = []

    def update(self, predicted_logits: Tensor, target_labels: Tensor) -> None:
        """
        Update the evaluator with new predictions and targets.

        :param predicted_logits: Predicted logits of shape (batch_size, num_classes).
        :param target_labels: Ground truth labels of shape (batch_size,).
        """

        predicted_scores = predicted_logits.softmax(dim=-1)
        predicted_labels = torch.argmax(predicted_scores, dim=-1)

        self.predicted_scores.append(predicted_scores.cpu())
        self.predicted_labels.append(predicted_labels.cpu())
        self.target_labels.append(target_labels.cpu())

    def compute(self) -> Dict[str, float]:
        """
        Compute the evaluation metrics.

        :return: A dictionary containing accuracy, precision, recall, and F1-score.
        """

        predicted_labels = torch.cat(self.predicted_labels)
        target_labels = torch.cat(self.target_labels)

        true_positives = ((predicted_labels == 1) & (target_labels == 1)).sum().item()
        true_negatives = ((predicted_labels == 0) & (target_labels == 0)).sum().item()
        false_positives = ((predicted_labels == 1) & (target_labels == 0)).sum().item()
        false_negatives = ((predicted_labels == 0) & (target_labels == 1)).sum().item()

        accuracy = (true_positives + true_negatives) / len(target_labels) if len(target_labels) > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
