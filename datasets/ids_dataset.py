from torch.utils.data import Dataset
from typing import Tuple
import pandas as pd
import torch
from torch import Tensor

LABELS = [
    "Benign",
    "DoS",
    "DDoS",
    "Bot",
    "Web",
    "Infiltration",
]

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

NUM_CLASSES = len(LABELS)


class IDSDataSet(Dataset):
    """
    Intrusion Detection System Dataset based on CIC-IDS2018.

    :param path: Path to the preprocessed CSV file.
    """

    def __init__(self, path: str) -> None:
        self.data = pd.read_csv(path)

        assert set(self.data["Label"].unique()).issubset(
            set(LABELS)
        ), f"Invalid labels found in {path}: {set(self.data['Label'].unique()) - set(LABELS)}"

        self.features = self.data.drop(columns=["Label"]).values
        self.labels = self.data["Label"].map(LABEL_TO_INDEX).values

        self.num_features = self.features.shape[1]
        self.num_classes = NUM_CLASSES

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)

        label = int(self.labels[idx])

        return features, label
