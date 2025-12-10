from torch.utils.data import Dataset
from typing import Tuple, Dict
import pandas as pd
import torch
from pathlib import Path

class ExplainDataSet(Dataset):
    """
    Dataset for Network Flow Explanation.
    Assumes a single CSV containing flow features, labels, and an 'explanation' column.
    """

    def __init__(self, data_path: str) -> None:
        # Load the single merged CSV
        self.data = pd.read_csv(data_path)
        
        # Ensure the explanation column exists
        if "explanation" not in self.data.columns:
            raise ValueError(f"The file {data_path} must contain an 'explanation' column.")
        
        # --- Prepare Numeric Features for MLP ---
        # We must exclude the label strings and the explanation text
        cols_to_drop = ["Label", "Fine Label", "explanation"]
        
        # Select only columns that are NOT in the drop list
        feature_df = self.data.drop(columns=cols_to_drop, errors='ignore')
        
        # Convert to numpy array for efficient indexing
        self.numeric_features = feature_df.values
        self.feature_names = feature_df.columns.tolist()
        
        # --- Prepare Explanations ---
        self.explanations = self.data["explanation"].values

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        # 1. Numeric Features (Normalized/Cleaned) -> For MLP
        features = torch.tensor(self.numeric_features[idx], dtype=torch.float32)

        # 2. Raw Text Description -> For LLM Prompt
        # We reconstruct the 'plain english' part dynamically from the row data
        # We use iloc to access the row by integer position
        row = self.data.iloc[idx]
        prompt_text = self._construct_text_prompt(row)

        # 3. Ground Truth Explanation -> For Loss Calculation
        explanation = str(self.explanations[idx])

        return {
            "mlp_features": features,
            "text_prompt": prompt_text,
            "ground_truth": explanation
        }

    def _construct_text_prompt(self, row: pd.Series) -> str:
        """Constructs the plain English description of flow features."""
        # Builds a string representation of the flow features
        return (
            f"Dst Port: {row.get('Dst Port', 'N/A')}, Protocol: {row.get('Protocol', 'N/A')}, Flow Duration: {row.get('Flow Duration', 'N/A')}\n"
            f"Tot Fwd Pkts: {row.get('Tot Fwd Pkts', 'N/A')}, Tot Bwd Pkts: {row.get('Tot Bwd Pkts', 'N/A')}\n"
            f"Fwd Pkt Len Max: {row.get('Fwd Pkt Len Max', 'N/A')}, Bwd Pkt Len Max: {row.get('Bwd Pkt Len Max', 'N/A')}\n"
            f"Flow Byts/s: {row.get('Flow Byts/s', 'N/A')}, Flow Pkts/s: {row.get('Flow Pkts/s', 'N/A')}\n"
            f"SYN Flag: {row.get('SYN Flag Cnt', 'N/A')}, ACK Flag: {row.get('ACK Flag Cnt', 'N/A')}\n"
        )