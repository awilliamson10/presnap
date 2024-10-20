import polars as pl
import torch
from torch.utils.data import Dataset
from typing import Dict

class PreSnapEncoderDataset(Dataset):
    def __init__(self, data: pl.DataFrame):
        self.data = data
        self.normalize_numerical_features()

    def input_features(self):
        return [c for c in self.data.columns if c != 'homeSpreadResult']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data[idx]
        inputs = torch.tensor([row[col] for col in self.input_features()], dtype=torch.float32)
        labels = torch.tensor(row['homeSpreadResult'], dtype=torch.float32)
        # attention mask should be 1 for all non-padding tokens
        attention_mask = torch.ones(inputs.size(0), dtype=torch.float32)
        return {
            "input_ids": inputs.squeeze(-1),
            "attention_mask": attention_mask,
            "labels": labels,
        }