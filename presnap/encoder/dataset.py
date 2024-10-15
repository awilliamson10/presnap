import polars as pl
import torch
from torch.utils.data import Dataset
from typing import Dict

def augment_numerical_features(numerical_features, noise_scale=0.1):
    noise = torch.randn_like(numerical_features) * noise_scale
    return numerical_features + noise

class PreSnapEncoderDataset(Dataset):
    def __init__(self, data: pl.DataFrame, token_map: dict):
        self.data = data
        self.token_map = token_map

    def categorical_features_vocab_sizes(self):
        return {col: len(self.token_map[col]) for col in self.categorical_features()}

    def categorical_features(self):
        return list(self.token_map.keys())
    
    def numerical_features(self):
        cols = self.data.columns
        return [col for col in cols if col not in self.token_map.keys()]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data[idx]
        input_ids = torch.tensor([row[col] for col in self.categorical_features()], dtype=torch.long)
        numerical_features = torch.tensor([row[col] for col in self.numerical_features()], dtype=torch.float32)

        cat_attn_mask = torch.tensor([
            val != self.token_map[col]['[UNK]']
            for col, val in zip(self.categorical_features(), input_ids)
        ], dtype=torch.bool)
        num_attn_mask = torch.tensor([val != -100 for val in numerical_features], dtype=torch.bool)
        attention_mask = torch.cat((cat_attn_mask, num_attn_mask))

        augmented_numerical_features = augment_numerical_features(numerical_features)

        return {
            "input_ids": input_ids.squeeze(-1),
            "numerical_features": numerical_features.squeeze(-1),  # Add batch dimension
            "attention_mask": attention_mask.unsqueeze(0),  # Add batch dimension
            "augmented_numerical_features": augmented_numerical_features.squeeze(-1),  # Add batch dimension
        }