import polars as pl
import torch
from torch.utils.data import Dataset

class PreSnapEncoderDataset(Dataset):
    def __init__(self, data: pl.DataFrame, device: str):
        self.data = data
        self.device = device
        self.feature_names = [col for col in data.columns if col != 'drives']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].to_dicts()[0]
        inputs = torch.tensor([item[k] for k in self.feature_names], dtype=torch.long).to(self.device)
        
        return {
            "input_ids": inputs,
            "attention_mask": torch.ones_like(inputs, dtype=torch.long).to(self.device)
        }