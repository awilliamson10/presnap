import polars as pl
import torch
from torch.utils.data import Dataset


def build_feature_index(vocab, feature_names):
    # Define the feature to embedding index mapping
    feature_index = {}
    for i, feature in enumerate(feature_names):
        if feature not in ['drives']:
            if feature.startswith('home') and f"away{feature[4:]}" in feature_names or feature.startswith('away') and f"home{feature[4:]}" in feature_names:
                index_key = feature[4:]
            else:
                index_key = feature
            feature_index[i] = list(vocab.keys()).index(index_key)
    return feature_index

class PreSnapEncoderDataset(Dataset):
    def __init__(self, data: pl.DataFrame, vocab: dict, device: torch.device):
        self.data = data
        self.device = device
        self.feature_names = [col for col in data.columns if col != 'drives']
        self.vocab = vocab
        self.feature_index = build_feature_index(vocab, self.feature_names)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].to_dicts()[0]
        inputs = torch.tensor([item[k] for k in self.feature_names if k != 'drives'], dtype=torch.long).to(self.device)
        return {
            'inputs': inputs,
            'labels': inputs,
            'label_ids': inputs,
        }