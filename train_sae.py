from presnap.encoder.dataset import PreSnapEncoderDataset
from presnap.utils import load_vocab
from presnap.sae.model import SparseAutoencoder, SparseAutoencoderConfig, train_sparse_autoencoder
from presnap.sae.utils import build_encoder_config
import polars as pl
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
import torch

def train_sae():
    # Load and split the data
    data = pl.read_parquet("/Users/aw/projects/presnap/presnap/data/pregame_training.parquet")
    train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)
    vocab = load_vocab("/Users/aw/projects/presnap/presnap/data/pregame_vocab.json")

    # Create the datasets and dataloaders
    train_dataset = PreSnapEncoderDataset(train_data, vocab=vocab, device=torch.device("mps"))
    eval_dataset = PreSnapEncoderDataset(eval_data, vocab=vocab, device=torch.device("mps"))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=8)
    # Create the model
    model_config = SparseAutoencoderConfig(
        vocab=train_dataset.vocab,
        feature_index=train_dataset.feature_index,
        embedding_dim=64,
        hidden_dim=4096,
        latent_dim=2048,
    )
    model = SparseAutoencoder(model_config)

    # Train the model
    model = train_sparse_autoencoder(
        model, 
        train_dataloader, 
        eval_dataloader, 
        num_epochs=10,
        learning_rate=1e-3,
        weight_decay=0.01,
        seed=42
    )

    return model

if __name__ == "__main__":
    train_sae()