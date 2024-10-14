import polars as pl
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback

import wandb
from presnap.encoder.dataset import PreSnapEncoderDataset
from presnap.encoder.model import PreSnapGameConfig, PreSnapGameModel
from presnap.utils import load_vocab
    

def train():
    # Initialize wandb
    wandb.init(project="presnap-game-model")

    # Load and split the data
    data = pl.read_parquet("/Users/aw/projects/presnap/presnap/data/presnap.parquet")
    train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)
    token_map = load_vocab("/Users/aw/projects/presnap/presnap/data/tokens.json")

    # Create the datasets
    train_dataset = PreSnapEncoderDataset(train_data, token_map=token_map)
    eval_dataset = PreSnapEncoderDataset(eval_data, token_map=token_map)

    # Create the model with updated config
    model_config = PreSnapGameConfig(
        categorical_features_vocab_sizes=train_dataset.categorical_features_vocab_sizes(),
        numerical_feature_size=len(train_dataset.numerical_features()),
        latent_dim=2048,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
    )
    model = PreSnapGameModel(model_config)
    model = model.to(torch.device("mps"))

    # Log number of trainable parameters
    print({"trainable_parameters": f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"})

    # Define training arguments with updated learning rate schedule
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,  # Increased number of epochs
        per_device_train_batch_size=8,  # Increased batch size if memory allows
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,  # Accumulate gradients to increase effective batch size
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        learning_rate=2e-3,  # Adjusted initial learning rate
        lr_scheduler_type="cosine",  # Changed to cosine schedule for better convergence
        max_grad_norm=1.0,  # Set max gradient norm for clipping
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbCallback()]  # Add WandbCallback
    )

    # Train the model
    trainer.train()

    # Save the best model
    trainer.save_model("./best_model")

    # Close wandb run
    wandb.finish()

    return model

if __name__ == "__main__":
    train()