import os
from dataclasses import dataclass
from typing import Optional

import polars as pl
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

from presnap.encoder.builder import build_encoder_model
from presnap.encoder.dataset import PreSnapEncoderDataset
from presnap.encoder.model import masked_reconstruction_loss
from presnap.utils import load_vocab


@dataclass
class PreSnapTrainingConfig:
    train_encoder: bool = True
    preprocessed_data_path: str = "/Users/aw/projects/presnap/presnap/data/pregame_training.parquet"
    encoder_vocab_path: str = "/Users/aw/projects/presnap/presnap/data/pregame_vocab.json"
    output_dir: str = "./output/encoder"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 1
    save_steps: int = 10_000
    save_total_limit: int = 2
    evaluation_strategy: str = "steps"
    eval_steps: int = 1000
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    seed: int = 42
    fp16: bool = False
    device: str = "mps"
    test_size: float = 0.1
    model_name: str = "presnap_encoder"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

def setup_wandb(config):
    if config.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, name=config.wandb_name)
        print(f"Weights & Biases initialized. Project: {config.wandb_project}, Run name: {config.wandb_name}")

def load_and_split_data(config):
    data = pl.read_parquet(config.preprocessed_data_path)
    train_data, eval_data = train_test_split(data, test_size=config.test_size, random_state=config.seed)
    return train_data, eval_data

def create_datasets(train_data, eval_data, device):
    train_dataset = PreSnapEncoderDataset(train_data, device)
    eval_dataset = PreSnapEncoderDataset(eval_data, device)
    return train_dataset, eval_dataset

def build_model(encoder_vocab, feature_names):
    return build_encoder_model(encoder_vocab, feature_names)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        reconstructed_logits = outputs["reconstructed_logits"]
        mask = outputs["mask"]
        loss = masked_reconstruction_loss(reconstructed_logits, inputs["input_ids"], mask, model.config.vocab_sizes)

        return (loss, outputs) if return_outputs else loss

def setup_trainer(config, model, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=config.fp16,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else None,
        max_grad_norm=1.0,
    )

    return CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

def main():
    print("Starting PreSnap Encoder Training Script")
    config = PreSnapTrainingConfig()
    print(f"Configuration loaded: \n{config}")

    pl.set_random_seed(config.seed)
    torch.manual_seed(config.seed)

    setup_wandb(config)

    train_data, eval_data = load_and_split_data(config)
    print(f"Train data shape: {train_data.shape}, Eval data shape: {eval_data.shape}")

    train_dataset, eval_dataset = create_datasets(train_data, eval_data, config.device)
    print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")

    encoder_vocab = load_vocab(config.encoder_vocab_path)
    print(f"Vocabulary loaded. Number of keys: {len(encoder_vocab)}")

    if config.train_encoder:
        encoder = build_model(encoder_vocab, train_dataset.feature_names)
        print(f"Encoder model built. Number of parameters: {sum(p.numel() for p in encoder.parameters())}")

        trainer = setup_trainer(config, encoder, train_dataset, eval_dataset)
        print("Trainer initialized")

        print("Starting training")
        try:
            trainer.train()
            print("Training completed")

            best_model_path = os.path.join(config.output_dir, f"{config.model_name}_best")
            trainer.save_model(best_model_path)
            print(f"Best model saved to {best_model_path}")
        except Exception as e:
            print(f"Error during training: {str(e)}")

        if config.use_wandb:
            print("Finishing Weights & Biases run")
            wandb.finish()

    print("PreSnap Encoder Training Script completed")

if __name__ == "__main__":
    main()