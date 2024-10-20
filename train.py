import polars as pl
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import wandb
from presnap.encoder.dataset import PreSnapEncoderDataset
from presnap.encoder.model import PreSnapGameConfig, PreSnapGameModelForSpread
from presnap.utils import load_vocab


class ScorePredictionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs['input_ids'], 
                        attention_mask=inputs['attention_mask'], 
                        labels=inputs['labels'],
                    )
        
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Flatten predictions and labels if they're 2D (e.g., [batch_size, 2] for home and away scores)
    predictions = predictions[1].reshape(-1)
    labels = labels.reshape(-1)
    
    ranges = {
        "prediction_range": {"min": predictions.min().item(), "max": predictions.max().item()},
        "label_range": {"min": labels.min().item(), "max": labels.max().item()},
    }
    
    # Calculate the mean squared error
    mse = mean_squared_error(labels, predictions)
    # Calculate the mean absolute error
    mae = mean_absolute_error(labels, predictions)
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    return {"mse": mse, "mae": mae, "rmse": rmse, **ranges}

    

def train():
    # Initialize wandb
    wandb.init(project="presnap-game-model")

    # Load and split the data
    data = pl.read_parquet("/root/presnap/presnap/data/training_data.parquet")
    train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)

    # Create the datasets
    train_dataset = PreSnapEncoderDataset(train_data)
    eval_dataset = PreSnapEncoderDataset(eval_data)

    # Create the model with updated config
    model_config = PreSnapGameConfig(
        input_size=len(train_dataset.input_features()),
        latent_dim=128,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=train_dataset.num_positions(),
        initializer_range=0.02,
        score_prediction_hidden_size=256,
    )
    model = PreSnapGameModelForSpread(model_config)
    model = model.to(torch.device("cuda"))

    # Log number of trainable parameters
    print({"trainable_parameters": f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"})

    # Define training arguments with updated learning rate schedule
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,  # Increased number of epochs
        per_device_train_batch_size=16,  # Increased batch size if memory allows
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,  # Accumulate gradients to increase effective batch size
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae",  # Choose the metric to use for saving the best model
        greater_is_better=False, 
        report_to="wandb",
        learning_rate=2e-3,  # Adjusted initial learning rate
        lr_scheduler_type="cosine",  # Changed to cosine schedule for better convergence
        max_grad_norm=1.0,  # Set max gradient norm for clipping
        remove_unused_columns=False,
        fp16=True,
    )

    # Create the trainer
    trainer = ScorePredictionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
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