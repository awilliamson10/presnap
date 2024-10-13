import polars as pl
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback

import wandb
from presnap.encoder.dataset import PreSnapEncoderDataset
from presnap.sae.model import SparseAutoencoder, SparseAutoencoderConfig
from presnap.utils import load_vocab

class SparseAutoencoderTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.max_grad_norm)

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        if isinstance(inputs, dict):
            inputs = inputs['inputs']
        encoded, outputs = model(inputs)
        
        # Compute reconstruction loss, ignoring -100 values
        reconstruction_loss = 0
        num_valid_elements = 0
        for output, input_tensor in zip(outputs, inputs.t()):
            # Create a mask for valid (non -100) entries
            mask = (input_tensor != -100)
            if mask.sum() > 0:  # Only compute loss if there are valid entries
                reconstruction_loss += torch.nn.functional.cross_entropy(
                    output[mask], 
                    input_tensor[mask], 
                    reduction='sum'
                )
                num_valid_elements += mask.sum().item()
        
        # Normalize the reconstruction loss
        if num_valid_elements > 0:
            reconstruction_loss /= num_valid_elements
        else:
            reconstruction_loss = torch.tensor(0.0, device=model.device)

        # L1 regularization
        l1_reg = torch.tensor(0., device=model.device)
        if model.config.l1_reg > 0:
            for param in model.parameters():
                l1_reg += torch.norm(param, p=1)
            l1_reg *= model.config.l1_reg

        # L2 regularization
        l2_reg = torch.tensor(0., device=model.device)
        if model.config.l2_reg > 0:
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            l2_reg *= model.config.l2_reg

        reconstruction_weight = 1.0
        loss = (reconstruction_weight * reconstruction_loss)

        sparsity_loss = torch.tensor(0., device=model.device)
        if model.config.sparsity_weight > 0:
            sparsity_loss = model.config.sparsity_weight * (
                model.config.sparsity_param * torch.log(model.config.sparsity_param / encoded.mean(dim=1)).sum() +
                (1 - model.config.sparsity_param) * torch.log((1 - model.config.sparsity_param) / (1 - encoded.mean(dim=1))).sum()
            )
            loss += sparsity_loss

        if model.config.l1_reg > 0:
            loss += l1_reg

        if model.config.l2_reg > 0:
            loss += l2_reg

        with torch.no_grad():
            l0_norm = (encoded != 0).float().sum(dim=1).mean().item()
            dead_neurons = (encoded == 0).all(dim=0).sum().item()

        self.log({
            "reconstruction_loss": reconstruction_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "l1_reg": l1_reg.item(),
            "l2_reg": l2_reg.item(),
            "total_loss": loss.item(),
            "avg_activation": encoded.mean().item(),
            "valid_elements_ratio": num_valid_elements / inputs.numel(),
            "l0_norm": l0_norm,
            "dead_neurons": dead_neurons,
            "dead_neurons_ratio": dead_neurons / encoded.size(1),
        })
        
        return (loss, outputs) if return_outputs else loss
    

def train_sae():
    # Initialize wandb
    wandb.init(project="sparse-autoencoder", name="sae-experiment")

    # Load and split the data
    data = pl.read_parquet("/Users/aw/projects/presnap/presnap/data/pregame_training.parquet")
    train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)
    vocab = load_vocab("/Users/aw/projects/presnap/presnap/data/pregame_vocab.json")

    # Create the datasets
    train_dataset = PreSnapEncoderDataset(train_data, vocab=vocab, device=torch.device("mps"))
    eval_dataset = PreSnapEncoderDataset(eval_data, vocab=vocab, device=torch.device("mps"))

    # Create the model with updated config
    model_config = SparseAutoencoderConfig(
        vocab=train_dataset.vocab,
        feature_index=train_dataset.feature_index,
        embedding_dim=64,
        hidden_dim=4096,
        latent_dim=2048,
        sparsity_param=0.05,
        sparsity_weight=0,  # Reduced further to balance with increased reconstruction focus
        l1_reg=0,  # L1 regularization strength
        l2_reg=0,  # L2 regularization strength
    )
    model = SparseAutoencoder(model_config)

    # Define training arguments with updated learning rate schedule
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,  # Increased number of epochs
        per_device_train_batch_size=16,  # Increased batch size if memory allows
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        learning_rate=2e-3,  # Adjusted initial learning rate
        lr_scheduler_type="cosine",  # Changed to cosine schedule for better convergence
        max_grad_norm=1.0,  # Set max gradient norm for clipping
    )

    # Create the trainer
    trainer = SparseAutoencoderTrainer(
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

    # Log final metrics
    wandb.log({
        "final_reconstruction_loss": trainer.state.log_history[-1]["eval_reconstruction_loss"],
        "final_l0_norm": trainer.state.log_history[-1]["eval_l0_norm"],
        "final_dead_neurons": trainer.state.log_history[-1]["eval_dead_neurons"],
        "final_dead_neurons_ratio": trainer.state.log_history[-1]["eval_dead_neurons_ratio"],
        "final_valid_elements_ratio": trainer.state.log_history[-1]["eval_valid_elements_ratio"],
    })

    # Close wandb run
    wandb.finish()

    return model

if __name__ == "__main__":
    train_sae()