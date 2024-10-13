import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PreTrainedModel, PretrainedConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

class SparseAutoencoderConfig(PretrainedConfig):
    model_type = "sparse-autoencoder"
    
    def __init__(
        self,
        vocab,
        feature_index,
        embedding_dim=64,
        hidden_dim=256,
        latent_dim=64,
        sparsity_param=0.05,
        sparsity_weight=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.feature_index = feature_index
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sparsity_param = sparsity_param
        self.sparsity_weight = sparsity_weight

class SparseAutoencoder(PreTrainedModel):
    config_class = SparseAutoencoderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(len(vocab), config.embedding_dim)
            for feature, vocab in config.vocab.items()
        })

        total_embedding_dim = len(config.feature_index) * config.embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(total_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, total_embedding_dim),
            nn.Sigmoid()
        )

        self.output_projections = nn.ModuleDict({
            feature: nn.Linear(config.embedding_dim, len(vocab))
            for feature, vocab in config.vocab.items()
        })

        self.init_weights()
        
    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(self, inputs):
        embedded = [self.embeddings[list(self.config.vocab.keys())[self.config.feature_index[i]]](input_tensor) 
                    for i, input_tensor in enumerate(inputs.t())]
        embedded = torch.cat(embedded, dim=1)
        
        encoded = self.encoder(embedded)
        decoded = self.decoder(encoded)
        
        decoded_embeddings = torch.split(decoded, self.config.embedding_dim, dim=1)
        
        outputs = [self.output_projections[list(self.config.vocab.keys())[self.config.feature_index[i]]](emb) 
                   for i, emb in enumerate(decoded_embeddings)]
        
        return encoded, outputs
    
    def kl_divergence(self, p, q):
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    
    def sparsity_penalty(self, encoded):
        avg_activations = torch.mean(encoded, dim=0)
        kl_div = self.kl_divergence(self.config.sparsity_param, avg_activations)
        return torch.sum(kl_div)
    
def train_sparse_autoencoder(model, train_dataloader, eval_dataloader=None, num_epochs=10, learning_rate=1e-3, weight_decay=0.01, seed=42):
    accelerator = Accelerator()
    set_seed(seed)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    if eval_dataloader:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    num_features = len(model.config.feature_index)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                encoded, outputs = model(batch)

                # Calculate loss for each feature and average over batch size
                reconstruction_loss = sum(
                    nn.functional.cross_entropy(output, input_tensor, reduction='sum')
                    for output, input_tensor in zip(outputs, batch.t())
                ) / (batch.size(0) * num_features)

                loss = reconstruction_loss 
                accelerator.backward(loss)
                optimizer.step()
            
            total_loss += loss.item()
            progress_bar.update(1)
            
            progress_bar.set_postfix({
                "loss": loss.item(), 
            })

        avg_loss = total_loss / len(train_dataloader)
        progress_bar.close()
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        if eval_dataloader:
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for batch in eval_dataloader:
                    encoded, outputs = model(batch)
                    reconstruction_loss = sum(
                        nn.functional.cross_entropy(output, input_tensor, reduction='sum')
                        for output, input_tensor in zip(outputs, batch.t())
                    ) / (batch.size(0) * num_features)
                    loss = reconstruction_loss
                    eval_loss += loss.item()
            
            avg_eval_loss = eval_loss / len(eval_dataloader)
            print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    
    model = accelerator.unwrap_model(model)
    return model