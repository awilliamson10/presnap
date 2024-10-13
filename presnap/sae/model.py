import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class SparseAutoencoderConfig(PretrainedConfig):
    model_type = "sparse-autoencoder"
    
    def __init__(
        self,
        vocab = None,
        feature_index = {},
        embedding_dim=64,
        hidden_dim=256,
        latent_dim=64,
        sparsity_param=0.05,
        sparsity_weight=0.1,
        l1_reg=0.0001,  # New parameter for L1 regularization
        l2_reg=0.0001,  # New parameter for L2 regularization
        dropout=0.1,
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
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout = dropout

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
            nn.ReLU()  # Changed from Sigmoid to ReLU
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

    def forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            inputs = inputs['inputs']
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
