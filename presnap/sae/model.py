import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

class SparseAutoencoderConfig(PretrainedConfig):
    model_type = "sparse-autoencoder"
    
    def __init__(
        self,
        vocab=None,
        feature_index={},
        embedding_dim=64,
        hidden_dim=256,
        latent_dim=64,
        encoder_layers=2,
        decoder_layers=2,
        nhead=8,
        dim_feedforward=1024,
        sparsity_param=0.05,
        sparsity_weight=0.1,
        l1_reg=0.0001,
        l2_reg=0.0001,
        dropout=0.1,
        dead_neuron_threshold=100,
        ghost_grads_weight=0.01,
        device="cpu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.feature_index = feature_index
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.sparsity_param = sparsity_param
        self.sparsity_weight = sparsity_weight
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.dead_neuron_threshold = dead_neuron_threshold
        self.ghost_grads_weight = ghost_grads_weight
        self.device = device

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

        # Input projection to match transformer input dimension
        self.input_projection = nn.Linear(total_embedding_dim, config.hidden_dim)

        # Build encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        
        # Latent layer
        self.latent_encoder = nn.Linear(config.hidden_dim, config.latent_dim)
        self.latent_decoder = nn.Linear(config.latent_dim, config.hidden_dim)
        self.latent_activation = nn.ReLU()

        # Build decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, total_embedding_dim),
            nn.Sigmoid()
        )

        self.output_projections = nn.ModuleDict({
            feature: nn.Linear(config.embedding_dim, len(vocab))
            for feature, vocab in config.vocab.items()
        })

        self.init_weights()
        
        # Initialize dead neuron tracking
        self.register_buffer('neuron_activations', torch.zeros(config.latent_dim))
        self.register_buffer('steps_since_activation', torch.zeros(config.latent_dim))

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
        
        # Project input to hidden dimension
        projected = self.input_projection(embedded)
        
        # Encoder forward pass
        encoded = self.encoder(projected)
        
        # Latent space
        latent = self.latent_encoder(encoded)
        latent = self.latent_activation(latent)
        
        # Update neuron activation tracking
        self.update_neuron_activations(latent)
        
        # Project latent back to hidden dimension for decoder
        latent_projected = self.latent_decoder(latent)
        
        # Decoder forward pass (using encoded as memory)
        decoded = self.decoder(latent_projected, encoded)
        
        # Project back to original dimension
        output = self.output_projection(decoded)
        
        decoded_embeddings = torch.split(output, self.config.embedding_dim, dim=1)
        
        outputs = [self.output_projections[list(self.config.vocab.keys())[self.config.feature_index[i]]](emb) 
                   for i, emb in enumerate(decoded_embeddings)]
        
        return latent, outputs
    
    def update_neuron_activations(self, encoded):
        with torch.no_grad():
            self.steps_since_activation += (encoded == 0).float().sum(dim=0)
            self.steps_since_activation[encoded.sum(dim=0) > 0] = 0
            self.neuron_activations += encoded.sum(dim=0)

    def reinitialize_dead_neurons(self):
        dead_neurons = (self.steps_since_activation >= self.config.dead_neuron_threshold).nonzero().squeeze()
        if dead_neurons.numel() / self.config.latent_dim > 0.1:
            print(f"Reinitializing {dead_neurons.numel()} dead neurons")
            with torch.no_grad():
                # Reinitialize weights for dead neurons in the latent encoder
                fan_in = self.latent_encoder.weight.size(1)
                std = (2.0 / fan_in) ** 0.5
                self.latent_encoder.weight[dead_neurons] = torch.randn_like(self.latent_encoder.weight[dead_neurons]) * std
                self.latent_encoder.bias[dead_neurons] = torch.zeros_like(self.latent_encoder.bias[dead_neurons])
                
                # Reinitialize corresponding weights in the latent decoder
                fan_out = self.latent_decoder.weight.size(0)
                std = (2.0 / fan_out) ** 0.5
                self.latent_decoder.weight[:, dead_neurons] = torch.randn_like(self.latent_decoder.weight[:, dead_neurons]) * std

            # Reset activation counters for reinitialized neurons
            self.neuron_activations[dead_neurons] = 0
            # self.steps_since_activation[dead_neurons] = 0

    def ghost_gradient_update(self):
        dead_neurons = (self.steps_since_activation >= self.config.dead_neuron_threshold).nonzero().squeeze()
        if dead_neurons.numel() / self.config.latent_dim > 0.1:
            # Apply small random updates to dead neurons
            with torch.no_grad():
                ghost_grads = torch.randn_like(self.latent_encoder.weight[dead_neurons]) * self.config.ghost_grads_weight
                self.latent_encoder.weight[dead_neurons] += ghost_grads
                self.latent_decoder.weight[:, dead_neurons] += ghost_grads.t()

    def kl_divergence(self, p, q):
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    
    def sparsity_penalty(self, encoded):
        avg_activations = torch.mean(encoded, dim=0)
        kl_div = self.kl_divergence(self.config.sparsity_param, avg_activations)
        return torch.sum(kl_div)