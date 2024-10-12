import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class SportsDataConfig(PretrainedConfig):
    model_type = "sports_data_encoder"
    def __init__(
        self,
        vocab_sizes,
        feature_to_embedding_index,
        num_features,
        hidden_size=768,
        num_hidden_layers=12,
        embedding_size=2048,
        mask_ratio=0.15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_sizes = vocab_sizes
        self.feature_to_embedding_index = feature_to_embedding_index
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embedding_size = embedding_size
        self.mask_ratio = mask_ratio
        self.model_input_names = ["input_ids", "attention_mask"]

class SportsDataEncoder(PreTrainedModel):
    config_class = SportsDataConfig
    base_model_prefix = "sports_data_encoder"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, config.hidden_size)
            for vocab_size in config.vocab_sizes
        ])

        self.encoder = nn.Sequential(
            *[nn.Linear(config.hidden_size, config.hidden_size),
              nn.ReLU(),
              nn.LayerNorm(config.hidden_size)] * config.num_hidden_layers
        )

        self.embedding_projector = nn.Linear(config.hidden_size, config.embedding_size)

        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size)
        )

        self.output_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, vocab_size)
            for vocab_size in config.vocab_sizes
        ])

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def apply_mask(self, x, mask_ratio=0.15):
        mask = torch.rand(x.shape[:2], device=x.device) < mask_ratio
        x[mask] = -200  # You might want to use a specific mask token instead of 0
        return x, mask

    def forward(self, input_ids, attention_mask=None):
        embedded_features = []
        for i in range(self.config.num_features):
            embedding_index = self.config.feature_to_embedding_index[i]
            embedded_features.append(self.embeddings[embedding_index](input_ids[:, i]))

        x = torch.stack(embedded_features, dim=1)

        # Apply masking
        x, mask = self.apply_mask(x, self.config.mask_ratio)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        encoded = self.encoder(x)

        # Mean pooling and projection to condensed embedding
        mean_pooled = torch.mean(encoded, dim=1)
        condensed_embedding = self.embedding_projector(mean_pooled)

        # Decode back to hidden size
        decoded = self.decoder(condensed_embedding)

        # Project to vocabulary sizes for each feature
        reconstructed_logits = []
        for i in range(self.config.num_features):
            embedding_index = self.config.feature_to_embedding_index[i]
            feature_logits = self.output_projections[embedding_index](decoded)
            reconstructed_logits.append(feature_logits)

        return {
            "embedding": condensed_embedding,
            "reconstructed_logits": reconstructed_logits,
            "mask": mask
        }

def masked_reconstruction_loss(reconstructed_logits, input_ids, mask, vocab_sizes):
    loss = 0
    for i, logits in enumerate(reconstructed_logits):
        feature_mask = mask[:, i]
        feature_input = input_ids[:, i][feature_mask]
        feature_logits = logits[feature_mask]
        if feature_input.nelement() > 0:  # Only compute loss if there are masked elements
            loss += nn.CrossEntropyLoss()(feature_logits, feature_input)
    return loss / len(reconstructed_logits)