import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class PreSnapGameConfig(PretrainedConfig):
    model_type = "presnap_game_model"
    def __init__(
        self,
        categorical_features_vocab_sizes,
        numerical_feature_size,
        latent_dim=2048,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.categorical_features_vocab_sizes = categorical_features_vocab_sizes
        self.numerical_feature_size = numerical_feature_size
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob

        self.model_input_names = ["input_ids", "attention_mask", "numerical_features"]

class PreSnapGameModel(PreTrainedModel):
    config_class = PreSnapGameConfig
    base_model_prefix = "presnap_game_model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.categorical_embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_size, config.hidden_size)
                for feature, vocab_size in config.categorical_features_vocab_sizes.items()
            }
        )
        
        self.numerical_embeddings = nn.Linear(1, config.hidden_size)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation=config.hidden_act,
                dropout=config.hidden_dropout_prob,
                batch_first=True,
            ),
            num_layers=config.num_hidden_layers,
        )

        self.pooler = nn.Linear(config.hidden_size, config.latent_dim)

        self.init_weights()

    def forward(self, input_ids, attention_mask, numerical_features):
        batch_size = input_ids.shape[0]
        num_cat_features = input_ids.shape[1]
        num_num_features = numerical_features.shape[1]
        seq_len = num_cat_features + num_num_features

        embeddings = []
        for i, (feature, embedding) in enumerate(self.categorical_embeddings.items()):
            embeddings.append(embedding(input_ids[:, i]))
        
        for i in range(numerical_features.shape[1]):
            embeddings.append(self.numerical_embeddings(numerical_features[:, i].unsqueeze(-1)))

        embeddings = torch.stack(embeddings, dim=1)  # Shape: (batch_size, seq_len, hidden_size)

        # Ensure attention_mask is the correct shape (batch_size, seq_len)
        if attention_mask.shape != (batch_size, seq_len):
            attention_mask = attention_mask.view(batch_size, seq_len)

        # The transformer expects mask to be float with 0.0 for masked positions and 1.0 for unmasked
        attention_mask = attention_mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

        hidden_states = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        pooled_output = self.pooler(hidden_states.mean(dim=1))  # Use mean pooling

        # Calculate contrastive loss
        norm_pooled_output = F.normalize(pooled_output, dim=1)
        similarity_matrix = torch.matmul(norm_pooled_output, norm_pooled_output.t())
        
        mask = torch.eye(batch_size, dtype=torch.bool, device=pooled_output.device)
        positive_pairs = similarity_matrix[mask].view(batch_size, 1)
        negative_pairs = similarity_matrix[~mask].view(batch_size, -1)

        temperature = self.config.temperature
        logits = torch.cat([positive_pairs, negative_pairs], dim=1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=pooled_output.device)

        loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "pooled_output": pooled_output}