import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

class PreSnapGameConfig(PretrainedConfig):
    model_type = "presnap_game_model"
    def __init__(
        self,
        categorical_features_vocab_sizes = {},
        numerical_feature_size = 0,
        latent_dim=2048,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        max_position_embeddings=256,
        score_prediction_hidden_size=256,
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
        self.max_position_embeddings = max_position_embeddings
        self.score_prediction_hidden_size = score_prediction_hidden_size

        self.model_input_names = ["input_ids", "attention_mask", "numerical_features", "labels"]

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

        # Positional embeddings
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Feature type embeddings
        self.feature_type_embedding = nn.Embedding(2, config.hidden_size)

        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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

        # Add positional embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=embeddings.device)
        embeddings += self.positional_embedding(position_ids).unsqueeze(0)

        # Add feature type embeddings
        feature_type_ids = torch.cat([
            torch.zeros(num_cat_features, dtype=torch.long, device=embeddings.device),
            torch.ones(num_num_features, dtype=torch.long, device=embeddings.device)
        ])
        embeddings += self.feature_type_embedding(feature_type_ids).unsqueeze(0)

        # Apply layer normalization and dropout
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        # Ensure attention_mask is the correct shape (batch_size, seq_len)
        if attention_mask.shape != (batch_size, seq_len):
            attention_mask = attention_mask.view(batch_size, seq_len)

        # The transformer expects mask to be float with 0.0 for masked positions and 1.0 for unmasked
        attention_mask = attention_mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

        hidden_states = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        pooled_output = self.pooler(hidden_states.mean(dim=1))

        return {"pooled_output": pooled_output}


class PreSnapGameModelForScore(PreSnapGameModel):
    def __init__(self, config):
        super().__init__(config)
        self.score_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.score_prediction_hidden_size),
            nn.ReLU(),
            nn.Linear(config.score_prediction_hidden_size, config.score_prediction_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.score_prediction_hidden_size // 2, 2)  # 2 outputs for home and away scores
        )

        self.init_weights()

    def forward(self, input_ids, attention_mask, numerical_features):
        outputs = super().forward(input_ids, attention_mask, numerical_features)
        pooled_output = outputs["pooled_output"]
        score_predictions = self.score_predictor(pooled_output)
        # Reshape to match the target shape
        score_predictions = score_predictions.unsqueeze(1)  # Shape: (batch_size, 1, 2)
        return {"pooled_output": pooled_output, "score_predictions": score_predictions}

    def score_prediction_loss(self, predictions, targets):
        return F.mse_loss(predictions, targets)
    
    def huber_loss(self, predictions, targets, delta=1.0):
        return F.smooth_l1_loss(predictions, targets, beta=delta)

    def log_cosh_loss(self, predictions, targets):
        def log_cosh(x):
            return x + torch.log(1 + torch.exp(-2 * x)) - torch.log(torch.tensor(2.0))
        return torch.mean(log_cosh(predictions - targets))

    def custom_score_loss(self, predictions, targets, mse_weight=0.7, mae_weight=0.3):
        mse = F.mse_loss(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        return mse_weight * mse + mae_weight * mae