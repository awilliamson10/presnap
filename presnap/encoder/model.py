import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

class PreSnapGameConfig(PretrainedConfig):
    model_type = "presnap_game_model"
    def __init__(
        self,
        input_size = 128,
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
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.score_prediction_hidden_size = score_prediction_hidden_size

        self.model_input_names = ["inputs", "attention_mask", "labels"]

class ResidualTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        out = super().forward(src, src_mask, src_key_padding_mask, **kwargs)
        return out + src

class MultiHeadSelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        weights = F.softmax(self.fc(attn_output).squeeze(-1), dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class PreSnapGameModel(PreTrainedModel):
    config_class = PreSnapGameConfig
    base_model_prefix = "presnap_game_model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.input_layer = nn.Linear(config.input_size, config.hidden_size)

        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = nn.TransformerEncoder(
            ResidualTransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation=config.hidden_act,
                dropout=config.hidden_dropout_prob,
                batch_first=True,
            ),
            num_layers=config.num_hidden_layers,
        )

        self.pooler = MultiHeadSelfAttentionPooling(config.hidden_size, config.num_attention_heads)
        self.projection = nn.Linear(config.hidden_size, config.latent_dim)

        self.init_weights()

    def forward(self, input_ids, attention_mask):
        x = self.input_layer(input_ids)
        x = self.layernorm(x)
        x = self.dropout(x)

        x = self.encoder(x, src_key_padding_mask=(1 - attention_mask).bool())
        pooled_output = self.pooler(x)
        pooled_output = self.projection(pooled_output)

        return {"pooled_output": pooled_output}


class PreSnapGameModelForSpread(PreSnapGameModel):
    def __init__(self, config):
        super().__init__(config)
        self.spread_predictor = nn.Linear(config.latent_dim, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, numerical_features, labels=None):
        outputs = super().forward(input_ids, attention_mask, numerical_features)
        pooled_output = outputs["pooled_output"]
        spread_predictions = self.spread_predictor(pooled_output)
        spread_predictions = spread_predictions.unsqueeze(1)

        if labels is not None:
            loss = self.spread_prediction_loss(spread_predictions, labels)
            return {"loss": loss, "pooled_output": pooled_output, "spread_predictions": spread_predictions}
        else:
            return {"pooled_output": pooled_output, "spread_predictions": spread_predictions}

    def spread_prediction_loss(self, predictions, targets):
        mse_loss = F.mse_loss(predictions.view(-1), targets.view(-1))
        direction_loss = F.binary_cross_entropy_with_logits(
            predictions.view(-1), (targets.view(-1) > 0).float()
        )
        return mse_loss + 0.1 * direction_loss