import torch
import torch.nn as nn
from transformers import PreTrainedModel, Trainer, TrainingArguments, LlamaConfig, LlamaForCausalLM, PretrainedConfig
from typing import List, Dict, Optional
import random
import numpy as np
import polars as pl
from tabformer.utils import process_data, build_vocab, make_dataset, SpecialTokens

class TabformerConfig(PretrainedConfig):
    def __init__(
        self,
        model_config: LlamaConfig,
        train_config: TrainingArguments,
        model_type: str = "tabular",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_config = model_config
        self.train_config = train_config


class Tabformer(PreTrainedModel):
    config_class = TabformerConfig
    base_model_prefix = "tabformer"

    def __init__(self, config):
        super().__init__(config)
        self.config = config.model_config
        self.train_config = config.train_config
        self.model = None

        self.columns: List[str] = []
        self.column_types: Dict[str, type] = {}
        self.processed_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.datetime_columns: List[str] = []
        self.vocab: Dict[str, dict] = {}

        self.tabular_max_length = None
        # Output length for generator model
        # including special tokens.
        self.tabular_max_length = None
        self.relational_max_length = None
        # Number of derived columns for the relational
        # and tabular data after performing the data transformation.
        # This will be used as record size validator in the
        # sampling stage.
        self.tabular_col_size = None
        self.relational_col_size = None

        # This stores the transformation
        # parameters for numeric columns.
        self.col_transform_data: Optional[Dict] = None

        # This is the col_transform_data
        # for the relational models's in_df.
        self.in_col_transform_data: Optional[Dict] = None

        self.col_idx_ids: Dict[int, list] = {}

        self.random_state = 1337

        self.numeric_precision = 2

        # Target column, when set, a copy of the column values will be
        # implicitly placed at the beginning of the dataframe.
        self.target_col = None

    def _extract_column_info(self, dataset: pl.DataFrame):
        self.columns = dataset.columns
        self.column_types = dataset.schema.to_python()
        
        self.numeric_columns = dataset.select(pl.selectors.numeric()).columns
        self.datetime_columns = dataset.select(pl.selectors.datetime()).columns

    def _generate_vocab(self, dataset: pl.DataFrame):
        return build_vocab(dataset, add_columns=False)

    def train_tabular(self, dataset: pl.DataFrame, device: str = "cpu"):
        self._extract_column_info(dataset)
        dataset, self.col_transform_data = process_data(
            dataset,
            numeric_precision=self.numeric_precision,
            col_transform_data=self.col_transform_data,
            target_col=self.target_col,
        )
        self.vocab = self._generate_vocab(dataset)
        self.tabular_col_size = dataset.shape[0]

        # NOTE: the index starts at zero, but should be adjusted
        # to account for the special tokens. For tabular data,
        # the index should start at 1.
        self.col_idx_ids = {
            ix: self.vocab["column_token_ids"][col]
            for ix, col in enumerate(self.processed_columns)
        }

        # Load the dataframe into a HuggingFace Dataset
        dataset = make_dataset(
            dataset, self.vocab
        )

        self.tabular_max_length = len(dataset[0]["input_ids"])
        self.dataset = dataset

        # Set up the config and the model
        self.config.bos_token_id = self.vocab["token2id"][SpecialTokens.BOS]
        self.config.eos_token_id = self.vocab["token2id"][SpecialTokens.EOS]
        self.config.vocab_size = len(self.vocab["id2token"])

        # Make sure that we have at least the number of
        # columns in the transformed data as positions.
        if self.config.max_position_embeddings < len(self.vocab["column_token_ids"]):
            self.config.max_position_embeddings = len(self.vocab["column_token_ids"]) + 128

        self.model = LlamaForCausalLM(self.config)
        device = torch.device(device)
        self.model.to(device)
    
        trainer = Trainer(
            model=self.model,
            args=self.train_config,
            train_dataset=self.dataset,
        )

        return trainer

    def train(self, dataset: pl.DataFrame, device: str = "cpu", resume_from_checkpoint: Optional[str] = None):
        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

            trainer = self._train_tabular(dataset, device=device)
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)