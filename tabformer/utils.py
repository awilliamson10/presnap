import polars as pl
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, fields
from datasets import Dataset
import random
from collections import defaultdict

@dataclass(frozen=True)
class SpecialTokens:
    UNK: str = "[UNK]"
    SEP: str = "[SEP]"
    PAD: str = "[PAD]"
    CLS: str = "[CLS]"
    MASK: str = "[MASK]"
    BOS: str = "[BOS]"
    EOS: str = "[EOS]"

    @staticmethod
    def tokens():
        return [field.default for field in fields(SpecialTokens)]


def process_data(
    df: pl.DataFrame,
    numeric_precision: int = 1,
    col_transform_data: Dict = None,
    target_col: str = None,
) -> Tuple[pl.DataFrame, Dict]:
    df = df.clone()

    if target_col is not None:
        # Add the target column to the beginning of the dataframe.
        df = df.select([target_col] + [col for col in df.columns if col != target_col])

    numeric_columns = df.select(pl.selectors.float()).columns
    # we should perform transforms on the numeric columns and save those
    # transformations for later use.
    if col_transform_data is None:
        col_transform_data = {}

    for col in numeric_columns:
        if col not in col_transform_data:
            col_transform_data[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
            }

    # Normalize, round, scale the numeric columns, multiply by 10**precision, convert to int, handling nulls
    df = df.with_columns([
        (((df[col] - col_transform_data[col]["mean"]) / col_transform_data[col]["std"])
        .round(numeric_precision) * (10 ** numeric_precision))
        .cast(int)
        .alias(col)
        for col in numeric_columns
    ])

    return df, col_transform_data
    

def build_vocab(df: pl.DataFrame, special_tokens: List[str] = None, add_columns: bool = False, min_freq: int = 1):
    if special_tokens is None:
        special_tokens = SpecialTokens.tokens()

    id2token = {i: token for i, token in enumerate(special_tokens)}
    curr_id = len(id2token)
    column_token_ids = {}

    # Calculate frequency of each unique value
    value_counts = defaultdict(int)
    for col in df.columns:
        for val in df[col]:
            if val is not None:
                value_counts[val] += 1

    # Track tokens already added to ensure uniqueness
    added_tokens = set(id2token.values())

    for col in df.columns:
        unique_values = sorted(
            val for val in df[col].unique() if val is not None and value_counts[val] >= min_freq
        )
        for val in unique_values:
            if val not in added_tokens:
                id2token[curr_id] = val
                added_tokens.add(val)
                curr_id += 1
        column_token_ids[col] = list(range(curr_id - len(unique_values), curr_id))

    token2id = {v: k for k, v in id2token.items()}

    return dict(
        id2token=id2token,
        token2id=token2id,
        column_token_ids=column_token_ids,
    )

def get_token_id(
    token: str, vocab_token2id: Dict[str, int], mask_rate: float = 0
) -> int:
    token_id = vocab_token2id.get(token, vocab_token2id[SpecialTokens.UNK])
    if mask_rate > 0:
        token_id = (
            vocab_token2id[SpecialTokens.RMASK]
            if random.random() < mask_rate
            else token_id
        )

    return token_id

def get_input_ids(
    example,
    vocab: Dict,
    columns: List,
    mask_rate: float = 0,
    return_label_ids: Optional[bool] = True,
    affix_bos: Optional[bool] = True,
    affix_eos: Optional[bool] = True,
) -> Dict:
    input_ids: List[int] = []

    if affix_bos:
        input_ids.append(vocab["token2id"][SpecialTokens.BOS])

    for k in columns:
        input_ids.append(get_token_id(example[k], vocab["token2id"], mask_rate))

    if affix_eos:
        input_ids.append(vocab["token2id"][SpecialTokens.EOS])
    data = dict(input_ids=input_ids)

    if return_label_ids:
        data["label_ids"] = input_ids

    return data

def make_dataset(
    df: pl.DataFrame,
    vocab: Dict,
    mask_rate: float = 0,
    affix_eos: bool = True,
) -> Dataset:
    # Load the dataframe into a HuggingFace Dataset
    training_dataset = Dataset.from_polars(df)

    return training_dataset.map(
        lambda example: get_input_ids(
            example,
            vocab,
            df.columns,
            mask_rate=mask_rate,
            affix_eos=affix_eos,
        ),
        remove_columns=training_dataset.column_names,
    )