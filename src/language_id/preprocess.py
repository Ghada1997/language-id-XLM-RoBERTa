from __future__ import annotations
from typing import Dict, Tuple, Optional

import pandas as pd
from datasets import Dataset, DatasetDict

# --- Minimal text cleaning ---

def clean_text(s: str) -> str:
    """Trim leading/trailing whitespace. WiLI doesn't need more."""
    return s.strip() if isinstance(s, str) else s


def apply_cleaning_ds(ds: DatasetDict, text_col: str):
    """Apply clean_text to the given text column in a DatasetDict."""
    return ds.map(lambda x: {text_col: clean_text(x[text_col])})

def tokenize_ds(ds, tokenizer, text_col: str, max_length: int):
    def _tok(batch):
        return tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    return ds.map(_tok, batched=True)
