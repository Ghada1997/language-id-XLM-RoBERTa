from datasets import load_dataset, DatasetDict, ClassLabel
from collections import defaultdict
import random


def load_wili():
    """
    Loads the full WiLI-2018 dataset as provided:
    {train (235k), test (235k)} without extra splitting.
    Returns (DatasetDict, text_col, label_col).
    """
    ds = load_dataset("MartinThoma/wili_2018")

    feats = ds["train"].features

    # Detect label column (ClassLabel) and text column (string)
    label_col = next((k for k,v in feats.items() if isinstance(v, ClassLabel)), "label")
    text_col  = next((k for k,v in feats.items() if getattr(v, "dtype", None) == "string"), "sentence")

    return ds, text_col, label_col

def make_train_val(ds, label_col="label", val_size=0.1, seed=42):
    """
    Split a training Dataset into train/validation (stratified).
    Returns a DatasetDict with keys 'train' and 'validation'. 90% for train and 10% validation in this project
    """
    split = ds['train'].train_test_split(test_size=val_size, seed=seed, stratify_by_column=label_col)
    return DatasetDict(train=split["train"], validation=split["test"], test=ds['test'])



def stratified_subset(split, label_col="label", size=0.2, seed=42):

    """Take approximately 'subset' of the 'split' with stratification by label.
    Returns the *subset* (size ~= frac * len(split)). Here it is 20%
    Note: train_test_split(test_size=frac) puts the subset in ['test']."""

    subsets = split.train_test_split(test_size=size, seed=seed, stratify_by_column=label_col)
    return subsets["test"] # returning the subset which represents the test here

def to_xy(split, text_col, label_col):
    """
    Convenience helper for scikit-learn baselines:
    returns (X, y) where X = list[str], y = list[str or int].
    If label_col is ClassLabel, y will be string labels (names).
    """
    feats = split.features[label_col]
    # If it's a ClassLabel, convert ints to string names; otherwise keep raw values
    if isinstance(feats, ClassLabel):
        y = [feats.int2str(i) for i in split[label_col]]
    else:
        y = split[label_col]
    X = split[text_col]
    return X, y

