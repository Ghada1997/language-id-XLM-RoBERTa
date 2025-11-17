
# src/language_id/metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def compute_metrics_light(eval_pred):
    preds, labels = eval_pred
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

def compute_metrics_full(eval_pred):
    preds, labels = eval_pred
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }

def per_class_report(labels: np.ndarray, preds: np.ndarray, id2label: Dict[int, str]) -> Dict:
    """Full per-class report as a dict (save to JSON; don't print 235 rows)."""
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    # Ensure labels are evaluated in id order
    target_names = [id2label[i] for i in sorted(id2label)]
    return classification_report(
        labels, preds, target_names=target_names, output_dict=True, zero_division=0
    )

def topk_per_class_f1(rep: Dict, k: int = 20, mode: str = "worst") -> List[Tuple[str, float, int]]:
    """
    Extract top-k summary from a classification_report dict.
    mode: 'worst' (lowest F1) or 'support' (highest support).
    Returns list of (label_name, f1, support).
    """
    rows = []
    for label, stats in rep.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        f1 = float(stats.get("f1-score", 0.0))
        support = int(stats.get("support", 0))
        rows.append((label, f1, support))
    if mode == "worst":
        rows.sort(key=lambda x: x[1])  # by f1 asc
    elif mode == "support":
        rows.sort(key=lambda x: x[2], reverse=True)  # by support desc
    return rows[:k]
