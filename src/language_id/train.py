# src/language_id/train.py
from __future__ import annotations
import os
from datasets import DatasetDict
from transformers import (
    AutoTokenizer, DataCollatorWithPadding,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from .utils import set_seed, save_json
from .metrics import compute_metrics_full
from .model import build_model

def run_final_training(
    tok_ds: DatasetDict,            # must contain 'train', 'validation', 'test'
    model_name: str,
    id2label: dict,
    label2id: dict,
    best_params: dict,
    output_dir: str,
    metrics_out: str,
    seed: int = 42,
    early_patience: int = 2,
):
    """Train final model with best params on full data and evaluate."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_out, exist_ok=True)

    set_seed(seed)

    # Tokenizer & collator
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Best params from Optuna
    lr   = float(best_params["learning_rate"])
    bsz  = int(best_params["per_device_train_batch_size"])
    ne   = int(best_params["num_train_epochs"])
    wd   = float(best_params["weight_decay"])
    wr   = float(best_params["warmup_ratio"])

    # Precision flags (reuse your GPU check)
    import torch
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    use_bf16 = has_cuda and any(k in gpu_name for k in ["A100", "L4"])
    fp16 = has_cuda and not use_bf16
    bf16 = use_bf16

    # Model
    model = build_model(model_name, len(id2label), id2label, label2id)

    # Training args
    targs = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        num_train_epochs=ne,
        weight_decay=wd,
        warmup_ratio=wr,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        logging_strategy="epoch",
        seed=seed,
        report_to=[],
        fp16=fp16,
        bf16=bf16,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_full,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_patience)],
    )

    trainer.train()
    val_metrics  = trainer.evaluate(tok_ds["validation"])
    test_metrics = trainer.evaluate(tok_ds["test"])

    # Save metrics
    save_json(val_metrics, os.path.join(metrics_out, "final_val_metrics.json"))
    save_json(test_metrics, os.path.join(metrics_out, "final_test_metrics.json"))

    return trainer, val_metrics, test_metrics
