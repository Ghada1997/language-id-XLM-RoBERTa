# tune.py
from __future__ import annotations
import os, json
import optuna
from functools import partial
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
)
from .metrics import compute_metrics_light  # your light metrics

def _build_model(model_name: str, num_labels: int, id2label: dict, label2id: dict):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

def _objective(
    trial: optuna.trial.Trial,
    tok_ds,                   # DatasetDict with 'train' and 'validation'
    model_name: str,
    id2label: dict,
    label2id: dict,
    seed: int,
    metric_for_best_model: str,
    trial_root_dir: str,      # <- absolute Drive path where trials should be written
):
    # Per-trial directory in your Drive
    trial_dir = os.path.join(trial_root_dir, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    # --- search space ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)

    # --- build model & tokenizer ---
    model = _build_model(model_name, len(id2label), id2label, label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # --- training args for tuning (NO checkpoint saving) ---
    args = TrainingArguments(
        output_dir=trial_dir,                # -> goes to Drive (your provided root)
        logging_dir=os.path.join(trial_dir, "logs"),
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,

        eval_strategy="epoch",
        save_strategy="no",                 # <- do not write checkpoints during tuning
        load_best_model_at_end=False,       # <- no checkpoints to load
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,

        logging_strategy="epoch",
        seed=seed,
        report_to=[],                       # no TB/W&B by default
        gradient_accumulation_steps=1, 
         # precision
        fp16=False,   # use True on T4/V100; set False if CPU-only
        bf16=True,  # set True (and fp16=False) on A100/L4. Here we are already using A100
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_light,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    # Return the metric Optuna should maximize
    return float(eval_metrics.get(f"eval_{metric_for_best_model}", 0.0))

def run_study(
    tok_ds,                       # subset: DatasetDict(train, validation)
    model_name: str,
    id2label: dict,
    label2id: dict,
    n_trials: int = 15,
    study_name: str = "xlmr_opt",
    metrics_out_dir: str = "./outputs/metrics",   # where the JSON summary goes
    trials_root_dir: str | None = None,           # absolute Drive dir for trial folders
    seed: int = 42,
    pruner: optuna.pruners.BasePruner | None = None,
):
    os.makedirs(metrics_out_dir, exist_ok=True)
    if trials_root_dir is None:
        # fallback (but you should pass a Drive path from the notebook)
        trials_root_dir = os.path.abspath("tmp_optuna_trials")
    os.makedirs(trials_root_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler, pruner=pruner)

    objective = partial(
        _objective,
        tok_ds=tok_ds,
        model_name=model_name,
        id2label=id2label,
        label2id=label2id,
        seed=seed,
        metric_for_best_model="macro_f1",
        trial_root_dir=trials_root_dir,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {
        "study": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "model_name": model_name,
    }
    out_path = os.path.join(metrics_out_dir, f"{study_name}_best_params.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best, out_path
