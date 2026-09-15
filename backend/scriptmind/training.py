"""QLoRA supervised fine-tuning for ScriptMind."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import atomic_write_json, file_sha256, stable_hash
from .paths import ScriptMindPaths


MODEL_IDS = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
VALID_ABLATIONS = {"multitask", "detection_only", "no_next_utterance"}


def _experiment_result_class():
    try:
        from machine_learning.result import ExperimentResult
    except ImportError:
        from backend.machine_learning.result import ExperimentResult
    return ExperimentResult


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _target_for_ablation(target: dict[str, Any], ablation: str) -> dict[str, Any]:
    if ablation == "detection_only":
        return {"label": target["label"]}
    value = dict(target)
    if ablation == "no_next_utterance":
        value["next_utterance"] = None
    return value


class EncodedChatDataset:
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int, ablation: str):
        self.rows: list[dict[str, list[int]]] = []
        for row in frame.itertuples(index=False):
            target = _target_for_ablation(row.target, ablation)
            prompt_messages = [
                {"role": "system", "content": "You are Horus ScriptMind. Return one defensive scam-analysis JSON object."},
                {"role": "user", "content": row.context},
            ]
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            answer = json.dumps(target, ensure_ascii=False) + (tokenizer.eos_token or "")
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(answer_ids) >= max_length:
                raise ValueError(f"Target exceeds max_length={max_length}; refusing to truncate the supervised answer")
            available_prompt = max(1, max_length - len(answer_ids))
            prompt_ids = prompt_ids[-available_prompt:]
            input_ids = prompt_ids + answer_ids
            self.rows.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + answer_ids,
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


@dataclass
class CausalCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(row["input_ids"]) for row in features)
        pad_id = self.tokenizer.pad_token_id
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            padding = max_length - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [pad_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def _load_training_frame(paths: ScriptMindPaths, split: str, origin_mode: str) -> pd.DataFrame:
    frame = pd.read_parquet(paths.data_root / "csid" / "csid.parquet")
    frame = frame[frame["split"] == split].copy()
    if origin_mode == "external":
        frame = frame[frame["origin"] == "external"].copy()
    elif origin_mode != "all":
        raise ValueError("origin_mode must be 'all' or 'external'")
    if frame.empty:
        raise RuntimeError(f"No CSID examples for split={split}, origin_mode={origin_mode}")
    return frame


def train_scriptmind(
    *, model_key: str = "llama8b", seed: int = 42, ablation: str = "multitask",
    origin_mode: str = "all", paths: ScriptMindPaths | None = None, force: bool = False,
    epochs: int = 5, learning_rate: float = 1e-4, max_length: int = 4096,
    per_device_batch_size: int = 2, gradient_accumulation_steps: int = 32,
) -> Any:
    if model_key not in MODEL_IDS:
        raise ValueError(f"model_key must be one of {sorted(MODEL_IDS)}")
    if ablation not in VALID_ABLATIONS:
        raise ValueError(f"ablation must be one of {sorted(VALID_ABLATIONS)}")
    paths = (paths or ScriptMindPaths.defaults()).ensure()
    model_id = MODEL_IDS[model_key]
    config = {
        "model_key": model_key, "model_id": model_id, "seed": seed, "ablation": ablation,
        "origin_mode": origin_mode, "epochs": epochs, "learning_rate": learning_rate,
        "max_length": max_length, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    run_id = stable_hash(json.dumps(config, sort_keys=True), 16)
    output_dir = paths.results_root / "training" / f"{model_key}__seed-{seed}__{ablation}__{origin_mode}__{run_id}"
    completed = output_dir / "complete.json"
    metrics_path = output_dir / "metrics.json"
    if completed.exists() and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        return _experiment_result_class()(
            strategy="scriptmind", config=config, df_metrics=pd.DataFrame([metrics]),
            artifacts_dir=str(output_dir), extra={"adapter_path": str(output_dir / "adapter"), "cached": True},
        )

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt and requirements-scriptmind.txt before training") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ScriptMind QLoRA training requires CUDA")
    _seed_everything(seed)
    train_frame = _load_training_frame(paths, "train", origin_mode)
    val_frame = _load_training_frame(paths, "internal_validation", origin_mode)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN") or None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=os.getenv("HF_TOKEN") or None, device_map="auto",
        torch_dtype=torch.bfloat16, quantization_config=quantization,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear",
    ))
    train_dataset = EncodedChatDataset(train_frame, tokenizer, max_length, ablation)
    val_dataset = EncodedChatDataset(val_frame, tokenizer, max_length, ablation)
    checkpoints = output_dir / "checkpoints"
    arguments = TrainingArguments(
        output_dir=str(checkpoints), num_train_epochs=epochs, learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_8bit", bf16=True, gradient_checkpointing=True, max_grad_norm=1.0,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=10,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, seed=seed, data_seed=seed, report_to=[], remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=arguments, train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    train_output = trainer.train()
    evaluation = trainer.evaluate()
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    history = pd.DataFrame(trainer.state.log_history)
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "history.csv", index=False)
    metrics = {
        "model_name": model_key, "dataset_name": "internal_validation",
        "train_loss": float(train_output.training_loss),
        "eval_loss": float(evaluation.get("eval_loss", np.nan)),
        "train_samples": len(train_dataset), "validation_samples": len(val_dataset),
    }
    atomic_write_json(metrics_path, metrics, paths)
    manifest = {"stage": "train", "run_id": run_id, "config": config, "csid_sha256": file_sha256(paths.data_root / "csid" / "csid.parquet")}
    atomic_write_json(output_dir / "manifest.json", manifest, paths)
    atomic_write_json(completed, {"complete": True, "run_id": run_id}, paths)
    return _experiment_result_class()(
        strategy="scriptmind", config=config, model=trainer.model,
        df_history=history, df_metrics=pd.DataFrame([metrics]), artifacts_dir=str(output_dir),
        extra={"adapter_path": str(adapter_dir), "cached": False},
    )
