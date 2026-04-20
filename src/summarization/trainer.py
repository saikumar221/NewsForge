"""
BART Two-Stage Fine-Tuning Trainer
===================================
Pipeline Stage: Summarization — Training (Section 5.5)

Stage 1 — Headline Generation (implemented here):
  Input:  article body (max 1024 tokens)
  Output: concise headline (max 48 tokens)
  Labels: derived from first sentence of CNN/DM article, CNN-dateline-stripped
          (prepared by src/preprocessing/cnn_dm_prep.py)

Stage 2 — Summary Generation (functions added in a later section):
  Input:  "{headline}\n{article}"
  Output: 2–3 sentence summary (max 128 tokens)
  Labels: CNN/DM highlights joined

Training strategy (tuned for Colab free-tier T4):
  - Start from facebook/bart-large-cnn (already CNN/DM-fine-tuned; faster convergence)
  - fp16 + gradient_checkpointing to fit BART-large in ~15GB VRAM
  - batch_size=4, grad_accum=8 → effective batch size 32
  - Best-model selection by BERTScore F1 (semantic metric chosen for abstractive
    headline generation; ROUGE family logged but not used for selection because
    ROUGE biases toward extractive outputs — see Status.md §Stage 1 discussion)

CLI entry point is provided for potential local use; the primary use site is
`notebooks/04_train_bart_colab.ipynb` which imports these functions and runs
them on a T4.
"""

import argparse
import os
from typing import Any, Callable

import numpy as np
import yaml
from datasets import Dataset, DatasetDict, load_from_disk


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- Model / tokenizer loading -----------------------------------------------------


def load_tokenizer_and_model(model_name: str):
    """
    Load a seq2seq BART model and matching tokenizer from HuggingFace.

    Imported lazily so that the module imports cheaply even without GPU libs
    available (useful for the local compute_metrics unit test).
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# --- Tokenization -------------------------------------------------------------------


def tokenize_stage_dataset(
    dataset: DatasetDict | Dataset,
    tokenizer,
    input_col: str,
    target_col: str,
    max_input: int,
    max_output: int,
    num_proc: int | None = None,
) -> DatasetDict | Dataset:
    """
    Tokenize `input_col` and `target_col` for seq2seq training.

    Uses no padding (handled dynamically by `DataCollatorForSeq2Seq` at batch
    assembly time). Truncation is applied at `max_input` / `max_output`.
    """

    def _fn(batch):
        model_inputs = tokenizer(
            batch[input_col], max_length=max_input, truncation=True
        )
        labels = tokenizer(
            text_target=batch[target_col], max_length=max_output, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_cols = (
        dataset["train"].column_names
        if isinstance(dataset, DatasetDict)
        else dataset.column_names
    )
    return dataset.map(
        _fn, batched=True, remove_columns=remove_cols, num_proc=num_proc
    )


# --- Metrics ------------------------------------------------------------------------


def compute_metrics_factory(
    tokenizer,
    bertscore_model_type: str = "roberta-large",
    rouge_types: list[str] | None = None,
) -> Callable:
    """
    Build a `compute_metrics` closure usable by `Seq2SeqTrainer`.

    Returns a dict on every eval call containing:
      rouge1 / rouge2 / rougeL (F1), bertscore_f1
    `eval_loss` is added by the Trainer itself.

    BERTScore model is initialised once and reused across eval calls (saves
    ~1.4GB re-download per epoch).
    """
    from bert_score import BERTScorer
    from rouge_score import rouge_scorer

    rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
    rouge = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    bertscorer = BERTScorer(
        model_type=bertscore_model_type,
        lang="en",
        rescale_with_baseline=False,
    )

    def _decode(ids: np.ndarray) -> list[str]:
        ids = np.where(ids != -100, ids, tokenizer.pad_token_id)
        return [s.strip() for s in tokenizer.batch_decode(ids, skip_special_tokens=True)]

    def compute_metrics(eval_pred) -> dict[str, float]:
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = _decode(preds)
        decoded_labels = _decode(labels)

        rouge_totals = {rt: 0.0 for rt in rouge_types}
        for pred, ref in zip(decoded_preds, decoded_labels):
            scores = rouge.score(ref, pred)
            for rt in rouge_types:
                rouge_totals[rt] += scores[rt].fmeasure
        n = max(len(decoded_preds), 1)
        out: dict[str, float] = {rt: rouge_totals[rt] / n for rt in rouge_types}

        _, _, f1 = bertscorer.score(decoded_preds, decoded_labels)
        out["bertscore_f1"] = float(f1.mean().item())
        return out

    return compute_metrics


# --- Training-argument construction -------------------------------------------------


def build_training_args(
    training_cfg: dict,
    generation_cfg: dict,
    stage_cfg: dict,
    output_dir: str,
    logging_dir: str | None = None,
):
    """
    Build a `Seq2SeqTrainingArguments` from our YAML config layout.

    transformers has churn across major versions (e.g., 5.0 removed
    `overwrite_output_dir` and `save_safetensors`). Rather than maintain a
    version matrix, this function introspects the installed class signature
    and silently drops kwargs it does not accept — printing which ones were
    dropped, so version-skew surprises stay visible.
    """
    import inspect

    from transformers import Seq2SeqTrainingArguments

    desired: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": training_cfg["epochs"],
        "per_device_train_batch_size": training_cfg["batch_size"],
        "per_device_eval_batch_size": training_cfg["batch_size"],
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "learning_rate": float(training_cfg["learning_rate"]),
        "warmup_steps": training_cfg["warmup_steps"],
        "fp16": training_cfg.get("fp16", False),
        "gradient_checkpointing": training_cfg.get("gradient_checkpointing", False),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "bertscore_f1",
        "greater_is_better": True,
        "predict_with_generate": True,
        "generation_max_length": stage_cfg["max_output_tokens"],
        "generation_num_beams": generation_cfg["num_beams"],
        "logging_dir": logging_dir or os.path.join(output_dir, "logs"),
        "logging_steps": 50,
        "report_to": "none",
        "save_safetensors": True,  # Defaults to True in transformers >=5.0 anyway
    }

    sig_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    accepted = {k: v for k, v in desired.items() if k in sig_params}
    dropped = sorted(set(desired) - set(accepted))
    if dropped:
        print(f"[build_training_args] Dropping unsupported kwargs for this transformers version: {dropped}")

    return Seq2SeqTrainingArguments(**accepted)


# --- Main training orchestration ----------------------------------------------------


def _select_subset(dataset: Dataset, n: int) -> Dataset:
    """Deterministic head-subset. Avoids re-seeding HF's own random state."""
    return dataset.select(range(min(n, len(dataset))))


def train_stage1(
    config: dict,
    stage1_data_dir: str,
    output_dir: str,
    eval_subset_size: int = 500,
) -> dict[str, Any]:
    """
    Fine-tune BART Stage 1 on the prepared CNN/DM headline dataset.

    Args:
        config: project configuration (YAML-loaded).
        stage1_data_dir: path to a DatasetDict directory saved by cnn_dm_prep.py
                         (contains train/validation/test splits with
                         `article` and `headline` columns).
        output_dir: where Seq2SeqTrainer writes checkpoints + logs.
        eval_subset_size: how many val examples to evaluate on per epoch.
                          The full val split is used for the post-training
                          `final_eval` call.

    Returns:
        {
          "train_result": transformers TrainOutput,
          "best_model_checkpoint": path,
          "final_full_val_metrics": dict (the full-split suite),
        }
    """
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

    summ_cfg = config["summarization"]
    stage_cfg = summ_cfg["stage1"]
    training_cfg = summ_cfg["training"]
    generation_cfg = config["generation"]
    eval_cfg = config["evaluation"]

    print(f"[train_stage1] Loading tokenizer + model: {summ_cfg['model_name']}")
    tokenizer, model = load_tokenizer_and_model(summ_cfg["model_name"])

    # BART-large-cnn ships with summary-style generation defaults (min_length=56,
    # max_length=142). For Stage 1 headlines (max 48 tokens) those cause the
    # "min_length > max_length" warning during eval. Override with our config.
    model.generation_config.min_length = generation_cfg["min_length"]
    model.generation_config.max_length = stage_cfg["max_output_tokens"]
    model.generation_config.num_beams = generation_cfg["num_beams"]
    model.generation_config.length_penalty = generation_cfg["length_penalty"]
    model.generation_config.no_repeat_ngram_size = generation_cfg["no_repeat_ngram_size"]
    model.generation_config.early_stopping = generation_cfg["early_stopping"]

    print(f"[train_stage1] Loading prepared data from {stage1_data_dir}")
    data = load_from_disk(stage1_data_dir)
    print(
        f"  splits: "
        f"train={len(data['train'])}, val={len(data['validation'])}, test={len(data['test'])}"
    )

    eval_subset = _select_subset(data["validation"], eval_subset_size)
    print(
        f"  per-epoch eval subset: {len(eval_subset)} / {len(data['validation'])} "
        f"val examples"
    )

    print("[train_stage1] Tokenizing datasets...")
    tokenized_train = tokenize_stage_dataset(
        data["train"],
        tokenizer,
        input_col="article",
        target_col="headline",
        max_input=stage_cfg["max_input_tokens"],
        max_output=stage_cfg["max_output_tokens"],
    )
    tokenized_eval_subset = tokenize_stage_dataset(
        eval_subset,
        tokenizer,
        input_col="article",
        target_col="headline",
        max_input=stage_cfg["max_input_tokens"],
        max_output=stage_cfg["max_output_tokens"],
    )
    tokenized_full_val = tokenize_stage_dataset(
        data["validation"],
        tokenizer,
        input_col="article",
        target_col="headline",
        max_input=stage_cfg["max_input_tokens"],
        max_output=stage_cfg["max_output_tokens"],
    )

    compute_metrics = compute_metrics_factory(
        tokenizer,
        bertscore_model_type=eval_cfg["bertscore_model"],
        rouge_types=eval_cfg["rouge_types"],
    )

    training_args = build_training_args(
        training_cfg=training_cfg,
        generation_cfg=generation_cfg,
        stage_cfg=stage_cfg,
        output_dir=output_dir,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, pad_to_multiple_of=8
    )

    # transformers 5.0 renamed `tokenizer=` → `processing_class=` on Trainer
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval_subset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    import inspect
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    trainer_kwargs["processing_class" if "processing_class" in sig.parameters else "tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("[train_stage1] Starting training...")
    train_result = trainer.train()
    trainer.save_model(os.path.join(output_dir, "best"))

    print("[train_stage1] Full-val evaluation on complete validation split...")
    final_metrics = trainer.evaluate(
        eval_dataset=tokenized_full_val, metric_key_prefix="final_val"
    )

    return {
        "train_result": train_result,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "final_full_val_metrics": final_metrics,
    }


# --- Inference helper ---------------------------------------------------------------


def generate_headlines(
    model,
    tokenizer,
    articles: list[str],
    generation_cfg: dict,
    max_input_tokens: int,
    max_output_tokens: int,
    batch_size: int = 8,
    device: str | None = None,
) -> list[str]:
    """Beam-search headline generation for a list of article texts."""
    import torch

    device = device or ("cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    outputs: list[str] = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_input_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_length=max_output_tokens,
                num_beams=generation_cfg["num_beams"],
                min_length=generation_cfg["min_length"],
                length_penalty=generation_cfg["length_penalty"],
                no_repeat_ngram_size=generation_cfg["no_repeat_ngram_size"],
                early_stopping=generation_cfg["early_stopping"],
            )
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


# --- CLI (secondary; main usage is the Colab notebook) ------------------------------


def main():
    parser = argparse.ArgumentParser(description="Stage 1 BART fine-tuning")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--stage1-data",
        default=None,
        help="Path to Stage 1 DatasetDict (defaults to paths.data_cnn_dm/stage1).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Checkpoint output directory (defaults to paths.model_checkpoints/stage1).",
    )
    parser.add_argument("--eval-subset-size", type=int, default=500)
    args = parser.parse_args()

    config = load_config(args.config)
    stage1_data_dir = args.stage1_data or os.path.join(
        config["paths"]["data_cnn_dm"], "stage1"
    )
    output_dir = args.output_dir or os.path.join(
        config["paths"]["model_checkpoints"], "stage1"
    )

    result = train_stage1(
        config=config,
        stage1_data_dir=stage1_data_dir,
        output_dir=output_dir,
        eval_subset_size=args.eval_subset_size,
    )

    print("\n=== Training complete ===")
    print(f"Best checkpoint: {result['best_model_checkpoint']}")
    print("Full-val metrics:")
    for k, v in result["final_full_val_metrics"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
