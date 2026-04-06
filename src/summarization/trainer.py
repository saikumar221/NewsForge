"""
BART Fine-Tuning Module
=========================
Pipeline Stage: Summarization — Training (Section 5.5)

Implements a two-stage fine-tuning pipeline for facebook/bart-large-cnn:

Stage 1 — Headline Generation:
  - Input: Article/cluster text (max 1024 tokens)
  - Output: Single headline (max 30 tokens)
  - Target: Original CNN/Daily Mail headlines

Stage 2 — Summary Generation:
  - Input: Generated headline prepended to article/cluster text
  - Output: 2-3 sentence summary (max 128 tokens)
  - Target: Concatenated CNN/Daily Mail highlights

Training Configuration:
  - Optimizer: AdamW, LR: 2e-5, Warmup: 500 steps
  - Batch size: 8, Gradient accumulation: 4 steps
  - Epochs: 3
"""

import os
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_cnn_dailymail():
    """
    Load the CNN/Daily Mail dataset from HuggingFace.

    Returns:
        HuggingFace DatasetDict with train/validation/test splits.
    """
    # TODO: Load 'cnn_dailymail' dataset (version '3.0.0')
    pass


def prepare_headline_data(dataset, tokenizer, max_input_len: int, max_target_len: int):
    """
    Prepare training data for Stage 1 (headline generation).

    Tokenizes articles as input and headlines as target.

    Args:
        dataset: CNN/Daily Mail dataset split.
        tokenizer: BART tokenizer.
        max_input_len: Maximum input token length (1024).
        max_target_len: Maximum target token length (30).

    Returns:
        Tokenized dataset ready for training.
    """
    # TODO: Tokenize 'article' field as input
    # TODO: Tokenize headline (first line of 'highlights') as target
    # TODO: Return tokenized dataset
    pass


def prepare_summary_data(
    dataset, tokenizer, max_input_len: int, max_target_len: int
):
    """
    Prepare training data for Stage 2 (summary generation).

    Tokenizes headline + article as input and full highlights as target.

    Args:
        dataset: CNN/Daily Mail dataset split.
        tokenizer: BART tokenizer.
        max_input_len: Maximum input token length (1024).
        max_target_len: Maximum target token length (128).

    Returns:
        Tokenized dataset ready for training.
    """
    # TODO: Prepend headline to article text as input
    # TODO: Tokenize concatenated 'highlights' as target
    # TODO: Return tokenized dataset
    pass


def get_training_args(
    config: dict, output_dir: str, stage: str
) -> Seq2SeqTrainingArguments:
    """
    Build Seq2SeqTrainingArguments from config.

    Args:
        config: Project configuration dictionary.
        output_dir: Directory for model checkpoints.
        stage: Training stage identifier ('stage1' or 'stage2').

    Returns:
        Configured Seq2SeqTrainingArguments.
    """
    # TODO: Map config training parameters to Seq2SeqTrainingArguments
    # TODO: Set predict_with_generate=True
    pass


def train_stage1(config: dict) -> str:
    """
    Fine-tune BART for headline generation (Stage 1).

    Args:
        config: Project configuration dictionary.

    Returns:
        Path to the saved Stage 1 checkpoint.
    """
    # TODO: Load tokenizer and model (facebook/bart-large-cnn)
    # TODO: Load and prepare CNN/Daily Mail data for headline generation
    # TODO: Set up Seq2SeqTrainer with training arguments
    # TODO: Train the model
    # TODO: Save checkpoint and return path
    pass


def train_stage2(config: dict, stage1_checkpoint: str | None = None) -> str:
    """
    Fine-tune BART for summary generation (Stage 2).

    Uses the headline (from Stage 1) as a prefix to the input.

    Args:
        config: Project configuration dictionary.
        stage1_checkpoint: Optional path to Stage 1 checkpoint to load from.

    Returns:
        Path to the saved Stage 2 checkpoint.
    """
    # TODO: Load tokenizer and model (optionally from stage1 checkpoint)
    # TODO: Load and prepare CNN/Daily Mail data for summary generation
    # TODO: Set up Seq2SeqTrainer with training arguments
    # TODO: Train the model
    # TODO: Save checkpoint and return path
    pass


def main():
    """Run the two-stage BART fine-tuning pipeline."""
    config = load_config()

    # TODO: Train Stage 1 (headline generation)
    # TODO: Train Stage 2 (summary generation with headline prefix)
    # TODO: Print training summary (loss, checkpoint paths)
    pass


if __name__ == "__main__":
    main()
