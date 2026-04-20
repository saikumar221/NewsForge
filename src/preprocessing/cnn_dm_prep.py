"""
CNN/Daily Mail Training Data Preparation
=========================================
Pipeline Stage: ⚙️ Training Environment

Prepares (input, target) pairs from CNN/Daily Mail for BART's two-stage fine-tuning.
Designed to be callable from both local Python and the Colab training notebook —
on Colab, we re-prep in-session rather than uploading large files.

Stage 1 (headline generation):
  input  = article body
  target = first sentence of article, with `(CNN) -- ` and `CITY (CNN) -- `
           datelines stripped (per EDA analysis in notebooks/01_eda.ipynb)

Stage 2 (summary generation):
  input  = f"{headline}\\n{article}"
  target = highlights joined with spaces (newlines collapsed)

CLI:
  python -m src.preprocessing.cnn_dm_prep                     # uses config values
  python -m src.preprocessing.cnn_dm_prep --smoke-test         # 5K/500/500 sanity run
  python -m src.preprocessing.cnn_dm_prep --n-train 10000      # override subset size
"""

import argparse
import os
import re
from typing import Any, Callable

import yaml
from datasets import Dataset, DatasetDict, load_dataset


# --- Shared derivation functions (ported from notebooks/01_eda.ipynb) ------------

# Catches `(CNN) --` and `CITY (CNN) --` variants (see §4 of the EDA notebook).
# Optional leading city/location prefix: capital start, letters/spaces/punct, ≤40 chars.
CNN_PREFIX_RE = re.compile(r"^\s*(?:[A-Z][A-Za-z .,'\-]{1,40})?\(CNN\)\s*--\s*")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def derive_headline(article: str) -> str:
    """
    Return the first sentence of `article` with CNN dateline prefixes removed.

    CNN/DM v3.0.0 articles have no newlines, so we use a regex sentence boundary
    split instead of a line split.
    """
    if not article:
        return ""
    first_sentence = SENTENCE_BOUNDARY_RE.split(article, maxsplit=1)[0].strip()
    return CNN_PREFIX_RE.sub("", first_sentence).strip()


def derive_summary_target(highlights: str) -> str:
    """Collapse highlights newlines into spaces (standard CNN/DM convention)."""
    if not highlights:
        return ""
    text = highlights.replace("\n", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


# --- Dataset loading and filtering ------------------------------------------------


def load_cnn_dm_subset(
    repo_id: str,
    version: str,
    split: str,
    n: int,
) -> list[dict[str, str]]:
    """Stream the first `n` examples of `split` from CNN/Daily Mail."""
    ds = load_dataset(repo_id, version, split=split, streaming=True)
    rows: list[dict[str, str]] = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        rows.append({"id": ex["id"], "article": ex["article"], "highlights": ex["highlights"]})
    return rows


def filter_by_length(
    rows: list[dict[str, Any]], min_article_words: int
) -> tuple[list[dict[str, Any]], int]:
    """Drop examples with fewer than `min_article_words` words. Returns (kept, dropped)."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for r in rows:
        if word_count(r.get("article", "")) >= min_article_words and r.get("highlights"):
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


# --- Stage dataset construction ----------------------------------------------------


def build_stage1_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stage 1: article → derived headline."""
    out: list[dict[str, str]] = []
    for r in rows:
        headline = derive_headline(r["article"])
        if not headline:
            continue
        out.append(
            {"id": r["id"], "article": r["article"], "headline": headline}
        )
    return out


def build_stage2_rows(
    rows: list[dict[str, str]], stage2_separator: str
) -> list[dict[str, str]]:
    """Stage 2: (headline + separator + article) → highlights-as-summary."""
    out: list[dict[str, str]] = []
    for r in rows:
        headline = derive_headline(r["article"])
        summary_target = derive_summary_target(r["highlights"])
        if not headline or not summary_target:
            continue
        combined_input = f"{headline}{stage2_separator}{r['article']}"
        out.append(
            {
                "id": r["id"],
                "input": combined_input,
                "target": summary_target,
                "headline": headline,
            }
        )
    return out


def build_stage_dataset(
    config: dict,
    n_train: int,
    n_val: int,
    n_test: int,
    builder: Callable[[list[dict[str, str]]], list[dict[str, str]]],
    describe: str,
) -> DatasetDict:
    """
    Generic scaffolding to fetch subsets for all three splits and apply the
    stage-specific row builder.
    """
    cfg = config["cnn_dm"]
    splits = {"train": n_train, "validation": n_val, "test": n_test}

    result: dict[str, Dataset] = {}
    totals = {}
    for split_name, n in splits.items():
        raw = load_cnn_dm_subset(
            cfg["repo_id"], cfg["version"], split_name, n
        )
        filtered, dropped = filter_by_length(raw, cfg["min_article_words"])
        built = builder(filtered)
        result[split_name] = Dataset.from_list(built)
        totals[split_name] = {
            "requested": n,
            "fetched": len(raw),
            "after_length_filter": len(filtered),
            "after_builder": len(built),
            "length_dropped": dropped,
        }

    print(f"\n--- {describe} ---")
    for split_name, counts in totals.items():
        print(
            f"  {split_name:<10} requested={counts['requested']:>6}  "
            f"after_filter={counts['after_length_filter']:>6}  "
            f"final={counts['after_builder']:>6}  "
            f"(dropped short: {counts['length_dropped']})"
        )
    return DatasetDict(result)


def build_stage1_dataset(config: dict, n_train: int, n_val: int, n_test: int) -> DatasetDict:
    return build_stage_dataset(
        config=config,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        builder=build_stage1_rows,
        describe="Stage 1 (headline) data",
    )


def build_stage2_dataset(config: dict, n_train: int, n_val: int, n_test: int) -> DatasetDict:
    separator = config["cnn_dm"]["stage2_separator"]
    return build_stage_dataset(
        config=config,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        builder=lambda rows: build_stage2_rows(rows, separator),
        describe="Stage 2 (summary) data",
    )


# --- Persistence -------------------------------------------------------------------


def save_datasets(
    stage1: DatasetDict,
    stage2: DatasetDict,
    output_root: str,
) -> tuple[str, str]:
    """Save both stages as HuggingFace `Dataset.save_to_disk` arrow directories."""
    stage1_dir = os.path.join(output_root, "stage1")
    stage2_dir = os.path.join(output_root, "stage2")
    stage1.save_to_disk(stage1_dir)
    stage2.save_to_disk(stage2_dir)
    return stage1_dir, stage2_dir


# --- CLI --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny subset (5K/500/500) for local sanity checking.",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output directory; defaults to paths.data_cnn_dm from config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["cnn_dm"]

    if args.smoke_test:
        n_train, n_val, n_test = 5000, 500, 500
        run_mode = "SMOKE TEST (5K/500/500)"
    else:
        n_train = args.n_train or cfg["n_train"]
        n_val = args.n_val or cfg["n_val"]
        n_test = args.n_test or cfg["n_test"]
        run_mode = f"FULL ({n_train}/{n_val}/{n_test})"

    print(f"CNN/Daily Mail prep — {run_mode}")
    print(f"  repo: {cfg['repo_id']} v{cfg['version']}")
    print(f"  min_article_words: {cfg['min_article_words']}")
    print(f"  stage2_separator: {cfg['stage2_separator']!r}")

    stage1 = build_stage1_dataset(config, n_train, n_val, n_test)
    stage2 = build_stage2_dataset(config, n_train, n_val, n_test)

    output_root = args.output_root or config["paths"]["data_cnn_dm"]
    stage1_path, stage2_path = save_datasets(stage1, stage2, output_root)

    print(f"\nSaved Stage 1 → {stage1_path}")
    print(f"Saved Stage 2 → {stage2_path}")

    # Final sanity — print one example from each
    print("\n--- Stage 1 sample (train[0]) ---")
    ex1 = stage1["train"][0]
    print(f"  article[:120]: {ex1['article'][:120]!r}")
    print(f"  headline:      {ex1['headline']!r}")

    print("\n--- Stage 2 sample (train[0]) ---")
    ex2 = stage2["train"][0]
    print(f"  input[:140]: {ex2['input'][:140]!r}")
    print(f"  target[:140]: {ex2['target'][:140]!r}")


if __name__ == "__main__":
    main()
