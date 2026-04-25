"""
End-to-End Summarization Pipeline
==================================
Pipeline Stage: 🔁 End-to-End Pipeline Test

Chains the two fine-tuned BART stages on pre-grouped event clusters:

  Multi-News cluster ──(concat + truncate)──▶ Stage 1 BART ──▶ headline
                                                                     │
                                                                     ▼
                                              Stage 2 BART ◀── "{headline}\\n{article}"
                                                    │
                                                    ▼
                                              summary (2–3 sentences)

Primary dataset: Multi-News (300 test clusters from `tingchih/multi_news_doc`,
pre-prepped by `src/collection/multinews_loader.py`).
NewsAPI clusters are supported by `build_newsapi_inputs` for completeness but
are not called from `main()` / the notebook in this section.

Expected use site: `notebooks/04_train_bart_colab.ipynb` — Colab T4 cells that
run after both BART stages have finished their smoke training. The module
loads the trained checkpoints fresh from disk, runs inference in batches, and
saves a per-cluster JSON to `data/processed/pipeline_outputs/`.

See Status.md §🔁 End-to-End Pipeline Test for the decision record.
"""

import argparse
import json
import os
from typing import Any

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- Model loading -----------------------------------------------------------------


def load_stage_model(checkpoint_path: str, device: str | None = None):
    """
    Load a fine-tuned BART model + tokenizer from a local checkpoint directory
    and move the model to the target device.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


# --- Multi-doc input construction --------------------------------------------------


def construct_multidoc_from_strings(
    articles: list[str],
    tokenizer,
    max_tokens: int,
    separator: str = "\n\n",
) -> str:
    """
    Concatenate a list of raw article strings into a single input, preserving
    articles whole until the token budget is exhausted (matches the approach
    in `src/clustering/clusterer.py`'s `construct_multi_doc_input`, but for
    raw string lists — Multi-News articles don't have the `text` field that
    NewsAPI articles do).

    Articles past the budget are either dropped (if no room) or BPE-truncated
    (if partial room remains).
    """
    running_tokens: list[int] = []
    joined_parts: list[str] = []
    for text in articles:
        if not text:
            continue
        article_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(running_tokens) + len(article_ids) <= max_tokens:
            running_tokens.extend(article_ids)
            joined_parts.append(text)
        else:
            remaining = max_tokens - len(running_tokens)
            if remaining > 0:
                truncated = tokenizer.decode(
                    article_ids[:remaining], skip_special_tokens=True
                )
                joined_parts.append(truncated)
            break
    return separator.join(joined_parts)


def _derive_reference_headline(reference_summary: str) -> str:
    """
    For Multi-News clusters, approximate the per-cluster reference headline as
    the first sentence of the reference summary. Multi-News has no native
    headline field, and the cluster-level reference summary is the closest
    human-written signal of "what this event is about" — its first sentence
    plays the same role as a headline for distribution-shift comparison.
    """
    import re

    _SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
    if not reference_summary:
        return ""
    return _SENT.split(reference_summary.strip(), maxsplit=1)[0].strip()


def build_multinews_inputs(
    clusters_path: str,
    tokenizer,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    """
    Load pre-grouped Multi-News clusters and build per-cluster inputs.

    Returns a list of dicts — one per cluster — with:
      cluster_id, n_articles, multi_doc_input (truncated),
      reference_summary, reference_headline (first sentence of ref summary),
      source_articles_preview (first 120 chars of each source article).
    """
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    inputs: list[dict[str, Any]] = []
    for c in clusters:
        multi_doc = construct_multidoc_from_strings(
            c["articles"], tokenizer, max_tokens=max_input_tokens
        )
        ref_summary = c.get("reference_summary", "")
        inputs.append(
            {
                "cluster_id": int(c["cluster_id"]),
                "source": "multinews",
                "n_articles": len(c["articles"]),
                "multi_doc_input": multi_doc,
                "reference_summary": ref_summary,
                "reference_headline": _derive_reference_headline(ref_summary),
                "source_articles_preview": [a[:120] for a in c["articles"]],
            }
        )
    return inputs


def build_newsapi_inputs(clusters_path: str) -> list[dict[str, Any]]:
    """
    Load NewsAPI HDBSCAN clusters (already have `multi_doc_input` computed by
    clusterer.py) and flatten into the pipeline-input format. Included for
    completeness; not called from the E2E Colab notebook (scope: Multi-News
    only this section).
    """
    with open(clusters_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return [
        {
            "cluster_id": int(c["cluster_id"]),
            "source": "newsapi",
            "n_articles": c["size"],
            "multi_doc_input": c["multi_doc_input"],
            "reference_summary": None,
            "reference_headline": None,
            "source_articles_preview": [
                a["title"][:120] for a in c["articles"]
            ],
        }
        for c in payload.get("clusters", [])
    ]


# --- Pipeline --------------------------------------------------------------------


def run_pipeline(
    stage1_model,
    stage1_tokenizer,
    stage2_model,
    stage2_tokenizer,
    inputs: list[dict[str, Any]],
    config: dict,
    stage1_batch_size: int = 4,
    stage2_batch_size: int = 2,
) -> list[dict[str, Any]]:
    """
    Run the full Stage 1 → Stage 2 pipeline on a list of pre-built inputs.

    Stage 1: multi_doc_input → generated_headline
    Stage 2: "{generated_headline}\\n{multi_doc_input}" → generated_summary

    Returns augmented input dicts with `generated_headline` and
    `generated_summary` fields.
    """
    from src.summarization.trainer import generate_headlines, generate_summaries

    stage1_cfg = config["summarization"]["stage1"]
    stage2_cfg = config["summarization"]["stage2"]
    stage1_gen = config["generation"]
    stage2_gen = config["generation_stage2"]
    stage2_sep = config["cnn_dm"]["stage2_separator"]

    multidoc_inputs = [it["multi_doc_input"] for it in inputs]

    print(
        f"[run_pipeline] Stage 1 generating {len(multidoc_inputs)} headlines "
        f"(batch={stage1_batch_size})..."
    )
    headlines = generate_headlines(
        model=stage1_model,
        tokenizer=stage1_tokenizer,
        articles=multidoc_inputs,
        generation_cfg=stage1_gen,
        max_input_tokens=stage1_cfg["max_input_tokens"],
        max_output_tokens=stage1_cfg["max_output_tokens"],
        batch_size=stage1_batch_size,
    )

    # Assemble Stage 2 inputs: headline + separator + multi-doc article text.
    # Tokenizer truncation at max_input handles the length overflow.
    stage2_inputs = [
        f"{h}{stage2_sep}{doc}" for h, doc in zip(headlines, multidoc_inputs)
    ]

    print(
        f"[run_pipeline] Stage 2 generating {len(stage2_inputs)} summaries "
        f"(batch={stage2_batch_size})..."
    )
    summaries = generate_summaries(
        model=stage2_model,
        tokenizer=stage2_tokenizer,
        inputs=stage2_inputs,
        generation_cfg=stage2_gen,
        max_input_tokens=stage2_cfg["max_input_tokens"],
        max_output_tokens=stage2_cfg["max_output_tokens"],
        batch_size=stage2_batch_size,
    )

    results: list[dict[str, Any]] = []
    for item, headline, summary in zip(inputs, headlines, summaries):
        out = dict(item)
        out["generated_headline"] = headline.strip()
        out["generated_summary"] = summary.strip()
        results.append(out)
    return results


def save_results(results: list[dict[str, Any]], output_path: str) -> str:
    """Write results list to JSON, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return output_path


# --- High-level orchestrator -------------------------------------------------------


def build_cnn_dm_test_inputs(
    test_data_dir: str,
    n_examples: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build pipeline inputs from a CNN/Daily Mail Stage 1 test split.

    The Stage 1 dataset has columns `[id, article, headline]` where `headline`
    is the first-bullet-of-highlights label we use for training. We use the
    article body as the multi_doc_input (single-doc here, since CNN/DM is not
    pre-grouped) and carry the reference headline + reference summary forward.

    For the reference summary at evaluation time, we re-derive the
    summary-target from the highlights (via `derive_summary_target` in
    cnn_dm_prep) — but since we don't have the highlights field on the Stage 1
    dataset, we instead read it from the Stage 2 dataset (which carries the
    same `id`s with the joined-highlights `target` field).
    """
    from datasets import load_from_disk

    stage1_dir = os.path.join(os.path.dirname(test_data_dir.rstrip("/")), "stage1") \
        if not test_data_dir.endswith("stage1") else test_data_dir
    stage2_dir = os.path.join(os.path.dirname(test_data_dir.rstrip("/")), "stage2") \
        if not test_data_dir.endswith("stage2") else test_data_dir

    s1 = load_from_disk(stage1_dir)["test"]
    s2 = load_from_disk(stage2_dir)["test"]

    # Align by `id` so we can carry both reference headline (from Stage 1) and
    # reference summary (from Stage 2 `target`).
    s2_target_by_id = {row["id"]: row["target"] for row in s2}

    if n_examples is not None:
        s1 = s1.select(range(min(n_examples, len(s1))))

    inputs: list[dict[str, Any]] = []
    for row in s1:
        ref_summary = s2_target_by_id.get(row["id"], "")
        inputs.append(
            {
                "cluster_id": row["id"],
                "source": "cnn_dm_test",
                "n_articles": 1,
                "multi_doc_input": row["article"],
                "reference_summary": ref_summary,
                "reference_headline": row["headline"],
                "source_articles_preview": [row["article"][:120]],
            }
        )
    return inputs


def run_cnn_dm_chained_pipeline(
    stage1_ckpt_dir: str,
    stage2_ckpt_dir: str,
    test_data_root: str,
    config: dict,
    output_path: str | None = None,
    n_examples: int | None = None,
    stage1_batch_size: int = 4,
    stage2_batch_size: int = 2,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run the chained Stage 1 → Stage 2 pipeline on the CNN/Daily Mail test split.

    Used by 📏 Automatic Evaluation to compute headline-summary consistency on
    single-doc inputs, comparable apples-to-apples with the Multi-News E2E
    consistency. The Stage 2 input format here (`"{stage1_headline}\\n{article}"`)
    matches inference time, NOT training time (which used reference headlines).
    """
    print(f"[run_cnn_dm_chained_pipeline] Loading Stage 1 from {stage1_ckpt_dir}")
    stage1_model, stage1_tokenizer = load_stage_model(stage1_ckpt_dir, device=device)

    print(f"[run_cnn_dm_chained_pipeline] Building CNN/DM test inputs")
    inputs = build_cnn_dm_test_inputs(test_data_root, n_examples=n_examples)
    print(f"  {len(inputs)} test articles prepared")

    print(f"[run_cnn_dm_chained_pipeline] Loading Stage 2 from {stage2_ckpt_dir}")
    stage2_model, stage2_tokenizer = load_stage_model(stage2_ckpt_dir, device=device)

    results = run_pipeline(
        stage1_model=stage1_model,
        stage1_tokenizer=stage1_tokenizer,
        stage2_model=stage2_model,
        stage2_tokenizer=stage2_tokenizer,
        inputs=inputs,
        config=config,
        stage1_batch_size=stage1_batch_size,
        stage2_batch_size=stage2_batch_size,
    )

    if output_path:
        save_results(results, output_path)
        print(
            f"[run_cnn_dm_chained_pipeline] Saved {len(results)} chained results "
            f"→ {output_path}"
        )

    return results


def run_multinews_pipeline(
    stage1_ckpt_dir: str,
    stage2_ckpt_dir: str,
    clusters_path: str,
    config: dict,
    output_path: str | None = None,
    stage1_batch_size: int = 4,
    stage2_batch_size: int = 2,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load both fine-tuned BART checkpoints, build inputs from Multi-News
    clusters, run the two-stage pipeline, and (optionally) save results.
    """

    def _resolve_clusters_path(path: str) -> str:
        """Resolve Multi-News clusters path robustly across local/Colab cwd."""
        if os.path.isfile(path):
            return path

        # Resolve relative paths against the repository root derived from this file.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate = os.path.join(repo_root, path)
        if os.path.isfile(candidate):
            return candidate

        # Common Colab working dirs may differ from repo root.
        colab_candidate = os.path.join("/content/NewsForge", path)
        if os.path.isfile(colab_candidate):
            return colab_candidate

        raise FileNotFoundError(
            "Multi-News clusters file not found. "
            f"Tried: '{path}', '{candidate}', '{colab_candidate}'. "
            "Generate it first via src/collection/multinews_loader.py or pass --clusters with an absolute path."
        )

    print(f"[run_multinews_pipeline] Loading Stage 1 from {stage1_ckpt_dir}")
    stage1_model, stage1_tokenizer = load_stage_model(stage1_ckpt_dir, device=device)

    resolved_clusters_path = _resolve_clusters_path(clusters_path)
    print(
        "[run_multinews_pipeline] Building Multi-News inputs from "
        f"{resolved_clusters_path}"
    )
    inputs = build_multinews_inputs(
        clusters_path=resolved_clusters_path,
        tokenizer=stage1_tokenizer,
        max_input_tokens=config["summarization"]["stage1"]["max_input_tokens"],
    )
    print(f"  {len(inputs)} clusters prepared")

    print(f"[run_multinews_pipeline] Loading Stage 2 from {stage2_ckpt_dir}")
    stage2_model, stage2_tokenizer = load_stage_model(stage2_ckpt_dir, device=device)

    results = run_pipeline(
        stage1_model=stage1_model,
        stage1_tokenizer=stage1_tokenizer,
        stage2_model=stage2_model,
        stage2_tokenizer=stage2_tokenizer,
        inputs=inputs,
        config=config,
        stage1_batch_size=stage1_batch_size,
        stage2_batch_size=stage2_batch_size,
    )

    if output_path:
        save_results(results, output_path)
        print(f"[run_multinews_pipeline] Saved {len(results)} results → {output_path}")

    return results


# --- CLI -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="End-to-end BART summarization pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--stage1-ckpt",
        required=True,
        help="Directory of the trained Stage 1 checkpoint (e.g., /content/checkpoints/stage1/best).",
    )
    parser.add_argument(
        "--stage2-ckpt",
        required=True,
        help="Directory of the trained Stage 2 checkpoint (e.g., /content/checkpoints/stage2/best).",
    )
    parser.add_argument(
        "--clusters",
        default=None,
        help="Multi-News clusters JSON (defaults to paths.data_multinews/clusters.json).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output results JSON path (defaults to data/processed/pipeline_outputs/multinews_results.json).",
    )
    parser.add_argument("--stage1-batch-size", type=int, default=4)
    parser.add_argument("--stage2-batch-size", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    clusters_path = args.clusters or os.path.join(
        config["paths"]["data_multinews"], "clusters.json"
    )
    output_path = args.output or "data/processed/pipeline_outputs/multinews_results.json"

    results = run_multinews_pipeline(
        stage1_ckpt_dir=args.stage1_ckpt,
        stage2_ckpt_dir=args.stage2_ckpt,
        clusters_path=clusters_path,
        config=config,
        output_path=output_path,
        stage1_batch_size=args.stage1_batch_size,
        stage2_batch_size=args.stage2_batch_size,
    )

    print(f"\n=== Pipeline complete — {len(results)} clusters processed ===")


if __name__ == "__main__":
    main()
