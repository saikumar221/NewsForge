"""
Multi-News Dataset Loader
==========================
Pipeline Stage: Data Ingestion (Section 5.1.1)

Loads the Multi-News dataset from the tingchih/multi_news_doc HuggingFace
mirror (parquet format), splits the concatenated `document` field into
individual articles on the `|||||` separator, strips the `– ` prefix from
reference summaries, and saves a configurable subset of the test split as
pre-grouped event clusters for downstream summarization and evaluation.
"""

import json
import os
from typing import Any

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download


# Parquet filenames per split as stored in the tingchih/multi_news_doc repo.
SPLIT_FILENAMES = {
    "train": [
        "data/train-00000-of-00002-8ab87a5d3dc1eacc.parquet",
        "data/train-00001-of-00002-3f498d0c852ba65c.parquet",
    ],
    "validation": ["data/validation-00000-of-00001-37c524d2a821b952.parquet"],
    "test": ["data/test-00000-of-00001-c7757872ba2a3d58.parquet"],
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_split(repo_id: str, split: str) -> list[str]:
    """
    Download the parquet file(s) for a given split via huggingface_hub and
    return their local cache paths.
    """
    filenames = SPLIT_FILENAMES.get(split)
    if not filenames:
        raise ValueError(
            f"Unsupported split '{split}'. Expected one of {list(SPLIT_FILENAMES)}"
        )
    paths: list[str] = []
    for fname in filenames:
        paths.append(hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset"))
    return paths


def load_split_dataframe(repo_id: str, split: str) -> pd.DataFrame:
    """Download and concatenate parquet shards for a split into a single DataFrame."""
    paths = download_split(repo_id, split)
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return df


def parse_cluster(
    row: dict[str, Any],
    cluster_id: int,
    separator: str,
    summary_prefix: str,
) -> dict[str, Any]:
    """
    Convert a single raw row into a normalized cluster dict.

    Args:
        row: Row with `document` and `summary` fields.
        cluster_id: Integer identifier for this cluster.
        separator: String that separates articles within `document`.
        summary_prefix: Prefix to strip from the reference summary (if present).

    Returns:
        Dict with 'cluster_id', 'articles' (list[str]), 'reference_summary' (str).
    """
    document: str = row["document"]
    summary: str = row["summary"]

    articles = [a.strip() for a in document.split(separator) if a.strip()]
    cleaned_summary = summary.lstrip()
    if cleaned_summary.startswith(summary_prefix):
        cleaned_summary = cleaned_summary[len(summary_prefix):].lstrip()

    return {
        "cluster_id": cluster_id,
        "articles": articles,
        "reference_summary": cleaned_summary,
    }


def load_clusters(config: dict) -> list[dict[str, Any]]:
    """
    Load Multi-News clusters from the configured split and return a subset.
    """
    cfg = config["multinews"]
    repo_id = cfg["repo_id"]
    split = cfg["split"]
    subset_size = cfg["subset_size"]
    separator = cfg["separator"]
    summary_prefix = cfg["summary_prefix"]

    print(f"Downloading Multi-News ({repo_id}, split={split})...")
    df = load_split_dataframe(repo_id, split)
    print(f"  Loaded {len(df)} total examples. Taking first {subset_size}.")

    subset = df.head(subset_size)
    clusters = [
        parse_cluster(
            row={"document": row["document"], "summary": row["summary"]},
            cluster_id=idx,
            separator=separator,
            summary_prefix=summary_prefix,
        )
        for idx, (_, row) in enumerate(subset.iterrows())
    ]
    return clusters


def save_clusters(clusters: list[dict[str, Any]], output_dir: str) -> str:
    """Save clusters to `clusters.json` in the given directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clusters.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    return output_path


def summarize_clusters(clusters: list[dict[str, Any]]) -> None:
    """Print a quick summary of cluster sizes and summary lengths."""
    article_counts = [len(c["articles"]) for c in clusters]
    summary_word_counts = [len(c["reference_summary"].split()) for c in clusters]
    article_char_counts = [
        sum(len(a) for a in c["articles"]) for c in clusters
    ]

    def stats(xs: list[int]) -> str:
        return (
            f"min={min(xs)}  max={max(xs)}  "
            f"mean={sum(xs)/len(xs):.1f}  median={sorted(xs)[len(xs)//2]}"
        )

    size_buckets: dict[int, int] = {}
    for n in article_counts:
        size_buckets[n] = size_buckets.get(n, 0) + 1

    print(f"\nClusters: {len(clusters)}")
    print(f"  articles/cluster       {stats(article_counts)}")
    print(f"  total_chars/cluster    {stats(article_char_counts)}")
    print(f"  summary_words/cluster  {stats(summary_word_counts)}")
    print("  cluster size distribution (N articles → count):")
    for size in sorted(size_buckets):
        print(f"    {size:>2} articles: {size_buckets[size]}")


def main():
    """Load Multi-News test subset, split into clusters, and save to JSON."""
    config = load_config()

    clusters = load_clusters(config)

    output_path = save_clusters(clusters, config["paths"]["data_multinews"])
    print(f"\nSaved {len(clusters)} clusters → {output_path}")

    summarize_clusters(clusters)


if __name__ == "__main__":
    main()
