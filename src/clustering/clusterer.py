"""
Article Clustering Module
==========================
Pipeline Stage: Semantic Clustering (Section 5.3) + Multi-Document
Input Construction (Section 5.4)

Groups NewsAPI articles covering the same real-world event using unsupervised
clustering on L2-normalized sentence embeddings:
- Primary method: HDBSCAN (euclidean on normalized vectors ≡ cosine distance)
- Baseline: K-Means
- Quality metric: Silhouette Score on non-noise points

Also constructs per-cluster multi-document inputs for BART summarization:
- Articles ordered by `publishedAt` ascending (insertion-order fallback for nulls)
- Truncated to 1024 BART tokens using the `facebook/bart-large-cnn` tokenizer
- Prioritizes leading content from each article by simple round-robin truncation
  at the whole-article level (keep full articles until budget is exhausted)

Multi-News arrives pre-clustered from the dataset itself, so this module is
only applied to NewsAPI live-mode input.
"""

import json
import os
from typing import Any

import hdbscan
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_embeddings_and_articles(
    embeddings_dir: str, processed_newsapi_dir: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load embeddings.npy and the matching article list (most-recent processed file)."""
    import glob

    emb_path = os.path.join(embeddings_dir, "embeddings.npy")
    embeddings = np.load(emb_path)

    processed_files = sorted(
        glob.glob(os.path.join(processed_newsapi_dir, "articles_*.json")),
        key=os.path.getmtime,
    )
    if not processed_files:
        raise FileNotFoundError(
            f"No processed NewsAPI articles in {processed_newsapi_dir}"
        )
    with open(processed_files[-1], "r", encoding="utf-8") as f:
        articles = json.load(f)

    if len(articles) != len(embeddings):
        raise ValueError(
            f"Row count mismatch: {len(articles)} articles vs "
            f"{len(embeddings)} embeddings. Re-run the embedder."
        )
    return embeddings, articles


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """
    Cluster L2-normalized embeddings with HDBSCAN (euclidean distance).

    On unit-norm vectors, euclidean distance is monotonic with cosine distance:
        ||x - y||^2 = 2 - 2·cos(x, y)
    so ordering is preserved.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    return clusterer.fit_predict(embeddings)


def cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 20) -> np.ndarray:
    """Baseline clustering with K-Means."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return km.fit_predict(embeddings)


def compute_silhouette(
    embeddings: np.ndarray, labels: np.ndarray
) -> float | None:
    """
    Compute silhouette score on non-noise points.

    Returns None if fewer than 2 valid clusters exist (silhouette undefined).
    """
    mask = labels != -1
    valid_labels = labels[mask]
    if len(set(valid_labels)) < 2:
        return None
    return float(silhouette_score(embeddings[mask], valid_labels, metric="euclidean"))


def group_articles_by_cluster(
    articles: list[dict[str, Any]], labels: np.ndarray
) -> dict[int, list[dict[str, Any]]]:
    """Bucket articles by cluster label. Excludes noise (-1)."""
    groups: dict[int, list[dict[str, Any]]] = {}
    for article, label in zip(articles, labels):
        label = int(label)
        if label == -1:
            continue
        groups.setdefault(label, []).append(article)
    return groups


def sort_cluster_articles(
    cluster_articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Sort articles by `publishedAt` ascending, preserving insertion order when
    timestamps are null or equal.
    """
    enumerated = list(enumerate(cluster_articles))

    def sort_key(item):
        idx, article = item
        ts = article.get("publishedAt")
        # Articles with null timestamps sort last while preserving insertion order.
        return (ts is None, ts or "", idx)

    enumerated.sort(key=sort_key)
    return [article for _, article in enumerated]


def construct_multi_doc_input(
    cluster_articles: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_tokens: int = 1024,
) -> str:
    """
    Concatenate articles within a cluster, ordered by `publishedAt`, truncated
    to `max_tokens` BART tokens. Articles are joined by a blank line for visual
    separation in downstream inspection; the summarizer only sees the tokenized
    form.
    """
    ordered = sort_cluster_articles(cluster_articles)
    running_tokens: list[int] = []
    joined_parts: list[str] = []

    for article in ordered:
        text = article.get("text", "")
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

    return "\n\n".join(joined_parts)


def build_cluster_payload(
    method: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    articles: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    params: dict[str, Any],
    max_input_tokens: int,
) -> dict[str, Any]:
    """Assemble the final JSON payload for a clustering run."""
    groups = group_articles_by_cluster(articles, labels)
    silhouette = compute_silhouette(embeddings, labels)

    cluster_entries: list[dict[str, Any]] = []
    for cluster_id in sorted(groups.keys()):
        cluster_articles = groups[cluster_id]
        cluster_articles_sorted = sort_cluster_articles(cluster_articles)
        multi_doc = construct_multi_doc_input(
            cluster_articles_sorted, tokenizer, max_tokens=max_input_tokens
        )
        cluster_entries.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(cluster_articles_sorted),
                "articles": cluster_articles_sorted,
                "multi_doc_input": multi_doc,
            }
        )

    n_noise = int((labels == -1).sum())
    return {
        "metadata": {
            "method": method,
            "n_clusters": len(cluster_entries),
            "n_noise": n_noise,
            "n_articles": int(len(articles)),
            "silhouette_score": silhouette,
            "params": params,
        },
        "clusters": cluster_entries,
    }


def save_payload(payload: dict[str, Any], output_dir: str, filename: str) -> str:
    """Save a cluster payload as JSON to `output_dir/filename`."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    """Run HDBSCAN + K-Means clustering and save cluster payloads."""
    config = load_config()
    processed_root = config["paths"]["data_processed"]
    newsapi_dir = os.path.join(processed_root, "newsapi")

    embeddings, articles = load_embeddings_and_articles(
        config["paths"]["data_embeddings"], newsapi_dir
    )
    print(f"Loaded {len(articles)} articles, embeddings shape {embeddings.shape}")

    hdbscan_params = config["clustering"]["hdbscan"]
    kmeans_params = config["clustering"]["kmeans"]

    print("\nLoading BART tokenizer for multi-doc truncation...")
    tokenizer = AutoTokenizer.from_pretrained(config["summarization"]["model_name"])
    max_input_tokens = config["input_construction"]["max_input_tokens"]

    # --- HDBSCAN ---
    print(f"\nRunning HDBSCAN (min_cluster_size={hdbscan_params['min_cluster_size']})")
    hdbscan_labels = cluster_hdbscan(
        embeddings, min_cluster_size=hdbscan_params["min_cluster_size"]
    )
    hdbscan_payload = build_cluster_payload(
        method="hdbscan",
        embeddings=embeddings,
        labels=hdbscan_labels,
        articles=articles,
        tokenizer=tokenizer,
        params={
            "min_cluster_size": hdbscan_params["min_cluster_size"],
            "metric": "euclidean (on L2-normalized vectors ≡ cosine)",
        },
        max_input_tokens=max_input_tokens,
    )
    print(
        f"  clusters={hdbscan_payload['metadata']['n_clusters']}  "
        f"noise={hdbscan_payload['metadata']['n_noise']}  "
        f"silhouette={hdbscan_payload['metadata']['silhouette_score']}"
    )

    # --- K-Means ---
    print(f"\nRunning K-Means (n_clusters={kmeans_params['n_clusters']})")
    kmeans_labels = cluster_kmeans(embeddings, n_clusters=kmeans_params["n_clusters"])
    kmeans_payload = build_cluster_payload(
        method="kmeans",
        embeddings=embeddings,
        labels=kmeans_labels,
        articles=articles,
        tokenizer=tokenizer,
        params={"n_clusters": kmeans_params["n_clusters"]},
        max_input_tokens=max_input_tokens,
    )
    print(
        f"  clusters={kmeans_payload['metadata']['n_clusters']}  "
        f"silhouette={kmeans_payload['metadata']['silhouette_score']}"
    )

    output_dir = config["paths"]["data_clusters"]
    hdbscan_path = save_payload(hdbscan_payload, output_dir, "newsapi_clusters.json")
    kmeans_path = save_payload(
        kmeans_payload, output_dir, "newsapi_clusters_kmeans.json"
    )

    print(f"\nSaved HDBSCAN payload → {hdbscan_path}")
    print(f"Saved K-Means payload → {kmeans_path}")


if __name__ == "__main__":
    main()
