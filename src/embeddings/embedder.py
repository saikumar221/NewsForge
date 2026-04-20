"""
Sentence Embedding Generation Module
======================================
Pipeline Stage: Semantic Clustering — Embedding (Section 5.2)

Generates 384-dimensional dense vector embeddings for each processed NewsAPI
article using the `all-MiniLM-L6-v2` Sentence Transformer model. Each article
is represented by the preprocessed `text` field (title + description +
truncated content). Embeddings are L2-normalized so that downstream HDBSCAN
clustering can use euclidean distance as a monotonic equivalent of cosine
distance.

Artifacts saved to `data/embeddings/`:
- `embeddings.npy`       — (N, 384) float32 matrix, L2-normalized
- `article_index.json`   — list of {row: int, url: str, title: str}
                           parallel to the embedding rows (ordering contract
                           between embeddings and article metadata)
"""

import glob
import json
import os
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_latest_processed_file(processed_dir: str) -> str:
    """Return the most-recently-modified `articles_*.json` in processed dir."""
    pattern = os.path.join(processed_dir, "articles_*.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No articles_*.json files in {processed_dir}")
    return max(matches, key=os.path.getmtime)


def load_processed_articles(path: str) -> list[dict[str, Any]]:
    """Load processed articles from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_name: str) -> SentenceTransformer:
    """Load the Sentence Transformer model by HuggingFace identifier."""
    return SentenceTransformer(model_name)


def generate_embeddings(
    articles: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode each article's `text` field into a dense vector.

    L2-normalizes the output so downstream clustering can use euclidean
    distance as a stand-in for cosine distance.
    """
    texts = [article["text"] for article in articles]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    articles: list[dict[str, Any]],
    output_dir: str,
) -> tuple[str, str]:
    """Save embeddings.npy and article_index.json to `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, "embeddings.npy")
    idx_path = os.path.join(output_dir, "article_index.json")

    np.save(emb_path, embeddings)
    index = [
        {"row": i, "url": a.get("url"), "title": a.get("title")}
        for i, a in enumerate(articles)
    ]
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    return emb_path, idx_path


def main():
    """Run the full embedding generation pipeline."""
    config = load_config()

    processed_root = config["paths"]["data_processed"]
    newsapi_dir = os.path.join(processed_root, "newsapi")

    input_path = find_latest_processed_file(newsapi_dir)
    print(f"Loading processed articles from {input_path}")
    articles = load_processed_articles(input_path)
    print(f"  {len(articles)} articles")

    model_name = config["embeddings"]["model_name"]
    print(f"Loading Sentence Transformer: {model_name}")
    model = load_model(model_name)

    print("Generating embeddings...")
    embeddings = generate_embeddings(articles, model)
    print(f"  embedding shape: {embeddings.shape}")
    print(f"  L2-norm check: mean={np.linalg.norm(embeddings, axis=1).mean():.4f} "
          f"(should be ~1.0)")

    output_dir = config["paths"]["data_embeddings"]
    emb_path, idx_path = save_embeddings(embeddings, articles, output_dir)
    print(f"\nSaved embeddings → {emb_path}")
    print(f"Saved article index → {idx_path}")


if __name__ == "__main__":
    main()
