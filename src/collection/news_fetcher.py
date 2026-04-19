"""
NewsAPI Article Collection Module
=================================
Pipeline Stage: Data Ingestion (Section 5.1)

Fetches news articles from NewsAPI across 7 categories (business, entertainment,
general, health, science, sports, technology). Articles are retrieved with
metadata (source, author, title, description, content, URL, published date)
and saved as raw JSON for downstream preprocessing.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

import yaml
from dotenv import load_dotenv
from newsapi import NewsApiClient


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def init_client(api_key: str) -> NewsApiClient:
    """
    Initialize the NewsAPI client.

    Args:
        api_key: NewsAPI authentication key.

    Returns:
        Authenticated NewsApiClient instance.
    """
    if not api_key:
        raise ValueError("API key is missing. Set NEWSAPI_KEY in .env.")
    return NewsApiClient(api_key=api_key)


def fetch_articles_by_category(
    client: NewsApiClient,
    category: str,
    country: str = "us",
    language: str = "en",
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch top headlines for a given news category.

    Args:
        client: Authenticated NewsAPI client.
        category: News category (e.g., 'business', 'technology').
        country: 2-letter ISO country code.
        language: Article language code.
        page_size: Maximum number of articles to retrieve.

    Returns:
        List of article dictionaries from the API response, tagged with the
        originating category.
    """
    response = client.get_top_headlines(
        category=category,
        country=country,
        language=language,
        page_size=page_size,
    )

    if response.get("status") != "ok":
        raise RuntimeError(
            f"NewsAPI request failed for category '{category}': {response}"
        )

    articles = response.get("articles", [])
    for article in articles:
        article["category"] = category
    return articles


def fetch_all_categories(
    client: NewsApiClient, config: dict
) -> list[dict[str, Any]]:
    """
    Fetch articles across all configured categories and deduplicate by URL.

    Args:
        client: Authenticated NewsAPI client.
        config: Project configuration dictionary.

    Returns:
        Combined, deduplicated list of articles from all categories.
    """
    newsapi_cfg = config["newsapi"]
    categories: list[str] = newsapi_cfg["categories"]
    country: str = newsapi_cfg.get("country", "us")
    language: str = newsapi_cfg.get("language", "en")
    page_size: int = newsapi_cfg.get("page_size", 100)

    seen_urls: set[str] = set()
    unique_articles: list[dict[str, Any]] = []

    for category in categories:
        articles = fetch_articles_by_category(
            client,
            category=category,
            country=country,
            language=language,
            page_size=page_size,
        )
        kept = 0
        for article in articles:
            url = article.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
                kept += 1
        print(
            f"  [{category:<14}] fetched={len(articles):>3}  "
            f"kept_after_dedup={kept:>3}"
        )

    return unique_articles


def save_raw_articles(articles: list[dict[str, Any]], output_dir: str) -> str:
    """
    Save fetched articles to a JSON file in the raw data directory.

    Args:
        articles: List of article dictionaries.
        output_dir: Path to the raw data directory.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = os.path.join(output_dir, f"articles_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    return output_path


def summarize_source_diversity(articles: list[dict[str, Any]]) -> dict[str, int]:
    """Count articles per source for a quick diversity check."""
    counts: dict[str, int] = {}
    for article in articles:
        source = (article.get("source") or {}).get("name") or "<unknown>"
        counts[source] = counts.get(source, 0) + 1
    return counts


def main():
    """Run the full article collection pipeline."""
    load_dotenv()

    config = load_config()
    api_key = os.getenv("NEWSAPI_KEY")

    client = init_client(api_key)

    print(
        f"Fetching top headlines from {len(config['newsapi']['categories'])} "
        f"categories (country={config['newsapi'].get('country', 'us')})..."
    )
    articles = fetch_all_categories(client, config)

    output_path = save_raw_articles(articles, config["paths"]["data_raw"])

    source_counts = summarize_source_diversity(articles)
    print(f"\nSaved {len(articles)} unique articles → {output_path}")
    print(f"Source diversity: {len(source_counts)} unique sources")
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 sources:")
    for name, count in top_sources:
        print(f"  {name:<40} {count}")


if __name__ == "__main__":
    main()
