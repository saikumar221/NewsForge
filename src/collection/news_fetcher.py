"""
NewsAPI Article Collection Module
=================================
Pipeline Stage: Data Ingestion (Section 5.1)

Fetches news articles from NewsAPI across 6 categories (business, entertainment,
general, health, science, technology). Articles are retrieved with metadata
(source, author, title, description, content, URL, published date) and saved
as raw JSON for downstream preprocessing.
"""

import json
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from newsapi import NewsApiClient

import yaml


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
    # TODO: Initialize and return NewsApiClient with the provided API key
    pass


def fetch_articles_by_category(
    client: NewsApiClient,
    category: str,
    language: str = "en",
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch top headlines for a given news category.

    Args:
        client: Authenticated NewsAPI client.
        category: News category (e.g., 'business', 'technology').
        language: Article language code.
        page_size: Maximum number of articles to retrieve.

    Returns:
        List of article dictionaries from the API response.
    """
    # TODO: Use client.get_top_headlines() to fetch articles for the category
    # TODO: Handle API errors and rate limits gracefully
    pass


def fetch_all_categories(client: NewsApiClient, config: dict) -> list[dict[str, Any]]:
    """
    Fetch articles across all configured categories.

    Args:
        client: Authenticated NewsAPI client.
        config: Project configuration dictionary.

    Returns:
        Combined list of articles from all categories.
    """
    # TODO: Iterate over config['newsapi']['categories']
    # TODO: Call fetch_articles_by_category for each
    # TODO: Deduplicate articles by URL
    pass


def save_raw_articles(articles: list[dict[str, Any]], output_dir: str) -> str:
    """
    Save fetched articles to a JSON file in the raw data directory.

    Args:
        articles: List of article dictionaries.
        output_dir: Path to the raw data directory.

    Returns:
        Path to the saved JSON file.
    """
    # TODO: Create output directory if it doesn't exist
    # TODO: Save articles as JSON with a timestamped filename
    pass


def main():
    """Run the full article collection pipeline."""
    load_dotenv()

    config = load_config()
    api_key = os.getenv("NEWSAPI_KEY")

    # TODO: Initialize client
    # TODO: Fetch articles from all categories
    # TODO: Save raw articles to data/raw/
    # TODO: Print summary (number of articles fetched per category)
    pass


if __name__ == "__main__":
    main()
