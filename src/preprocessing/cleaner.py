"""
Text Cleaning and Normalization Module
=======================================
Pipeline Stage: Data Ingestion — Preprocessing (Section 5.1.2)

Cleans and normalizes raw articles fetched from NewsAPI:
- Strips source suffixes from titles (e.g., " - TechCrunch")
- Removes `[+N chars]` truncation markers and preceding ellipsis
- Normalizes whitespace, line breaks, smart quotes, ellipsis, and HTML entities
- Assembles a single working `text` field via smart dedup of title+description+content
- Flags paywall preview content (e.g., FT "Then $X per month..." blurbs)
- Drops degenerate entries (text < 20 words)
"""

import glob
import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import yaml


# Characters to normalize to ASCII equivalents.
_SMART_QUOTE_MAP = str.maketrans({
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201C": '"',  # left double quote
    "\u201D": '"',  # right double quote
    "\u2013": "-",  # en-dash
    "\u2014": "-",  # em-dash
})

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TRUNCATION_RE = re.compile(r"\s*…?\s*\[\+\d+\s*chars?\]\s*$")
_WHITESPACE_RE = re.compile(r"\s+")
_MIN_TEXT_WORDS = 20
_PAYWALL_KEYWORDS = (
    "per month",
    "subscribe",
    "sign in",
    "cancel anytime",
    "unlimited access",
    "complete digital access",
    "start your free trial",
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_latest_raw_file(raw_dir: str) -> str:
    """Return the path to the most recently modified articles_*.json file."""
    pattern = os.path.join(raw_dir, "articles_*.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No articles_*.json files in {raw_dir}")
    return max(matches, key=os.path.getmtime)


def load_raw_articles(path: str) -> list[dict[str, Any]]:
    """Load raw articles from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_truncation_marker(text: str) -> str:
    """Remove the trailing `… [+N chars]` marker appended by NewsAPI free tier."""
    if not text:
        return ""
    return _TRUNCATION_RE.sub("", text).rstrip()


def normalize_text(text: str) -> str:
    """Normalize whitespace, line breaks, smart quotes, ellipsis, and HTML."""
    if not text:
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.translate(_SMART_QUOTE_MAP)
    text = text.replace("\u2026", "...")
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_title_suffix(title: str, source_name: str | None) -> str:
    """
    Remove a trailing ` - Source`, ` — Source`, or ` | Source` suffix when it
    matches `source_name`. Case-insensitive match on the source name.
    """
    if not title or not source_name:
        return title
    title = title.rstrip()
    source_lower = source_name.strip().lower()
    for sep in (" - ", " — ", " | "):
        idx = title.lower().rfind(sep + source_lower)
        if idx != -1 and idx + len(sep) + len(source_lower) == len(title):
            return title[:idx].rstrip()
    return title


def is_paywall_preview(text: str) -> bool:
    """Heuristic paywall detection: short text containing paywall keywords."""
    if not text:
        return False
    lowered = text.lower()
    hits = sum(1 for kw in _PAYWALL_KEYWORDS if kw in lowered)
    if hits == 0:
        return False
    word_count = len(text.split())
    return word_count < 50 and hits >= 1


def assemble_text(title: str, description: str, content: str) -> str:
    """
    Combine title, description, and content into a single working text.

    Smart dedup: if `description` is a prefix of `content` (common in NewsAPI
    responses), skip description to avoid duplication.
    """
    parts: list[str] = []
    if title:
        parts.append(title.rstrip(".") + ".")

    use_description = bool(description)
    if use_description and content:
        if content.lower().startswith(description.lower()[:60]):
            use_description = False
    if use_description:
        parts.append(description)
    if content:
        parts.append(content)

    return " ".join(parts).strip()


def clean_article(article: dict[str, Any]) -> dict[str, Any] | None:
    """
    Apply the full cleaning pipeline to a single article.

    Returns the cleaned article dict, or None if the resulting `text` has
    fewer than _MIN_TEXT_WORDS words.
    """
    source_name = (article.get("source") or {}).get("name") or ""

    raw_title = article.get("title") or ""
    title = strip_title_suffix(normalize_text(raw_title), source_name)

    description = normalize_text(article.get("description") or "")
    content = normalize_text(strip_truncation_marker(article.get("content") or ""))

    text = assemble_text(title, description, content)
    word_count = len(text.split())
    if word_count < _MIN_TEXT_WORDS:
        return None

    return {
        "title": title,
        "description": description,
        "content": content,
        "text": text,
        "url": article.get("url"),
        "publishedAt": article.get("publishedAt"),
        "source": source_name,
        "category": article.get("category"),
        "author": article.get("author"),
        "word_count": word_count,
        "is_paywall_preview": is_paywall_preview(content),
    }


def process_articles(
    raw_articles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Clean a list of raw articles and return (cleaned, stats).

    Stats tracks counts for reporting: input, dropped_short, paywall_flagged, kept.
    """
    stats = {"input": len(raw_articles), "dropped_short": 0, "paywall_flagged": 0, "kept": 0}
    cleaned: list[dict[str, Any]] = []
    for article in raw_articles:
        result = clean_article(article)
        if result is None:
            stats["dropped_short"] += 1
            continue
        if result["is_paywall_preview"]:
            stats["paywall_flagged"] += 1
        cleaned.append(result)
    stats["kept"] = len(cleaned)
    return cleaned, stats


def save_processed_articles(
    articles: list[dict[str, Any]], output_dir: str
) -> str:
    """Save cleaned articles to a timestamped JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = os.path.join(output_dir, f"articles_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    return output_path


def main():
    """Run the full NewsAPI preprocessing pipeline."""
    config = load_config()

    raw_dir = config["paths"]["data_raw"]
    processed_root = config["paths"]["data_processed"]
    output_dir = os.path.join(processed_root, "newsapi")

    raw_path = find_latest_raw_file(raw_dir)
    print(f"Loading raw articles from {raw_path}")
    raw_articles = load_raw_articles(raw_path)

    cleaned, stats = process_articles(raw_articles)
    output_path = save_processed_articles(cleaned, output_dir)

    print(f"Saved {stats['kept']} processed articles → {output_path}")
    print("\nPreprocessing stats:")
    print(f"  input articles:      {stats['input']:>4}")
    print(f"  dropped (<20 words): {stats['dropped_short']:>4}")
    print(f"  paywall flagged:     {stats['paywall_flagged']:>4}")
    print(f"  kept:                {stats['kept']:>4}")

    # Word count distribution
    if cleaned:
        wcs = sorted(a["word_count"] for a in cleaned)
        print("\nWord count per article (text field):")
        print(
            f"  min={wcs[0]}  p25={wcs[len(wcs)//4]}  median={wcs[len(wcs)//2]}  "
            f"p75={wcs[3*len(wcs)//4]}  max={wcs[-1]}"
        )


if __name__ == "__main__":
    main()
