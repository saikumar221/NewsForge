"""
Named Entity Recognition Module
=================================
Pipeline Stage: NER Extraction (Section 5.6)

Extracts named entities from generated summaries and source articles using
spaCy's transformer-based model (`en_core_web_trf`).

Entity types extracted: PERSON, ORG, GPE, DATE, EVENT

Post-processing:
- Deduplicate entities across articles in a cluster
- Rank by frequency of occurrence
- Filter low-confidence extractions
"""

from typing import Any

import spacy
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_ner_model(model_name: str = "en_core_web_trf") -> spacy.language.Language:
    """
    Load the spaCy NER model.

    Args:
        model_name: spaCy model identifier.

    Returns:
        Loaded spaCy Language pipeline.
    """
    # TODO: Load and return the spaCy model
    # TODO: Handle case where model is not installed (prompt user to download)
    pass


def extract_entities(
    text: str,
    nlp: spacy.language.Language,
    entity_types: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Extract named entities from a text.

    Args:
        text: Input text (summary or article).
        nlp: Loaded spaCy NER pipeline.
        entity_types: List of entity types to extract (e.g., ['PERSON', 'ORG']).
                      If None, extracts all types.

    Returns:
        List of dicts with 'text', 'label', and 'start'/'end' character offsets.
    """
    # TODO: Process text with spaCy pipeline
    # TODO: Filter entities by type if entity_types is specified
    # TODO: Return list of entity dicts
    pass


def deduplicate_entities(
    entities: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Deduplicate entities and rank by frequency.

    Args:
        entities: List of entity dicts (possibly with duplicates).

    Returns:
        Deduplicated list sorted by frequency (descending), each with a 'count' field.
    """
    # TODO: Count occurrences of each (text, label) pair
    # TODO: Sort by frequency descending
    # TODO: Return deduplicated list with counts
    pass


def extract_entities_for_cluster(
    summary: str,
    articles: list[dict[str, Any]],
    nlp: spacy.language.Language,
    config: dict,
) -> list[dict[str, Any]]:
    """
    Extract and aggregate entities from a cluster's summary and source articles.

    Args:
        summary: Generated summary for the cluster.
        articles: Source articles in the cluster.
        nlp: Loaded spaCy NER pipeline.
        config: Project configuration dictionary.

    Returns:
        Deduplicated, ranked list of entities for the cluster.
    """
    # TODO: Extract entities from the summary
    # TODO: Extract entities from each source article
    # TODO: Combine, deduplicate, and rank all entities
    # TODO: Filter low-confidence extractions
    pass


def main():
    """Run NER extraction on all cluster summaries."""
    config = load_config()

    # TODO: Load spaCy model
    # TODO: Load cluster summaries from models/results/
    # TODO: Load source articles for each cluster
    # TODO: Extract entities for each cluster
    # TODO: Save entity results
    # TODO: Print summary (top entities per cluster)
    pass


if __name__ == "__main__":
    main()
