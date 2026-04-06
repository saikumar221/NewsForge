"""
Streamlit User Interface
=========================
Pipeline Stage: User Interface (Section 7 — ui/)

Presents summarized news events in a clean, readable format with:
- Event cards displaying headline + summary for each cluster
- Named entity highlights (PERSON, ORG, GPE, DATE, EVENT)
- Source article links and metadata
- Category and date filters
"""

import json
import os
from typing import Any

import streamlit as st
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_results(results_dir: str) -> list[dict[str, Any]]:
    """
    Load summarization and NER results for display.

    Args:
        results_dir: Path to the model results directory.

    Returns:
        List of event dictionaries with headlines, summaries, entities, and sources.
    """
    # TODO: Load summarization results JSON
    # TODO: Load NER results JSON
    # TODO: Merge into a unified list of event dicts
    pass


def render_entity_highlights(text: str, entities: list[dict[str, str]]) -> str:
    """
    Annotate text with HTML-styled entity highlights.

    Args:
        text: Summary text to annotate.
        entities: List of entity dicts with 'text' and 'label' fields.

    Returns:
        HTML string with highlighted entities.
    """
    # TODO: Replace entity mentions in text with colored HTML spans
    # TODO: Use distinct colors for each entity type
    pass


def render_event_card(event: dict[str, Any]):
    """
    Render a single event card in the Streamlit UI.

    Args:
        event: Dictionary with 'headline', 'summary', 'entities', and 'sources'.
    """
    # TODO: Display headline as a header
    # TODO: Display summary with entity highlights
    # TODO: Display top entities as tags/badges
    # TODO: Display expandable section with source article links
    pass


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Event-Centric News Summarizer",
        page_icon="📰",
        layout="wide",
    )

    st.title("Event-Centric News Summarizer")
    st.markdown("Multi-document summarization of news events with entity highlights.")

    config = load_config()

    # TODO: Add sidebar filters (category, date range)
    # TODO: Load results
    # TODO: Display event cards for each cluster
    # TODO: Handle case where no results are available yet

    st.info("Run the pipeline first to generate results. See README.md for instructions.")


if __name__ == "__main__":
    main()
