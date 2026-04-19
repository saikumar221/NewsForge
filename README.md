# Event-Centric Multi-Document News Summarization using Transformer Models

**Course:** Natural Language Processing — CS 6120

An end-to-end news summarization system that collects articles from multiple sources, groups those covering the same real-world event, and generates a unified headline and coherent summary for each event cluster. Named entities are extracted and highlighted to give users an at-a-glance understanding of key actors and locations involved.

## Pipeline Overview

```
                   ┌─ Multi-News (pre-grouped clusters) ──────────────┐
Data Ingestion ────┤                                                   ├─→ BART Summarization → NER → Streamlit UI
                   └─ NewsAPI (live) → Embedding → HDBSCAN Clustering ┘
```

1. **Data Collection**
   - *Primary:* Load pre-grouped event clusters from **Multi-News** (with reference summaries for evaluation)
   - *Live mode:* Fetch articles from **NewsAPI** across 7 categories
2. **Preprocessing** — Clean HTML, normalize text, filter short articles (NewsAPI path)
3. **Embedding** — Encode articles with Sentence Transformers (`all-MiniLM-L6-v2`) *(NewsAPI path only)*
4. **Clustering** — Group articles by event using HDBSCAN, baseline: K-Means *(NewsAPI path only; Multi-News arrives pre-clustered)*
5. **Summarization** — Two-stage BART fine-tuning on **CNN/Daily Mail**: headline generation → summary generation with headline prefix
6. **NER** — Extract entities (PERSON, ORG, GPE, DATE, EVENT) with spaCy
7. **Evaluation** — ROUGE, BERTScore (vs. CNN/DM single-doc and Multi-News multi-doc references), headline–summary consistency, human evaluation
8. **UI** — Streamlit app displaying event cards with entity highlights

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd news_summarization
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your NewsAPI key
```

## Running the Pipeline

Each stage can be run independently as a Python module:

```bash
# Stage 1a: Load Multi-News pre-grouped event clusters (primary pipeline input)
python -m src.collection.multinews_loader

# Stage 1b: Collect articles from NewsAPI (optional live mode)
python -m src.collection.news_fetcher

# Stage 2: Preprocess and clean articles (NewsAPI path)
python -m src.preprocessing.cleaner

# Stage 3: Generate sentence embeddings (NewsAPI path)
python -m src.embeddings.embedder

# Stage 4: Cluster articles by event (NewsAPI path; Multi-News is pre-clustered)
python -m src.clustering.clusterer

# Stage 5: Fine-tune BART on CNN/Daily Mail (headline + summary)
python -m src.summarization.trainer

# Stage 6: Run inference on clusters (Multi-News or NewsAPI)
python -m src.summarization.summarizer

# Stage 7: Extract named entities
python -m src.ner.entity_extractor
```

## Evaluation

```bash
# Run automatic evaluation (ROUGE, BERTScore)
python -m src.evaluation.evaluator
```

See `notebooks/03_evaluation.ipynb` for detailed evaluation results, ablation studies, and human evaluation analysis.

## Streamlit UI

```bash
streamlit run ui/app.py
```

## Running Tests

```bash
python -m pytest tests/
```

## Project Structure

```
├── data/                   # Raw, processed, embeddings, and cluster data
├── models/                 # Checkpoints and evaluation results
├── src/                    # Pipeline source code
│   ├── collection/         # Multi-News loader (primary) & NewsAPI fetcher (live mode)
│   ├── preprocessing/      # Text cleaning and normalization
│   ├── embeddings/         # Sentence embedding generation
│   ├── clustering/         # HDBSCAN and K-Means clustering
│   ├── summarization/      # BART fine-tuning and inference
│   ├── ner/                # spaCy NER extraction
│   └── evaluation/         # ROUGE and BERTScore evaluation
├── ui/                     # Streamlit application
├── notebooks/              # EDA, clustering experiments, evaluation
├── tests/                  # Unit tests
├── configs/                # Project configuration (config.yaml)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── MasterPlan.md           # Full project specification
```
