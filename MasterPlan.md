# MasterPlan.md
# Event-Centric Multi-Document News Summarization using Transformer Models

---

## 1. Project Overview

**Title:** Event-Centric Multi-Document News Summarization using Transformer Models
**Course:** Natural Language Processing – CS 6120
**Categories:** Text Summarization (primary), Named Entity Recognition (secondary)
**Duration:** 4 Weeks

### Summary
This project builds an intelligent, end-to-end news summarization system that automatically collects articles from multiple sources, groups those covering the same real-world event, and generates a unified headline and coherent summary for each event cluster. Named entities are extracted and highlighted to give users an at-a-glance understanding of key actors and locations involved. The final system is presented through a user-facing interface that displays summarized events in a clean, readable format.

---

## 2. Objectives

1. Ingest multi-document event clusters from **Multi-News** (primary pipeline input, pre-grouped with reference summaries) and, as an optional live mode, **NewsAPI**
2. Represent articles as semantic embeddings using Sentence Transformers
3. Cluster articles covering the same event using unsupervised clustering (HDBSCAN) — applied to NewsAPI articles; Multi-News arrives pre-clustered
4. Fine-tune BART on **CNN/Daily Mail** in a two-stage pipeline to generate a headline and summary per cluster
5. Extract named entities (people, organizations, locations) using spaCy
6. Evaluate summarization quality using ROUGE and BERTScore — against CNN/Daily Mail (single-doc) and Multi-News (multi-doc) references
7. Present results in a Streamlit UI displaying summarized events with entity highlights

### Dataset Roles

| Dataset | Role | Notes |
|---------|------|-------|
| **CNN/Daily Mail** | BART training corpus | Labeled (article → headline + highlights) pairs for Stage 1 & Stage 2 fine-tuning |
| **Multi-News** | Primary pipeline input & eval | Pre-grouped event clusters (2–10 articles each) with a single reference summary per cluster — used for end-to-end pipeline evaluation and the Streamlit demo |
| **NewsAPI** | Optional live mode | Current-day articles; requires our HDBSCAN clustering step before summarization |

---

## 3. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                        │
│   Primary:  Multi-News (pre-grouped event clusters)         │
│   Optional: NewsAPI (live articles → HDBSCAN clustering)    │
│             → Preprocessing → Clean Articles                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│               SEMANTIC CLUSTERING (NewsAPI only)             │
│   Clean Articles → Sentence Embeddings → HDBSCAN Clusters   │
│   (Skipped for Multi-News — clusters already provided)      │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   SUMMARIZATION PIPELINE                      │
│   Cluster Text → BART Stage 1 (Headline)                    │
│               → BART Stage 2 (Summary, headline as prefix)  │
│   (BART fine-tuned on CNN/Daily Mail)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 NAMED ENTITY RECOGNITION                      │
│   Summary + Articles → spaCy NER → Entity Tags              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     USER INTERFACE                            │
│   Streamlit App → Event Cards (Headline + Summary + Entities)│
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Methodology

### 5.1 Data Sources

The project uses three datasets with distinct, non-overlapping roles.

#### 5.1.1 Multi-News — Primary Pipeline Input
- **Source:** HuggingFace `alexfabbri/multi_news`
- **Shape:** Each example is a cluster of 2–10 articles covering a single event, paired with one human-written reference summary
- **Why it's the primary input:**
  - Articles arrive **pre-grouped** by event → clustering becomes an optional ablation rather than a dependency
  - Each cluster ships with a **reference summary** → enables ROUGE/BERTScore on the full multi-doc pipeline (not only single-doc training)
  - **Full article text** (no truncation), unlike free-tier news APIs
- **Usage:** Select a subset of the test split (~100–500 clusters) for end-to-end pipeline evaluation and the Streamlit demo

#### 5.1.2 NewsAPI — Optional Live Mode
- Fetch articles from NewsAPI across 7 categories (business, entertainment, general, health, science, sports, technology)
- Free tier caveat: content is truncated to ~200 chars. Use `title + description + truncated content` as the working text (sufficient for clustering and short-form demo output)
- Clean and normalize text:
  - Strip source suffixes from titles (e.g., `" - TechCrunch"`)
  - Remove `[+N chars]` truncation markers
  - Normalize whitespace, smart quotes, line breaks
- Used to drive a **live mode** in the Streamlit UI for demo purposes

#### 5.1.3 CNN/Daily Mail — Training Corpus
- **Source:** HuggingFace `cnn_dailymail` (version 3.0.0)
- Labeled `(article → headline, highlights)` pairs
- Used exclusively for BART Stage 1 (headline) and Stage 2 (summary) fine-tuning — **not** for the live pipeline

### 5.2 Semantic Embedding Generation
- Model: `all-MiniLM-L6-v2` from Sentence Transformers
- Encode each article's title + first 3 sentences (representative snippet)
- Save embeddings as NumPy arrays for reuse
- Dimensionality: 384-dimensional dense vectors

### 5.3 Article Clustering
- **Primary:** HDBSCAN (`min_cluster_size=3`, `metric='cosine'`)
  - Handles noise (unclustered articles) gracefully
  - Does not require specifying number of clusters
- **Baseline:** K-Means (for comparison)
- **Evaluation:** Silhouette Score + manual inspection of 20–30 clusters
- Discard singleton clusters and noise points

### 5.4 Multi-Document Input Construction
- Concatenate articles within each cluster ordered by publication time
- Truncate to BART's 1024-token input limit
- Prioritize first 2–3 sentences from each article to maximize coverage

### 5.5 BART Two-Stage Fine-Tuning

#### Stage 1 — Headline Generation
- **Model:** `facebook/bart-large-cnn`
- **Input:** Article/cluster text (max 1024 tokens)
- **Output:** Single headline (max 30 tokens)
- **Target:** Original CNN/Daily Mail headlines

#### Stage 2 — Summary Generation
- **Model:** `facebook/bart-large-cnn`
- **Input:** Generated headline prepended to article/cluster text
- **Output:** 2–3 sentence summary (max 128 tokens)
- **Target:** Concatenated CNN/Daily Mail highlights

#### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Warmup Steps | 500 |
| Batch Size | 8 |
| Gradient Accumulation Steps | 4 |
| Epochs | 3 |
| Max Input Tokens | 1024 |
| Max Output Tokens (headline) | 30 |
| Max Output Tokens (summary) | 128 |
| Hardware | Google Colab Pro (A100) or university cluster |

### 5.6 Named Entity Recognition
- **Model:** `en_core_web_trf` (spaCy transformer model)
- **Entity types:** PERSON, ORG, GPE, DATE, EVENT
- **Applied to:** Generated summaries + source articles
- **Post-processing:** Deduplicate, rank by frequency, filter low-confidence extractions

### 5.7 Ablation Studies
1. **With vs. without headline prefix** in Stage 2 — does the headline improve summary coherence?
2. **Single-document vs. multi-document input** — feed a single representative article vs. the full pre-grouped Multi-News cluster into Stage 2; measure ROUGE/BERTScore against the Multi-News reference summary to quantify the gain from multi-doc input.
3. **Gold clusters vs. HDBSCAN clusters** *(bonus)* — on the NewsAPI live mode or a shuffled Multi-News subset, compare summaries produced from Multi-News' ground-truth clusters against summaries produced from our HDBSCAN-discovered clusters. Isolates clustering quality from summarization quality.

---

## 6. Evaluation

### 6.1 Automatic Metrics

Evaluation is performed against two reference sets, corresponding to the two roles in the pipeline:

| Reference Set | Used For | Reference |
|---------------|----------|-----------|
| CNN/Daily Mail test split | BART Stage 1 & Stage 2 (**single-doc**) | Original article headlines and highlights |
| Multi-News test split | End-to-end **multi-doc** pipeline | Human-written cluster-level reference summaries |

| Metric | Applied To |
|--------|-----------|
| ROUGE-1 | Headline and summary unigram overlap |
| ROUGE-2 | Headline and summary bigram overlap |
| ROUGE-L | Longest common subsequence |
| BERTScore | Semantic similarity of generated outputs |
| Headline–Summary Consistency | BERTScore between headline and summary |
| Silhouette Score | Clustering quality (NewsAPI / HDBSCAN only) |

### 6.2 Human Evaluation
Rate 50–100 sampled headline–summary pairs on a 1–5 scale:

| Criterion | Description |
|-----------|-------------|
| Accuracy | Does the summary correctly reflect the source articles? |
| Fluency | Is the language natural and grammatical? |
| Conciseness | Is the headline punchy and the summary appropriately brief? |
| Coherence | Do the headline and summary work well as a pair? |
| Coverage | Does the summary capture info from multiple sources? |

---

## 7. Project Structure

```
news_summarization/
├── data/
│   ├── raw/                    # Raw articles from NewsAPI
│   ├── processed/              # Cleaned and preprocessed articles
│   ├── embeddings/             # Saved sentence embeddings
│   └── clusters/               # Cluster assignments and grouped articles
├── models/
│   ├── checkpoints/            # Fine-tuned BART model checkpoints
│   └── results/                # Evaluation results (ROUGE, BERTScore)
├── src/
│   ├── collection/
│   │   └── news_fetcher.py     # NewsAPI article collection
│   ├── preprocessing/
│   │   └── cleaner.py          # Text cleaning and normalization
│   ├── embeddings/
│   │   └── embedder.py         # Sentence embedding generation
│   ├── clustering/
│   │   └── clusterer.py        # HDBSCAN and K-Means clustering
│   ├── summarization/
│   │   ├── trainer.py          # BART fine-tuning (Stage 1 & 2)
│   │   └── summarizer.py       # Inference pipeline
│   ├── ner/
│   │   └── entity_extractor.py # spaCy NER extraction
│   └── evaluation/
│       └── evaluator.py        # ROUGE and BERTScore evaluation
├── ui/
│   └── app.py                  # Streamlit UI
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_clustering.ipynb     # Clustering experiments
│   └── 03_evaluation.ipynb     # Results and evaluation
├── tests/
│   └── test_pipeline.py        # Unit tests for each module
├── configs/
│   └── config.yaml             # All project configurations
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Ignore data, checkpoints, .env
├── MasterPlan.md               # This file
└── README.md                   # Setup and run instructions
```

---

## 8. Technology Stack

| Component | Tool / Library |
|-----------|---------------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Transformers | Hugging Face Transformers |
| Datasets | Hugging Face Datasets |
| Embeddings | Sentence Transformers |
| Clustering | HDBSCAN, scikit-learn |
| NER | spaCy (`en_core_web_trf`) |
| Evaluation | rouge-score, bert-score |
| News Collection | newsapi-python |
| UI | Streamlit |
| Config Management | PyYAML, python-dotenv |
| Experiment Tracking | (Optional) Weights & Biases |
| Version Control | Git + GitHub |

---

## 9. Project Todo List

### 🛠️ Project Setup
- [x] Create GitHub repository and clone locally
- [x] Set up Python 3.10+ virtual environment
- [x] Install all dependencies from requirements.txt
- [x] Configure .env file with NewsAPI key and directory paths
- [x] Verify config.yaml is correctly set up
- [x] Add MasterPlan.md to repo root

### 📰 News Collection (Live Mode — Optional)
- [x] Implement NewsAPI fetcher (news_fetcher.py)
- [x] Fetch articles across all 7 categories (business, entertainment, general, health, science, sports, technology)
- [x] Save raw articles to data/raw/
- [x] Verify article count and source diversity

### 📥 Multi-News Pipeline Input (Primary)
- [x] Load Multi-News dataset from Hugging Face (`tingchih/multi_news_doc` — parquet mirror)
- [x] Inspect cluster sizes, article counts, and reference summary lengths
- [x] Select a subset of the test split (300 clusters) as the pipeline eval set
- [x] Normalize article separators (split on `|||||`) and strip `– ` summary prefix
- [x] Save pre-grouped cluster inputs + reference summaries to data/processed/multi_news/clusters.json

### 🧹 Preprocessing (NewsAPI Live Mode)
- [x] Implement text cleaner (cleaner.py)
- [x] Strip source suffixes from titles (e.g., " - TechCrunch")
- [x] Remove `[+N chars]` truncation markers
- [x] Remove HTML tags and boilerplate
- [x] Normalize whitespace, smart quotes, line breaks, encoding
- [x] Build working text = title + description + truncated content (free-tier workaround)
- [x] Flag paywall-preview articles (`is_paywall_preview` field)
- [x] Save processed articles to data/processed/newsapi/

### 📊 CNN/Daily Mail EDA
- [ ] Load CNN/Daily Mail dataset via Hugging Face
- [ ] Analyze article and summary length distributions
- [ ] Inspect headline and highlight quality
- [ ] Construct headline targets from metadata
- [ ] Concatenate highlights into 2–3 sentence summary targets
- [ ] Document EDA findings in 01_eda.ipynb

### 🔢 Embeddings & Clustering
- [ ] Implement embedder.py using all-MiniLM-L6-v2
- [ ] Encode article title + first 3 sentences per article
- [ ] Save embeddings to data/embeddings/
- [ ] Implement HDBSCAN clustering (clusterer.py)
- [ ] Implement K-Means as baseline comparison
- [ ] Tune HDBSCAN parameters (min_cluster_size, metric)
- [ ] Evaluate cluster quality with Silhouette Score
- [ ] Manually inspect 20–30 clusters for coherence
- [ ] Filter singleton clusters and noise points
- [ ] Save cluster assignments to data/clusters/
- [ ] Construct multi-document inputs per cluster
- [ ] Document clustering experiments in 02_clustering.ipynb

### ⚙️ Training Environment
- [ ] Set up Google Colab Pro or university cluster with GPU
- [ ] Verify PyTorch + CUDA installation
- [ ] Prepare CNN/Daily Mail training splits
- [ ] Prepare headline targets (Stage 1 labels)
- [ ] Prepare summary targets (Stage 2 labels)

### 🏷️ BART Stage 1 — Headline Generation
- [ ] Implement training loop in trainer.py for Stage 1
- [ ] Load facebook/bart-large-cnn from Hugging Face
- [ ] Fine-tune on CNN/Daily Mail headlines (max 30 output tokens)
- [ ] Monitor training loss and validation loss
- [ ] Save Stage 1 checkpoint to models/checkpoints/
- [ ] Evaluate Stage 1 on validation set (ROUGE-1, ROUGE-2, ROUGE-L)
- [ ] Qualitatively inspect 20–30 generated headlines

### 📝 BART Stage 2 — Summary Generation
- [ ] Implement training loop in trainer.py for Stage 2
- [ ] Prepend generated headline as context prefix to input
- [ ] Fine-tune on CNN/Daily Mail summaries (max 128 output tokens)
- [ ] Monitor training and validation loss
- [ ] Save Stage 2 checkpoint to models/checkpoints/
- [ ] Evaluate Stage 2 with ROUGE and BERTScore
- [ ] Compute headline–summary consistency score (BERTScore)
- [ ] Qualitatively inspect 20–30 headline–summary pairs

### 📚 Supplementary Fine-Tuning
- [ ] Load Multi-News dataset from Hugging Face
- [ ] Fine-tune Stage 2 on Multi-News for multi-document improvement
- [ ] Compare ROUGE scores before and after Multi-News fine-tuning

### 🔁 End-to-End Pipeline Test
- [ ] Implement inference pipeline in summarizer.py
- [ ] Run full pipeline on 10–20 Multi-News clusters (primary)
- [ ] Run full pipeline on 10–20 NewsAPI HDBSCAN clusters (live-mode smoke test)
- [ ] Inspect generated headline–summary pairs for quality
- [ ] Log any hallucinations or incoherence issues

### 🏷️ Named Entity Recognition
- [ ] Install and set up spaCy with en_core_web_trf model
- [ ] Implement entity_extractor.py
- [ ] Apply NER to generated summaries
- [ ] Apply NER to source articles as fallback
- [ ] Extract PERSON, ORG, GPE, DATE, EVENT entity types
- [ ] Deduplicate and rank entities by frequency
- [ ] Filter low-confidence extractions

### 🖥️ Streamlit UI
- [ ] Design UI layout — event cards with headline, summary, entities, source links
- [ ] Implement entity highlighting in summaries
- [ ] Add topic category filters
- [ ] Add source attribution per event card
- [ ] Primary mode: display pre-computed Multi-News pipeline results
- [ ] Live mode toggle: fetch NewsAPI → cluster → summarize on demand
- [ ] Test UI end-to-end on Multi-News clusters
- [ ] Test UI end-to-end with live NewsAPI data
- [ ] Refine UI styling and usability

### 🔬 Ablation Studies
- [ ] Ablation 1: Run Stage 2 WITHOUT headline prefix — record ROUGE and BERTScore
- [ ] Ablation 1: Run Stage 2 WITH headline prefix — record ROUGE and BERTScore
- [ ] Ablation 1: Compare and document results
- [ ] Ablation 2: Run pipeline with single-document input on Multi-News — record scores
- [ ] Ablation 2: Run pipeline with full multi-document Multi-News clusters — record scores
- [ ] Ablation 2: Compare and document results
- [ ] Ablation 3 (bonus): Compare Multi-News gold clusters vs. HDBSCAN-discovered clusters
- [ ] Ablation 3 (bonus): Document clustering-quality impact on summary quality

### 📏 Automatic Evaluation
- [ ] Run final ROUGE-1, ROUGE-2, ROUGE-L on CNN/Daily Mail test set (single-doc)
- [ ] Run final BERTScore on CNN/Daily Mail test set (single-doc)
- [ ] Run ROUGE + BERTScore on Multi-News test subset against reference summaries (multi-doc)
- [ ] Compute headline–summary consistency scores
- [ ] Compile all results into a summary table (single-doc vs multi-doc)
- [ ] Document results in 03_evaluation.ipynb

### 👤 Human Evaluation
- [ ] Sample 50–100 headline–summary pairs for evaluation
- [ ] Rate each pair on accuracy, fluency, conciseness, coherence, and coverage (1–5)
- [ ] Compute average scores per criterion
- [ ] Compare human scores with automatic metrics

### 🔍 Error Analysis
- [ ] Identify and document hallucination examples
- [ ] Identify and document poor cluster examples
- [ ] Identify and document missed or incorrect entity extractions
- [ ] Identify and document incoherent headline–summary pairs
- [ ] Summarize failure patterns and root causes

### 📄 Final Report
- [ ] Write Introduction section
- [ ] Write Related Work section
- [ ] Write Methodology section
- [ ] Write Results section (with tables and figures)
- [ ] Write Discussion and Ablation Analysis section
- [ ] Write Conclusion section
- [ ] Proofread and finalize report

### 🎤 Presentation & Demo
- [ ] Create presentation slides (motivation, pipeline, results, demo)
- [ ] Prepare live demo on Streamlit with example events
- [ ] Rehearse presentation and demo walkthrough
- [ ] Push final codebase to GitHub with clean README

---