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

1. Automatically collect news articles from multiple sources using NewsAPI
2. Represent articles as semantic embeddings using Sentence Transformers
3. Cluster articles covering the same event using unsupervised clustering (HDBSCAN)
4. Fine-tune BART in a two-stage pipeline to generate a headline and summary per cluster
5. Extract named entities (people, organizations, locations) using spaCy
6. Evaluate summarization quality using ROUGE and BERTScore
7. Present results in a Streamlit UI displaying summarized events with entity highlights

---

## 3. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                        │
│   NewsAPI → Raw Articles → Preprocessing → Clean Articles   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    SEMANTIC CLUSTERING                        │
│   Clean Articles → Sentence Embeddings → HDBSCAN Clusters   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   SUMMARIZATION PIPELINE                      │
│   Cluster Text → BART Stage 1 (Headline)                    │
│               → BART Stage 2 (Summary, headline as prefix)  │
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

### 5.1 Data Collection & Preprocessing
- Fetch articles from NewsAPI across 6 categories
- Clean and normalize text:
  - Remove HTML tags and boilerplate
  - Filter articles shorter than 100 words
  - Normalize whitespace and encoding
- Store raw and processed articles separately

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
2. **Single-document vs. multi-document input** — does clustering improve summary quality?

---

## 6. Evaluation

### 6.1 Automatic Metrics
| Metric | Applied To |
|--------|-----------|
| ROUGE-1 | Headline and summary unigram overlap |
| ROUGE-2 | Headline and summary bigram overlap |
| ROUGE-L | Longest common subsequence |
| BERTScore | Semantic similarity of generated outputs |
| Headline–Summary Consistency | BERTScore between headline and summary |
| Silhouette Score | Clustering quality |

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

### 📰 News Collection
- [ ] Implement NewsAPI fetcher (news_fetcher.py)
- [ ] Fetch articles across all 6 categories (politics, tech, sports, health, business, science)
- [ ] Save raw articles to data/raw/
- [ ] Verify article count and source diversity

### 🧹 Preprocessing
- [ ] Implement text cleaner (cleaner.py)
- [ ] Remove HTML tags and boilerplate
- [ ] Filter articles shorter than 100 words
- [ ] Normalize whitespace and encoding
- [ ] Save processed articles to data/processed/

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
- [ ] Run full pipeline on 10–20 NewsAPI clusters
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
- [ ] Integrate full pipeline into UI (collection → clustering → BART → NER → display)
- [ ] Test UI end-to-end with live NewsAPI data
- [ ] Refine UI styling and usability

### 🔬 Ablation Studies
- [ ] Ablation 1: Run Stage 2 WITHOUT headline prefix — record ROUGE and BERTScore
- [ ] Ablation 1: Run Stage 2 WITH headline prefix — record ROUGE and BERTScore
- [ ] Ablation 1: Compare and document results
- [ ] Ablation 2: Run pipeline with single-document input — record scores
- [ ] Ablation 2: Run pipeline with multi-document cluster input — record scores
- [ ] Ablation 2: Compare and document results

### 📏 Automatic Evaluation
- [ ] Run final ROUGE-1, ROUGE-2, ROUGE-L on CNN/Daily Mail test set
- [ ] Run final BERTScore on CNN/Daily Mail test set
- [ ] Compute headline–summary consistency scores
- [ ] Evaluate cluster-level summaries against WCEP references
- [ ] Compile all results into a summary table
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