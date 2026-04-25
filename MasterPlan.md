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
- **Output:** Single headline (max 48 tokens)
- **Target:** First bullet of CNN/DM `highlights` (editor-curated salience; ~11-word median). Initially tried "first sentence of article" with CNN-dateline strip, but that caused the model to learn extractive copy; pivoted after observing v1 training outputs — see [Status.md](Status.md) for the full debrief.
- **Selection metric during training:** BERTScore F1 on 500-example val subset per epoch (ROUGE-1/2/L + eval_loss logged but not used for selection — ROUGE would bias toward extraction on this task).

#### Stage 2 — Summary Generation
- **Model:** `facebook/bart-large-cnn` (fresh; Stage 2 is trained independently of Stage 1)
- **Input:** Reference (or Stage-1-generated) headline prepended to article/cluster text, newline-separated: `"{headline}\n{article}"`
- **Output:** 2–3 sentence summary (max 128 tokens)
- **Target:** Concatenated CNN/Daily Mail highlights (newlines → spaces, `" ."` artifact normalized to `"."`)
- **Selection metric during training:** `rougeLsum` (HF's canonical per-sentence LCS metric for summarization; predictions/references are sentence-split on `. ! ?` before scoring so rougeLsum doesn't degenerate to rougeL). rouge1/rouge2/rougeL + bertscore_f1 logged alongside.
- **Generation params:** `generation_stage2` config block — `num_beams=4, min_length=30, length_penalty=2.0, no_repeat_ngram_size=3`. `length_penalty=2.0` matches BART's published CNN/DM summarization default; `min_length=30` ensures summaries aren't truncated to a headline.

#### Training Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | |
| Learning Rate | 2e-5 | |
| Warmup Steps | 500 | On the 5K smoke test this exceeds total steps (468) → LR never reaches target; fine at 50K scale. See Status.md open item #11. |
| Batch Size | 4 | Reduced from 8 for Colab free-tier T4 VRAM (~15 GB) |
| Gradient Accumulation Steps | 8 | Keeps effective batch size at 32 |
| Epochs | 3 | |
| Max Input Tokens | 1024 | |
| Max Output Tokens (headline) | 48 | Raised from 30 per EDA finding |
| Max Output Tokens (summary) | 128 | |
| Mixed Precision | fp16 | Halves memory, faster on T4 |
| Gradient Checkpointing | enabled | Trades compute for memory |
| Hardware | Google Colab free tier (Tesla T4, 15 GB VRAM) | Final choice; smoke test validated label design; 50K full run deferred as polish |

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
- [x] Load CNN/Daily Mail dataset via Hugging Face (`abisee/cnn_dailymail` v3.0.0)
- [x] Analyze article and summary length distributions (15K sample: 5K × train/val/test)
- [x] Inspect headline and highlight quality (sample inspection + prefix frequency analysis)
- [x] Construct headline targets = first **sentence** of article with `(CNN) -- ` + `CITY (CNN) -- ` dateline strip (finding: articles have no newlines in v3.0.0; rule covers ~29.6% of articles)
- [x] Concatenate highlights into summary targets (newlines → spaces)
- [x] Document EDA findings in 01_eda.ipynb (incl. recommendation to raise `stage1.max_output_tokens` from 30 → ~48)

### 🔢 Embeddings & Clustering
- [x] Implement embedder.py using all-MiniLM-L6-v2 (L2-normalized 384-dim vectors)
- [x] Encode each article's `text` field (preprocessing already assembles title + description + truncated content)
- [x] Save embeddings to data/embeddings/ (`embeddings.npy` + `article_index.json`)
- [x] Implement HDBSCAN clustering (clusterer.py) — euclidean on L2-normalized vectors ≡ cosine
- [x] Implement K-Means as baseline comparison
- [x] Tune HDBSCAN parameters via sweep in notebook (`min_cluster_size ∈ {2, 3, 5}`); winner: `mcs=2`
- [x] Evaluate cluster quality with Silhouette Score (HDBSCAN 0.094 vs K-Means 0.030)
- [x] Manually inspect clusters for coherence (notebook §7; UMAP 2D plot in §6)
- [x] Filter singleton clusters and noise points (handled by HDBSCAN `min_cluster_size`)
- [x] Save cluster assignments to data/clusters/ (`newsapi_clusters.json` primary, K-Means alongside)
- [x] Construct multi-document inputs per cluster (BART-tokenizer truncation at 1024 tokens, publishedAt-ordered)
- [x] Document clustering experiments in 02_clustering.ipynb

### ⚙️ Training Environment
- [x] Set up GPU — **Google Colab free tier (T4)** selected; actual session happens in Stage 1/2 via `notebooks/04_train_bart_colab.ipynb`
- [ ] Verify PyTorch + CUDA installation — deferred to the Colab training notebook (Stage 1)
- [x] Prepare CNN/Daily Mail training splits (50K train / 1K val / 1K test by default; `--smoke-test` for 5K/500/500)
- [x] Prepare headline targets (Stage 1 labels) — **first bullet of CNN/DM highlights** (editor-curated salience; headline-shaped ~11-word median). Initially used sentence-boundary split of the article but that caused the model to learn extractive copy; pivoted to first-highlight labels after inspecting training outputs.
- [x] Prepare summary targets (Stage 2 labels) — highlights joined (`\n` → space); input = `"{headline}\n{article}"`
- [x] Update training config for T4 — batch_size 4 (↓ from 8), grad_accum 8 (↑ from 4), fp16, gradient_checkpointing

### 🏷️ BART Stage 1 — Headline Generation
- [x] Implement training loop in trainer.py for Stage 1 (Seq2SeqTrainer; ported into `train_stage1()`)
- [x] Load facebook/bart-large-cnn from Hugging Face (`load_tokenizer_and_model()`)
- [x] Build Colab training notebook (`notebooks/04_train_bart_colab.ipynb`) — 11-cell flow, Drive-mounted checkpoints
- [x] Add generation block to config.yaml (num_beams=4, min_length=5, length_penalty=1.0, no_repeat_ngram_size=3)
- [x] Local validation — compute_metrics unit test passed with 3 hand-crafted pairs (rouge1=0.34, bertscore_f1=0.92)
- [x] First Colab training run completed (v1 labels — first sentence of article) — ROUGE-1=0.71 was artificially high; model learned extractive copy instead of abstractive headline generation
- [x] **Pivot**: change headline labels from first-sentence-of-article to first-bullet-of-highlights (editor-curated salience, genuinely abstractive, ~11-word median)
- [x] Fix `min_length=56` warning by overriding bart-large-cnn's generation_config defaults from our config
- [ ] Fine-tune on CNN/Daily Mail (v2 labels, max 48 output tokens) — *run on Colab*
- [ ] Monitor training loss and validation loss — *observed during Colab run*
- [ ] Save Stage 1 checkpoint to models/checkpoints/ — *auto-saved to Drive*
- [ ] Evaluate Stage 1 on validation set (ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1) — *cell 9 (final full-val eval)*
- [ ] Qualitatively inspect 20–30 generated headlines — *cell 10 (20 sample headlines with refs)*

### 📝 BART Stage 2 — Summary Generation
- [x] Implement training loop in trainer.py for Stage 2 (`train_stage2()`; reuses `tokenize_stage_dataset`/`compute_metrics_factory`/`build_training_args` with stage2 config)
- [x] Prepend generated headline as context prefix to input — at training time the *reference* first-highlight headline is used; Stage 1's generated headline is plugged in at inference (evaluation section)
- [x] Add `generation_stage2` block to config.yaml (length_penalty=2.0, min_length=30 — BART's published CNN/DM summarization defaults)
- [x] Extend `compute_metrics_factory` with `rougeLsum` + sentence-boundary splitting (HF's canonical summarization metric)
- [x] Extend `build_training_args` with `selection_metric` parameter (replaces hardcoded bertscore_f1)
- [x] Append Stage 2 cells to `notebooks/04_train_bart_colab.ipynb` (paths+toggle, train, full-val metrics, 20-sample inspection, checkpoint verify)
- [x] Local validation — compute_metrics returns `{rouge1, rouge2, rougeL, rougeLsum, bertscore_f1}`; rougeLsum ≠ rougeL on multi-sentence test, confirming sentence-split is effective
- [x] Fine-tune on CNN/Daily Mail summaries (smoke 5K/500/500, max 128 output tokens) — best checkpoint at epoch 1, val loss climbing after → smoke saturates quickly
- [x] Monitor training and validation loss — train loss 7.42 → 6.10, val loss 1.12 → 1.21 (same warmup-too-long artifact as Stage 1 smoke)
- [x] Save Stage 2 checkpoint — `/content/checkpoints/stage2/best/` (~1.6 GB, ephemeral; Drive persistence available via `SAVE_TO_DRIVE_STAGE2` toggle)
- [x] Evaluate Stage 2 with ROUGE and BERTScore — final full-val: `ROUGE-1=0.518, ROUGE-2=0.397, ROUGE-L=0.479, rougeLsum=0.503, BERTScore-F1=0.917`
- [x] Qualitatively inspect 20 samples — summaries are multi-sentence, abstractive, entity-rich; several preserve or exceed the reference in specificity (samples 10, 16, 17, 19). One factual drift catalogued for 🔍 Error Analysis (sample 15: age drift). No repetition or degenerate outputs.
- [ ] *(Optional polish)* Full 50K Stage 2 fine-tune — ~2h Colab; current smoke model is already strong because `bart-large-cnn` base is already CNN/DM-fine-tuned

*(Headline–summary consistency score moved to §📏 Automatic Evaluation — it's a cross-stage metric computed after both models exist.)*

### 📚 Supplementary Fine-Tuning
- [ ] Load Multi-News dataset from Hugging Face
- [ ] Fine-tune Stage 2 on Multi-News for multi-document improvement
- [ ] Compare ROUGE scores before and after Multi-News fine-tuning

### 🔁 End-to-End Pipeline Test
- [x] Implement inference pipeline in `src/summarization/summarizer.py` — `load_stage_model`, `construct_multidoc_from_strings`, `build_multinews_inputs`, `build_newsapi_inputs`, `run_pipeline`, `run_multinews_pipeline`, `save_results`, CLI `main()`
- [x] Append E2E cells to `notebooks/04_train_bart_colab.ipynb` (cleanup + runner + 10-sample inspection + auto-download)
- [x] Local validation — built inputs for all 300 Multi-News clusters from cached `clusters.json`; 228 hit the 1024-token cap (expected for real multi-doc); 72 stay under
- [ ] Run full pipeline on all **300 Multi-News clusters** — *executes on Colab after Stage 1 + Stage 2 smoke training*
- [ ] Inspect 10 random (headline, summary, reference) triples for quality — *cell 33 of Colab notebook*
- [ ] Log any hallucinations or incoherence issues — *user review from inspection output*
- [x] **Deferred:** NewsAPI HDBSCAN clusters (skipped this section; `build_newsapi_inputs` exists for future use). Headline–summary consistency score → 📏 Automatic Evaluation. ROUGE/BERTScore vs references → 📏 Automatic Evaluation.

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