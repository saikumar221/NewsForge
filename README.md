# Event-Centric Multi-Document News Summarization using Transformer Models

**Course:** Natural Language Processing — CS 6120

An end-to-end news summarization system that collects articles from multiple sources, groups those covering the same real-world event, and generates a unified headline and coherent summary for each event cluster. Named entities are extracted and highlighted to give users an at-a-glance understanding of key actors and locations involved.

> **Project state:** see [Status.md](Status.md) for a full decision log and journey. [MasterPlan.md](MasterPlan.md) is the detailed specification.

## Current progress

| Stage | Status |
|-------|--------|
| 🛠️ Project Setup | ✅ Complete |
| 📰 News Collection (NewsAPI live mode) | ✅ 340 articles, 203 sources |
| 📥 Multi-News Pipeline Input | ✅ 300 test clusters |
| 🧹 Preprocessing (NewsAPI) | ✅ 335 cleaned, paywall-flagged |
| 📊 CNN/Daily Mail EDA | ✅ 15K sample analyzed |
| 🔢 Embeddings & Clustering | ✅ 21 HDBSCAN clusters |
| ⚙️ Training Environment | ✅ Colab T4, CNN/DM prep module |
| 🏷️ BART Stage 1 (headline) | ✅ Smoke-validated — abstractive headlines (first-highlight labels); 50K full run optional |
| 📝 BART Stage 2 (summary) | ✅ Smoke-validated — rougeLsum=0.503, BERTScore=0.917; 50K full run optional |
| 🏷️ NER | ⏭️ Next |
| 🔁 E2E Pipeline / 📏 Eval / 🔬 Ablations / 🖥️ UI / 👤 Human Eval / 🔍 Error Analysis / 📄 Report | ⏳ Pending |

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
git clone https://github.com/saikumar221/NewsForge.git
cd NewsForge
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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

Each stage can be run independently as a Python module. Everything that runs locally works on CPU; only BART fine-tuning needs a GPU (we use Colab free-tier T4 via `notebooks/04_train_bart_colab.ipynb`).

```bash
# Stage 1a: Load Multi-News pre-grouped event clusters (primary pipeline input)
python -m src.collection.multinews_loader

# Stage 1b: Collect articles from NewsAPI (optional live mode)
python -m src.collection.news_fetcher

# Stage 2: Preprocess and clean NewsAPI articles (live mode only)
python -m src.preprocessing.cleaner

# Stage 3: Prepare CNN/Daily Mail training data (Stage 1 + Stage 2 labels)
python -m src.preprocessing.cnn_dm_prep                   # 50K/1K/1K (full)
python -m src.preprocessing.cnn_dm_prep --smoke-test      # 5K/500/500 smoke

# Stage 4: Generate sentence embeddings (NewsAPI path only)
python -m src.embeddings.embedder

# Stage 5: Cluster articles by event (NewsAPI only; Multi-News arrives pre-clustered)
python -m src.clustering.clusterer

# Stage 6: Fine-tune BART on Colab (not locally — needs GPU)
#   Open notebooks/04_train_bart_colab.ipynb in Colab
#   Set Runtime → T4 GPU, then Runtime → Run all.
#   Cell 5 has SMOKE_TEST and SAVE_TO_DRIVE toggles.

# Stage 7: Run inference on clusters (Multi-News or NewsAPI) — not yet implemented
# python -m src.summarization.summarizer

# Stage 8: Extract named entities — not yet implemented
# python -m src.ner.entity_extractor
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
├── data/
│   ├── raw/                            # NewsAPI raw JSON dumps
│   ├── processed/
│   │   ├── newsapi/                    # Cleaned, paywall-flagged NewsAPI articles
│   │   ├── multi_news/                 # Pre-grouped Multi-News clusters + refs
│   │   └── cnn_dailymail/              # Stage 1 + Stage 2 HF DatasetDicts
│   ├── embeddings/                     # (335, 384) L2-normalized sentence embeddings
│   └── clusters/                       # HDBSCAN + K-Means cluster assignments
├── models/
│   ├── checkpoints/                    # BART fine-tuned checkpoints (local; Colab uses /content or Drive)
│   └── results/                        # Evaluation results
├── src/
│   ├── collection/
│   │   ├── news_fetcher.py             # NewsAPI article collection
│   │   └── multinews_loader.py         # Multi-News parquet loader (primary pipeline input)
│   ├── preprocessing/
│   │   ├── cleaner.py                  # NewsAPI text cleaning / paywall detection
│   │   └── cnn_dm_prep.py              # CNN/DM Stage 1 + Stage 2 label construction
│   ├── embeddings/embedder.py          # Sentence Transformer encoding (NewsAPI)
│   ├── clustering/clusterer.py         # HDBSCAN + K-Means + multi-doc input construction
│   ├── summarization/
│   │   ├── trainer.py                  # BART two-stage fine-tuning (Seq2SeqTrainer)
│   │   └── summarizer.py               # Inference pipeline (skeleton — pending)
│   ├── ner/entity_extractor.py         # spaCy NER (skeleton — pending)
│   └── evaluation/evaluator.py         # ROUGE + BERTScore (skeleton — pending)
├── ui/app.py                           # Streamlit UI (skeleton — pending)
├── notebooks/
│   ├── 01_eda.ipynb                    # CNN/DM + NewsAPI EDA (executed)
│   ├── 02_clustering.ipynb             # HDBSCAN sweep + UMAP viz (executed)
│   ├── 03_evaluation.ipynb             # Evaluation report (skeleton)
│   └── 04_train_bart_colab.ipynb       # Colab T4 training notebook for BART Stage 1/2
├── tests/test_pipeline.py              # Unit tests (skeleton)
├── configs/config.yaml                 # All project config (model, training, paths, metrics)
├── requirements.txt                    # Python dependencies (incl. Colab-safe extras)
├── .env.example                        # Environment variable template
├── MasterPlan.md                       # Full project specification
└── Status.md                           # Decision log and journey
```
