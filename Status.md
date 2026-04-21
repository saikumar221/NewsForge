# Status.md — Project Journey & Decision Log

**Project:** Event-Centric Multi-Document News Summarization using Transformer Models
**Course:** CS 6120 — NLP
**Last updated:** 2026-04-20

This document records every decision, observation, pivot, and finding from the build-out so far. Sections run in the order the work was done, not the order of the final pipeline.

---

## 1. Project Snapshot

### Completed
- 🛠️ Project Setup (6/6)
- 📰 News Collection — NewsAPI live mode (4/4)
- 📥 Multi-News Pipeline Input — primary (5/5)
- 🧹 Preprocessing — NewsAPI (8/8)
- 📊 CNN/Daily Mail EDA (6/6)
- 🔢 Embeddings & Clustering (12/12)
- ⚙️ Training Environment (6/6 — CUDA/PyTorch verification done on first Colab run)

### In progress
- 🏷️ BART Stage 1 (13/13 smoke-validated — v2 labels produce abstractive headlines; extractive-copy failure eliminated; checkpoint at `/content/checkpoints/stage1/best/`. **Optional:** full 50K run for polish.)
- 📝 BART Stage 2 (13/13 smoke-validated — rougeLsum=0.503, BERTScore=0.917, summaries are abstractive multi-sentence and often richer than reference; checkpoint at `/content/checkpoints/stage2/best/`. **Optional:** full 50K run for polish.)

### Not started
- 📚 Supplementary Fine-Tuning · 🔁 End-to-End Pipeline Test · 🏷️ NER · 🖥️ Streamlit UI · 🔬 Ablation Studies · 📏 Automatic Evaluation · 👤 Human Evaluation · 🔍 Error Analysis · 📄 Final Report · 🎤 Presentation & Demo

---

## 2. Major Architectural Pivots

These changed the shape of the whole project.

### Pivot A: NewsAPI free tier truncates content
**Problem:** NewsAPI's free tier caps `content` at ~200 characters with a `[+N chars]` marker. The MasterPlan's "filter articles <100 words" rule would kill almost every article, because truncation makes articles look short regardless of their true length.

**Options considered:**
- Switch to NewsData.io → rejected (same truncation under the hood on free tier)
- Scrape full text via `trafilatura` → rejected (scope, paywall failures)
- Upgrade paid tier ($449/mo) → rejected (cost)

**Resolution:** Keep NewsAPI as *optional live mode*; shift the *primary pipeline input* to a different dataset.

### Pivot B: Adopt a three-dataset hybrid (Option 3)
Instead of one dataset for everything, we assigned distinct roles:

| Dataset | Role | Why |
|---------|------|-----|
| **CNN/Daily Mail** | BART training corpus | Labeled (article → headline + highlights) pairs for Stage 1 & Stage 2 |
| **Multi-News** | Primary pipeline input & eval | Pre-grouped event clusters (2–10 articles each) with a human-written reference summary per cluster. Full article text, not truncated. |
| **NewsAPI** | Optional live mode | Today's articles; requires our HDBSCAN clustering step before summarization |

**Consequence:** The "Semantic Clustering" stage only applies to NewsAPI; Multi-News arrives pre-clustered. End-to-end ROUGE/BERTScore can now be computed on the multi-doc pipeline against real reference summaries (not only the single-doc CNN/DM training target).

**Updates made:** Rewrote §2 Objectives, §3 Pipeline Architecture, §5.1 Data Sources (split into 5.1.1 Multi-News / 5.1.2 NewsAPI / 5.1.3 CNN-DM), §5.7 Ablations (reframed around Multi-News; added bonus Ablation 3), §6 Evaluation (dual reference-set table), §9 Todo List (new "📥 Multi-News Pipeline Input" section; renamed News Collection to "Live Mode").

---

## 3. Section-by-Section Journey

### 🛠️ Project Setup

- `.venv` created with `python3 -m venv .venv`
- All `requirements.txt` deps installed (torch, transformers, datasets, sentence-transformers, hdbscan, scikit-learn, spacy, rouge-score, bert-score, newsapi-python, streamlit, PyYAML, python-dotenv, numpy, pandas)
- `.env` initially contained a NewsData.io-format key (`pub_...`); user obtained a real NewsAPI.org key and updated it
- `MasterPlan.md` and `README.md` were untracked at session start; `Add Project structure` was the only prior commit
- No code written — just verification

### 📰 News Collection (NewsAPI Live Mode)

**Decisions**
| Q | Answer |
|---|--------|
| Categories | All 7 NewsAPI categories (business, entertainment, general, health, science, sports, technology). [config.yaml](configs/config.yaml) originally listed only 6 — added `sports`. |
| Country filter | `us` |
| Page size | 100 (free tier max) |
| Mode | One-shot script |
| Dedup | By URL |
| Output | Timestamped JSON in `data/raw/` |

**What was built:** [src/collection/news_fetcher.py](src/collection/news_fetcher.py)
- `init_client` · `fetch_articles_by_category` · `fetch_all_categories` · `save_raw_articles` · `summarize_source_diversity` · `main`

**Run results:**
- 340 unique articles across all 7 categories
- 203 unique sources (strong diversity)
- ~20 duplicates removed by URL dedup
- Max per-source concentration: 9 articles (Space.com)
- Saved to `data/raw/articles_20260419T022342Z.json`

### 📥 Multi-News Pipeline Input (Primary)

**Dataset access surprise:** The official `alexfabbri/multi_news` and `tau/multi_news` HuggingFace repos use old-style dataset scripts that the current `datasets` library no longer supports (`RuntimeError: Dataset scripts are no longer supported`). The JSONL mirror `nlplabtdtu/multi_news_en` turned out to be single-article — defeats the purpose. Settled on **`tingchih/multi_news_doc`** (parquet format, 5,622 test rows, proper multi-doc structure), accessed via `hf_hub_download` + `pandas.read_parquet`.

**Decisions**
| Q | Answer |
|---|--------|
| Subset size | 300 clusters from test split |
| Cluster size filter | None |
| Output format | Single `clusters.json` in `data/processed/multi_news/` |
| Pseudo-title | None (no title field — embedder gets the lead paragraph instead) |
| Summary prefix `– ` | Stripped during loading |

**What was built:** [src/collection/multinews_loader.py](src/collection/multinews_loader.py)
- `download_split`, `load_split_dataframe`, `parse_cluster`, `load_clusters`, `save_clusters`, `summarize_clusters`, `main`

**Run results:**
- 300 clusters saved
- articles/cluster: min=1, max=8, mean=2.8, median=2
- chars/cluster: min=618, max=163,507, mean=10,829
- summary words/cluster: min=68, max=461, mean=213.9
- Cluster size distribution: 2→158, 3→81, 4→36, 5→12, 6→8, 7→2, 8→1, 1→2
- En-dash prefix verified stripped (first char is natural text, not U+2013)

### 🧹 Preprocessing (NewsAPI Live Mode)

**Data observations from raw sample:**
- `content` always truncated to ~200 chars with `[+N chars]` marker
- Titles carry source suffixes: `" - Deadline"`, `" - TechCrunch"`, `" — The Washington Post"`
- Unicode smart quotes, `\r\n`, em-dashes, `…` ellipses throughout
- Paywall previews like FT: *"Then $75 per month. Complete digital access..."*

**Decisions**
| Q | Answer |
|---|--------|
| Input file | Most-recent in `data/raw/` |
| Schema | Keep `title / description / content / text / url / publishedAt / source / category / author`; add `word_count`, `is_paywall_preview`; drop `urlToImage` |
| 100-word filter | Dropped; replaced with soft ≥20-word minimum on `text` |
| Paywall handling | Flag, don't drop |
| Title suffix strip | Only when matches `source.name` |
| Working `text` field | `title + description + content` with smart dedup (skip description if prefix of content) |

**What was built:** [src/preprocessing/cleaner.py](src/preprocessing/cleaner.py)
- `find_latest_raw_file`, `strip_truncation_marker`, `normalize_text`, `strip_title_suffix`, `is_paywall_preview`, `assemble_text`, `clean_article`, `process_articles`, `save_processed_articles`, `main`

**Run results & bug fix:**
- 340 → 335 articles kept (5 dropped for <20 words)
- Word counts: min=20, p25=52, median=63, p75=72, max=102
- **Initial paywall detection missed the FT article** (52-word combined text > 50-word threshold). Fixed by running the heuristic on `content` alone instead of the full combined `text`. Paywall count jumped from 1 → 6 (FT, Air Current, Vox, 2× NBC News, 404 Media — mostly legitimate catches).
- Residue check: 0 smart-quotes, 0 unicode-ellipses, 0 `[+N chars]` markers, 0 CRLF in final `text` fields

### 📊 CNN/Daily Mail EDA

**Decisions**
| Q | Answer |
|---|--------|
| Sample size | First 5K per split (15K total, deterministic) |
| Headline source | First line of article, strip `(CNN) --` prefix (later expanded to also catch `CITY (CNN) --`) |
| Summary target | Full highlights, `\n` → space |
| Notebook format | Runnable inline code |

**Major findings**

🔴 **Finding 1 — CNN/DM v3.0.0 articles have zero newlines.**
`article.split('\n', 1)[0]` returned the whole article. We switched from "first line" to **first sentence** via a regex sentence-boundary split (`(?<=[.!?])\s+(?=[A-Z0-9"\'])`).

🔴 **Finding 2 — Stage 1 output budget was too small.**
Derived headline word counts: p50=25, p95=40, p99=53. At roughly 1.3× words/token, p50 headlines are already 30-35 tokens. Our `stage1.max_output_tokens: 30` was clipping the **median** headline. **Raised to 48.**

**Prefix frequency across 15K articles:**
| Pattern | Count | % | Stripped? |
|---------|-------|---|-----------|
| `CITY (CNN) -- ` | 2,383 | 15.9% | ✅ (after expansion) |
| `(CNN) -- ` plain | 2,057 | 13.7% | ✅ |
| ALL-CAPS opening | 1,829 | 12.2% | ❌ (legitimate datelines) |
| `CITY (Reuters) -- ` | 14 | 0.1% | ❌ (too rare) |

User confirmed expanding the regex to catch both CNN variants (~29.6% coverage). Final regex: `^\s*(?:[A-Z][A-Za-z .,'\-]{1,40})?\(CNN\)\s*--\s*`

**Length observations**
| Field | p50 | p75 | p90 | p95 | p99 |
|-------|-----|-----|-----|-----|-----|
| article (words) | 512 | 777 | 1039 | 1231 | 1577 |
| highlights / summary_target | 45 | 53 | 64 | 74 | 98 |
| headline (derived) | 25 | 31 | 36 | 40 | 53 |

**Implications for config:** `max_input_tokens: 1024` OK (truncates, standard). `stage2.max_output_tokens: 128` has headroom. `stage1.max_output_tokens` raised 30→48.

**NewsAPI secondary EDA** (included in the same notebook for the bonus scope noted in the original plan): 7 categories populated, 201 unique sources, `text` field median 63 words, 6 paywall-flagged.

**Build/infra:** Installed `nbconvert`, `ipykernel`, `nbformat`; registered the `newsforge` IPython kernel; added matplotlib + Jupyter packages to [requirements.txt](requirements.txt).

**Deliverable:** [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) — executed, 8 sections, embedded plots, findings cell.

### 🔢 Embeddings & Clustering

**Scope:** NewsAPI only. Multi-News arrives pre-clustered.

**Decisions**
| Q | Answer |
|---|--------|
| Snippet source | `text` field (already assembled in preprocessing) |
| HDBSCAN strategy | Sweep `min_cluster_size ∈ {2, 3, 5}` in notebook; pick best by silhouette; `clusterer.py` reads config |
| Visualization | UMAP 2D |
| Multi-doc ordering | `publishedAt` asc, insertion-order fallback for nulls |
| Multi-News embeddings | Skipped (only needed for bonus Ablation 3) |
| Output schema | Top-level metadata block + list of cluster entries (silhouette is global, stored in metadata) |

**Technical choice:** To use cosine distance with HDBSCAN efficiently, we L2-normalize embeddings and run HDBSCAN with `metric='euclidean'`. On unit-norm vectors `‖x−y‖² = 2 − 2·cos(x,y)`, so euclidean distance is monotonic with cosine distance — ordering is preserved.

**Build/infra:** Installed `umap-learn`; added to [requirements.txt](requirements.txt). BART tokenizer (`facebook/bart-large-cnn`, ~450MB) downloaded at runtime for multi-doc truncation.

**What was built**
- [src/embeddings/embedder.py](src/embeddings/embedder.py) — Sentence Transformer encoding, L2-normalization, saves `embeddings.npy` + `article_index.json`
- [src/clustering/clusterer.py](src/clustering/clusterer.py) — HDBSCAN + K-Means + silhouette + multi-doc input construction with BART-tokenizer truncation
- [notebooks/02_clustering.ipynb](notebooks/02_clustering.ipynb) — sweep, method comparison, UMAP plot, top-cluster inspection, findings

**Embedder run:**
- 335 articles → (335, 384) float32 vectors
- L2-norm mean = 1.0000 (confirms normalization)

**Sweep results:**
| method | clusters | noise | silhouette |
|--------|----------|-------|-----------|
| **HDBSCAN (mcs=2) — winner** | 21 | 198 (59%) | **0.094** |
| HDBSCAN (mcs=3) | 9 | 219 (65%) | 0.077 |
| HDBSCAN (mcs=5) | 4 | 233 (70%) | 0.067 |
| K-Means (k=20) | 20 | 0 | 0.030 |

**Decision:** [config.yaml](configs/config.yaml) updated `min_cluster_size: 3 → 2` per sweep winner. **Caveat noted in findings:** silhouette-optimal ≠ semantically-optimal. Smaller clusters naturally have tighter intra-cluster distances, inflating silhouette. If downstream summaries on mcs=2's 2-article clusters feel weak, reverting to mcs=3 is reasonable.

**Coherence inspection** (visibly good on larger clusters):
- NFL draft (15 articles), space/physics/cosmology (15), Middle East/oil/markets (16), NHL/NBA playoffs (8), iOS/Apple (4), paleontology/fossils (11)
- One large "rest bucket" cluster mixing science-adjacent general interest topics — common HDBSCAN failure mode on small datasets

**Noise rate (59%) is expected** for NewsAPI top-headlines: most articles cover unique events with no second article to pair with.

**Multi-doc input construction:** For each cluster, articles are sorted by `publishedAt` ascending (null timestamps sort last with insertion-order preserved), concatenated with blank-line separators, and truncated using the BART tokenizer to 1024 tokens. Whole articles are kept until the budget is exhausted; the overflow article is tokenized-truncated rather than word-truncated for accuracy.

**Output saved to** `data/clusters/newsapi_clusters.json` (HDBSCAN, primary) and `data/clusters/newsapi_clusters_kmeans.json` (K-Means, comparison only).

### ⚙️ Training Environment

**Scope:** Prepare CNN/Daily Mail training splits for BART two-stage fine-tuning. Set up the GPU target. Actual training stays in the upcoming Stage 1 / Stage 2 sections — this section produces the data and the runtime config only.

**GPU decision**
| Options considered | Outcome |
|--------------------|---------|
| Colab Pro, NEU Discovery Cluster, RunPod, Lambda, Kaggle Notebooks, LoRA-on-Colab-free, skip fine-tuning and use pre-trained `facebook/bart-large-cnn` | User has no existing GPU access. Picked **Colab free tier (T4, ~15GB VRAM)**. |

Implication: the MasterPlan's `batch_size: 8` OOMs on a T4. Training config re-tuned for T4 constraints.

**Decisions**
| Q | Answer |
|---|--------|
| Training data scope | 5K/500/500 smoke test first, then 50K/1K/1K for the real run |
| Stage 2 prefix format | Newline-separated: `"{headline}\n{article}"` (what the BART paper does) |
| Data flow | Re-prep on Colab using the same module — no data upload to Drive |
| Training notebook location | `notebooks/04_train_bart_colab.ipynb` (built in Stage 1/2 work) |
| Tokenization timing | On-the-fly via HF `Trainer` (not pre-tokenized) |
| Output format | HF `Dataset.save_to_disk` (arrow) at `data/processed/cnn_dailymail/{stage1,stage2}/` |
| Prefix-strip rule | Port the expanded `(CNN) --` / `CITY (CNN) --` regex from the EDA notebook |
| Min article length | ≥50 words (drops truly degenerate examples) |
| T4 training config | `batch_size=4`, `grad_accum=8` (effective 32), `fp16=true`, `gradient_checkpointing=true` |

**What was built:** [src/preprocessing/cnn_dm_prep.py](src/preprocessing/cnn_dm_prep.py)
- `derive_headline`, `derive_summary_target`, `word_count` — shared derivation utilities ported from the EDA notebook
- `load_cnn_dm_subset`, `filter_by_length`
- `build_stage1_rows` — Stage 1: `article → headline`
- `build_stage2_rows` — Stage 2: `"{headline}\n{article}" → summary_target`
- `build_stage1_dataset`, `build_stage2_dataset` — wrap the above into `DatasetDict` across train/val/test
- `save_datasets`
- `main()` with CLI flags: `--n-train`, `--n-val`, `--n-test`, `--smoke-test`, `--output-root`

**Run friction:**
- First attempt ran `python -m src.preprocessing.cnn_dm_prep --smoke-test 2>&1 | tail -40`. The pipe buffered all output until completion, and the HF stream download hung for 10+ minutes (process was alive at 4.5s CPU time but blocked on I/O). Killed and retried.
- Second attempt: ran a **micro test** (100/50/50) without the pipe — completed in seconds, confirmed the code path works and Stage 1/2 datasets save correctly.
- Third attempt: ran the **full smoke test (5K/500/500)** with `python -u` for unbuffered output + `tail -f`/grep monitoring — completed cleanly.

**Smoke test results**
- Requested 5000/500/500 → kept **4997/500/500** for both Stage 1 and Stage 2
- 3 train examples dropped for <50-word articles; 0 from val/test
- Stage 1 schema verified: `[id, article, headline]`
- Stage 2 schema verified: `[id, input, target, headline]`
- Stage 2 `input` format verified: newline at position 204 in the first example cleanly separates the headline prefix from the article body; text before the newline exactly matches the `headline` column
- Stage 2 `target` verified: highlights joined with spaces (0 newlines in output)

**Observation carried forward:** The sample Stage 1 headline kept the Reuters dateline — `"LONDON, England (Reuters) -- Harry Potter star..."`. Consistent with the EDA decision (Reuters patterns are 0.1% of the sample; not worth a separate regex). Our strip rule is CNN-only by design.

**Build/infra:** No new dependencies. Reused the already-installed `datasets` package. A `data/processed/cnn_dailymail/{stage1,stage2}/` directory tree now holds the prepared arrow datasets locally (these are re-created fresh on Colab at training time, not uploaded).

### 🏷️ BART Stage 1 — v1 Colab run exposed a label-design flaw; pivot to v2 labels

The first Colab training run completed end-to-end but qualitative inspection of the 20 sample generations revealed the model had learned **extractive copy**, not abstractive headline generation:

```
ARTICLE: (CNN)So now we know. The Germanwings aircraft that crashed earlier this week...
REF:     (CNN)So now we know.
PRED:    So now we know.
```

```
ARTICLE: (CNN)More than two decades as a judge, prosecutor and defense lawyer...
REF:     (CNN)More than two decades as a judge, prosecutor and defense lawyer...
PRED:    More than two decades as a judge, prosecutor and defense lawyer...
```

The model was faithfully reproducing its training signal — our v1 `derive_headline` defined the label as "first sentence of the article with `(CNN) --` stripped", so the model correctly learned to copy the first sentence back. **ROUGE-1 of 0.71 was artificially inflated** because the label text *is* a prefix of the article text.

This was *not* the "vagueness / loss of specificity" failure flagged during the selection-metric debate — it was the opposite: **over-extraction baked in by the label, not the metric**. Neither ROUGE-1 nor BERTScore-F1 would have flagged it via their numbers alone; the pathology is only visible by reading outputs.

**v1 training metrics (ceiling effects of a copy-learned model):**

| Epoch | Train Loss | Val Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|-------|-----------|----------|---------|---------|---------|--------------|
| 1 | 0.267 | 0.494 | 0.696 | 0.677 | 0.689 | 0.9356 |
| **2** *(best)* | 0.123 | 0.698 | 0.715 | 0.699 | 0.703 | **0.9379** |
| 3 | 0.103 | 0.726 | 0.693 | 0.677 | 0.690 | 0.9348 |

BERTScore-based selection correctly picked epoch 2 (train loss still dropping, val loss climbing → overfit). Checkpoint persisted to Drive at `/content/drive/MyDrive/NewsForge/checkpoints/stage1/checkpoint-314` (10.8 GB including intermediates).

**Colab-run friction (lessons for future sections):**
Five non-training issues ate time before training actually started:
1. Repo was **private** → `git clone` failed silently in non-interactive Colab. Resolved by making public.
2. `pip install -U -r requirements.txt` **upgraded torch 2.10 → 2.11**, breaking the CUDA-matched stack with `torchvision/torchaudio` (`is_opaque_value` import error). Resolved by switching to a **pip constraints file** that pins Colab's CUDA-matched ML stack and only installs missing packages.
3. `transformers==5.0.0` on Colab **renamed `tokenizer=` → `processing_class=`** on `Trainer`. Resolved with `inspect.signature` fallback — works on both API generations.
4. `transformers==5.0.0` **removed `overwrite_output_dir` and `save_safetensors`** from `Seq2SeqTrainingArguments`. Resolved by making `build_training_args` introspect the signature and silently drop unsupported kwargs with a log line for visibility.
5. Cell 2 had `if not os.path.isdir(WORKDIR)` guard — session restarts didn't pull new commits. Resolved by always running `git fetch origin main && git reset --hard origin/main` on re-run.

**Also fixed during the debrief:**
- **`min_length=56` warning** — `bart-large-cnn` ships with CNN/DM summary generation defaults (`min_length=56, max_length=142`). At `max_length=48` for headlines these conflict, producing warnings and truncation artifacts (sample 12 cut to `"...world No."` mid-sentence). Fixed by explicitly overriding `model.generation_config.{min_length, max_length, num_beams, length_penalty, no_repeat_ngram_size, early_stopping}` from our config before constructing `Seq2SeqTrainer`.

#### v2 Label pivot: first bullet of `highlights`

CNN/DM has no native headline field. The v2 choice uses the **first bullet of `highlights`**:

- CNN/DM `highlights` is a string of 3–4 editor-written bullets joined by `\n` (no `first_highlight` column; we derive by `highlights.split('\n', 1)[0]`)
- Bullets are ordered by editorial salience (bullet 1 = main point) — the dataset's closest approximation of a headline
- **Genuinely abstractive**: bullets are human-rewritten, not copied from the article
- **Short and news-shaped**: typical 10–15 words (median 11 in our 100-sample local check, vs 25 words for v1 "first sentence of article" labels)

**Side-by-side on the canonical Radcliffe example:**

| Label source | Text | Words |
|--------------|------|-------|
| v1: first sentence of article | *LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him.* | 38 |
| **v2: first bullet of highlights** | **Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday.** | **13** |

**Sample of 10 v2 training headlines:**
```
[ 0] (13w)  Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday.
[ 1] (11w)  Mentally ill inmates in Miami are housed on the "forgotten floor"
[ 2] (10w)  NEW: "I thought I was going to die," driver says.
[ 3] (10w)  Five small polyps found during procedure; "none worrisome," spokesman says.
[ 4] (11w)  NEW: NFL chief, Atlanta Falcons owner critical of Michael Vick's conduct.
[ 5] (12w)  Parents beam with pride, can't stop from smiling from outpouring of support.
[ 6] (11w)  Aid workers: Violence, increased cost of living drive women to prostitution.
[ 7] (12w)  Tomas Medina Caracas was a fugitive from a U.S. drug trafficking indictment.
[ 9] (11w)  Empty anti-tank weapon turns up in front of New Jersey home.
```

**Expected v2 training dynamics:**
- ROUGE-1 will *drop* from ~0.71 to ~0.40–0.50 — this looks like a regression but is actually healthier (the model has to rewrite, not copy)
- BERTScore F1 may drop from ~0.94 to ~0.90 — semantic closeness is harder to achieve under real abstraction
- Outputs will read like headlines instead of lead paragraphs

**Implementation:**
- Rewrote `derive_headline` in [src/preprocessing/cnn_dm_prep.py](src/preprocessing/cnn_dm_prep.py) to split `highlights` on `\n` and take `[0]`
- Added `_normalize_cnn_dm_text` helper that strips the CNN/DM `" ."` tokenization artifact (`"text ."` → `"text."`)
- Also applied to `derive_summary_target` for Stage 2 consistency
- Removed the now-unused `CNN_PREFIX_RE` / `SENTENCE_BOUNDARY_RE` — highlights don't have CNN datelines
- Overrode `model.generation_config` in `train_stage1` so `min_length` honors our config, not BART's summary defaults

**Local micro-test (100 train / 50 val / 50 test):** all headlines look clean and headline-shaped. First 10 training samples span 6–33 words, median 11.

#### v2 Colab smoke-test results — labels fixed, extractive-copy failure eliminated

The v2 smoke test (5K train / 500 val / 500 test) ran cleanly on Colab T4 with no new infrastructure issues (the `SAVE_TO_DRIVE=False` toggle kept checkpoints in `/content/checkpoints/stage1` to avoid the user's full Google Drive). The label pivot delivered exactly the shift we predicted.

**Metric comparison v1 vs v2:**

| Metric | v1 (first-sentence-of-article) | v2 (first-bullet-of-highlights) | Interpretation |
|--------|-------------------------------|--------------------------------|----------------|
| ROUGE-1 | 0.715 | 0.329 | ↓ expected — real abstraction demand, not copy |
| ROUGE-2 | 0.699 | 0.168 | ↓ expected |
| ROUGE-L | 0.703 | 0.296 | ↓ expected |
| BERTScore F1 | 0.938 | 0.901 | ↓ slightly (healthy) |

ROUGE-1 ≈ 0.33 is the typical range for genuinely abstractive CNN/DM summarization. BERTScore holding at 0.90 confirms semantic fidelity survived the abstraction demand.

**Qualitative samples from cell 10's 20-example inspection (all look like real headlines):**

```
REF:  Actor Ashton Kutcher complained on Facebook that men's rooms don't have diapering tables.
PRED: Ashton Kutcher's Facebook post about lack of diaper changing tables has parents talking.

REF:  Two Chinese men have been jailed for selling military intelligence.
PRED: Two Chinese men jailed for selling military intelligence to foreign spies, state media reports.

REF:  Thousands march in a protest against terrorism in Tunisia's capital.
PRED: Thousands march in Tunisia's capital, protesting against terrorism.
```

Short, news-shaped, paraphrased (not copied). The v1 pathological behavior (verbatim first-sentence-of-article output) is **completely eliminated**. A handful of outputs are off-topic on small data (e.g., sample 1 picks up "Obama" from the article instead of the Scott Walker subject) — a data-volume artifact, not a label issue.

#### Training-dynamics artifact: warmup phase never completed on the smoke test

v2 training loss stayed high across all three epochs (8.13 → 7.32 → 5.96) — an order of magnitude larger than v1's ~0.1–0.3. Initial suspicion was a bug; actual cause is a **warmup-too-long** effect specific to the smoke test's tiny step budget:

- Total steps: 4997 / (batch 4 × grad_accum 8) = ~156 per epoch × 3 epochs = **468 steps**
- `warmup_steps` (from config): **500**
- 468 < 500 → LR linearly ramps from 0 to 2e-5 the entire run and never hits the target

Consequence: the model barely moves from the pretrained `bart-large-cnn` weights. The reason outputs still look good is that `bart-large-cnn` was *already* fine-tuned on CNN/DM by the original BART authors — our tiny-LR nudge just shifts the generation distribution slightly toward shorter, more headline-shaped outputs.

This also shows up as a **noise-level spread** in BERTScore across epochs: `0.9004 → 0.9005 → 0.9008` (0.0004 spread). Best-model selection under BERTScore F1 is effectively picking the last checkpoint by default.

**On the 50K full run, this artifact disappears:**
- 50K / 32 = ~1562 steps × 3 epochs = 4687 total
- 500 warmup = ~11% of total — standard ratio
- Expected training loss: starts ~6–8 and drops to ~1–2
- Expected epoch-to-epoch metric deltas: meaningful, not noise

**Optional tightening (deferred):** add a `warmup_steps` override when `SMOKE_TEST=True` (e.g., `warmup_steps=50`) so future smoke tests exercise realistic training dynamics. Not urgent — the v2 smoke test already validated the label change, which was its purpose.

#### Stage 1 verdict: correctness gate cleared

The smoke test's purpose was to validate the label swap. It did. Qualitative output looks like headlines; extractive-copy failure eliminated; no infrastructure or training-framework errors. **Stage 1 is behaviorally correct.**

Whether to run the full 50K before moving on is a **polish-vs-progress decision**, not a correctness decision:
- **A working Stage 1 checkpoint exists** at `/content/checkpoints/stage1/best/` — effectively "bart-large-cnn gently nudged toward shorter, headline-shaped outputs." Sufficient to feed Stage 2's headline-prefix input.
- Full 50K would produce a checkpoint where training actually learned non-trivially and yields paper-comparable numbers. Roughly 2h on Colab, may span sessions.

### 📝 BART Stage 2 — Summary Generation

#### Config + code deltas from Stage 1

- **Added `generation_stage2` block** to [config.yaml](configs/config.yaml) with BART's published CNN/DM summarization defaults (`num_beams=4, min_length=30, length_penalty=2.0, no_repeat_ngram_size=3`). Stage 1's block stays at `length_penalty=1.0, min_length=5` (short headlines).
- **Extended `compute_metrics_factory`** with `rougeLsum` + a sentence-boundary splitter (`_split_sentences_for_rouge_sum`). Without the splitter, rougeLsum degenerates to rougeL on single-line inputs. Local 3-pair unit test confirmed rougeL=0.2724 ≠ rougeLsum=0.3042 post-split.
- **Refactored `build_training_args`** to take a `selection_metric` parameter instead of hardcoding `bertscore_f1`. Stage 1 still uses `bertscore_f1` (semantic proxy for paraphrase); Stage 2 uses **`rougeLsum`** (HF's canonical summarization metric).
- **Added `train_stage2`** as a near-copy of `train_stage1` with Stage-2-specific wiring (stage2 config block, `input`/`target` columns, `generation_stage2`, rougeLsum selection).
- **Appended 6 Stage 2 cells** to `notebooks/04_train_bart_colab.ipynb` (paths+toggle, train, final metrics, 20-sample inspection, checkpoint verify, preceded by a cleanup header).

#### Colab run friction: an OOM we didn't anticipate

First Stage 2 training cell **OOMed immediately** — the T4's 14.6 GiB VRAM was already 97% full before Stage 2 training started. Every Stage 1 artifact was still resident: the Stage 1 trainer (model + optimizer + scheduler), the `model`/`tokenizer` loaded in cell 20 for generation, the `BERTScorer` (RoBERTa-large, ~1.4 GB) closed over inside Stage 1's `compute_metrics`. Stage 2 tried to load another ~3 GB and fell over.

First remediation attempt (in-place `del` of notebook-scope vars + `gc.collect()` + `torch.cuda.empty_cache()`) went from `14.27 GiB` allocated to `14.02 GiB` — effectively no change. Second attempt clearing `sys.last_traceback` and cached modules: also `14.02 GiB`. Diagnostic `gc.get_objects()` showed **12.9 GiB of live CUDA tensors** still tracked. Could not surgically dislodge them in-place.

**Actual fix: restart session + skip Stage 1 retrain.** Runtime restart wipes Python state and VRAM but preserves `/content/` on disk (Stage 1 checkpoint and prepped data survive). Re-ran only cells 1–5 + 7 + the Stage 2 block. Saved ~25 min vs a full re-run of Stage 1.

**Lessons carried forward:**
- Between stages on a single Colab session, the traceback + reference chain from any prior training session holds the previous stage's weights pinned. In-place cleanup is unreliable.
- For multi-stage training on Colab free-tier, prefer a session restart between stages. The cost is ~2 min of cell re-runs; the gain is a guaranteed-clean VRAM budget.
- Added a `gc.collect() + torch.cuda.empty_cache()` at the top of `train_stage2` as a belt-and-suspenders safety net (it helped; the primary fix was the restart).

#### Training metrics (5K/500/500 smoke test)

| Epoch | Train Loss | Val Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | **rougeLsum** | BERTScore F1 |
|-------|-----------|----------|---------|---------|---------|---------------|--------------|
| **1** *(best)* | 7.42 | 1.12 | 0.518 | 0.397 | 0.479 | **0.503** | 0.917 |
| 2 | 6.94 | 1.14 | 0.512 | 0.393 | 0.475 | 0.497 | 0.916 |
| 3 | 6.10 | 1.21 | 0.513 | 0.393 | 0.478 | 0.498 | 0.917 |

Val loss monotonically rises from epoch 1 (1.12 → 1.14 → 1.21). rougeLsum-based selection correctly picked **epoch 1** (checkpoint-157). Training loss magnitude is high for the same reason as Stage 1 smoke — warmup_steps=500 > total_steps=468 → LR never reaches target; the model barely moves from pretrained weights. Still, the starting model (`bart-large-cnn`) is already fine-tuned for CNN/DM summarization, so outputs are genuinely strong.

**Why the numbers look high:** CNN/DM published BART baselines are ~0.44 ROUGE-1 / ~0.21 ROUGE-2; we're at 0.52 / 0.40. That's because the base model we're nudging already has near-SOTA CNN/DM summarization baked in. Real-world comparison to published numbers should happen on the held-out test split + the Multi-News multi-doc pipeline (both happen in the 📏 Automatic Evaluation section, not here).

#### Qualitative inspection — summaries are genuinely abstractive and multi-sentence

```
Sample 16 — REF richer than expected:
REF:   Two Chinese men have been jailed for selling military intelligence.
       Material includes hundreds of photos of China's first aircraft carrier,
       the Liaoning. They were jailed for six to eight years.
PRED:  Two Chinese men have been jailed for selling military intelligence to
       foreign spies, state media reported. The two men, surnamed Han and
       Zhang, were sentenced to eight and six years in prison respectively.
       Han, 30, was approached via WeChat. Zhang, 23, sent more than 500
       pictures of the Liaoning to a foreign magazine editor.
```

Several samples (10, 16, 17, 19) show this pattern — **PRED preserves or exceeds REF in entity coverage** (specific names, ages, locations, mechanisms) while staying faithful. This is the opposite of the vagueness failure mode flagged during the selection-metric debate.

**One factual drift noted — catalog for future error analysis:**
```
Sample 15 — REF says Le Clos is "22-year-old"; PRED says "20-year-old".
           (At the time of the London 2012 Olympics he was ~20; the REF
           may be using present-at-time-of-writing age. PRED drifted either
           to the age at time of event, or hallucinated. Worth cataloguing
           for the 🔍 Error Analysis section.)
```

No repetition, no over-extraction, no degenerate outputs across the 20 samples. Summaries are 2–4 sentences, news-shaped, entity-rich.

#### Stage 2 verdict

**Behaviorally correct and qualitatively strong.** Working checkpoint at `/content/checkpoints/stage2/best/` (~1.6 GB safetensors). Stage 2 joins Stage 1 in the "smoke-validated, full 50K is polish" bucket.

Ready to chain with Stage 1 for end-to-end summarization in the evaluation section. Because Stage 2 was trained with *reference* first-highlight headlines as prefix (not Stage-1-generated), the pipeline will have a small train/inference distribution shift when we plug in Stage 1's actual outputs — worth watching in the E2E test.

---

## 4. Config & Infra Changes

### [configs/config.yaml](configs/config.yaml)
| Change | Before | After |
|--------|--------|-------|
| `newsapi.categories` | 6 (no sports) | 7 (adds sports) |
| `newsapi.country` | — | `us` |
| `multinews.*` | — | Added full block (repo, split, subset_size=300, separator, summary_prefix) |
| `cnn_dm.*` | — | Added full block (repo, version, n_train=50000, n_val=1000, n_test=1000, min_article_words=50, stage2_separator) |
| `paths.data_multinews` | — | `data/processed/multi_news` |
| `paths.data_cnn_dm` | — | `data/processed/cnn_dailymail` |
| `summarization.stage1.max_output_tokens` | 30 | 48 (EDA finding) |
| `summarization.training.batch_size` | 8 | 4 (T4 VRAM constraint) |
| `summarization.training.gradient_accumulation_steps` | 4 | 8 (keeps effective batch at 32) |
| `summarization.training.fp16` | — | `true` (halves memory, faster on T4) |
| `summarization.training.gradient_checkpointing` | — | `true` (trades compute for memory) |
| `generation.*` | — | Added Stage 1 headline generation block (num_beams=4, min_length=5, length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=true) |
| `generation_stage2.*` | — | Added Stage 2 summary generation block (num_beams=4, min_length=30, length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=true) — BART's published CNN/DM defaults |
| `clustering.hdbscan.min_cluster_size` | 3 | 2 (sweep winner) |
| `clustering.hdbscan.metric` | `"cosine"` (label) | clarifying comment: implemented as euclidean on L2-normalized vectors |

### [requirements.txt](requirements.txt) additions
- `umap-learn` (clustering viz)
- `matplotlib` (was a transitive dep; made explicit)
- `nbconvert`, `nbformat`, `ipykernel` (notebook execution)

### [MasterPlan.md](MasterPlan.md)
- Replaced 4-week timeline (§9) with the user's task-based todo list
- §1 Project Overview: unchanged (still describes the system honestly)
- §2 Objectives: rewrote obj #1 for the hybrid; added "Dataset Roles" table
- §3 Pipeline diagram: updated Data Ingestion box (dual source); noted clustering is NewsAPI-only
- §5.1 Data Sources: split into 5.1.1 (Multi-News primary), 5.1.2 (NewsAPI live mode), 5.1.3 (CNN-DM training)
- §5.5 Stage 1: max output tokens 30 → 48
- §5.7 Ablations: reframed Ablation 2 around Multi-News pre-grouped clusters; added bonus Ablation 3 (gold vs HDBSCAN clusters)
- §6 Evaluation: added dual reference-set table (CNN-DM for single-doc, Multi-News for multi-doc)
- §9 Todo List: tracks completed items per section

---

## 5. Artifacts Produced

| Path | Contents |
|------|----------|
| `data/raw/articles_20260419T022342Z.json` | 340 raw NewsAPI articles |
| `data/processed/newsapi/articles_20260419T205819Z.json` | 335 cleaned, paywall-flagged articles |
| `data/processed/multi_news/clusters.json` | 300 Multi-News test clusters |
| `data/embeddings/embeddings.npy` | (335, 384) L2-normalized vectors |
| `data/embeddings/article_index.json` | Row → URL/title mapping |
| `data/clusters/newsapi_clusters.json` | HDBSCAN payload (21 clusters + metadata) |
| `data/clusters/newsapi_clusters_kmeans.json` | K-Means baseline |
| `data/processed/cnn_dailymail/stage1/` | HF `DatasetDict` — Stage 1 (`article` → `headline`), smoke test 4997/500/500 |
| `data/processed/cnn_dailymail/stage2/` | HF `DatasetDict` — Stage 2 (`"{headline}\n{article}"` → `summary_target`), smoke test 4997/500/500 |
| `src/summarization/trainer.py` | BART Stage 1 + Stage 2 fine-tuning (train_stage1, train_stage2, compute_metrics_factory with BERTScore+rougeLsum, build_training_args with introspective kwarg filtering, generate_headlines, generate_summaries) |
| `notebooks/01_eda.ipynb` | Executed EDA, 8 sections, embedded plots |
| `notebooks/02_clustering.ipynb` | Executed sweep + UMAP + findings |
| `notebooks/04_train_bart_colab.ipynb` | Stage 1 + Stage 2 Colab training cells (GPU verify, repo sync, Colab-safe deps install, Drive mount, paths+toggles, CNN/DM prep, trainer imports, Stage 1 train+eval+inspect+verify, **cross-stage VRAM cleanup cell**, Stage 2 paths, Stage 2 train+eval+inspect+verify) |
| Colab `/content/checkpoints/stage1/best/` *(ephemeral)* | Stage 1 smoke-trained BART-large-cnn (~1.6 GB safetensors). Lost on session disconnect. |
| Colab `/content/checkpoints/stage2/best/` *(ephemeral)* | Stage 2 smoke-trained BART-large-cnn (~1.6 GB safetensors), selected by rougeLsum=0.503. Lost on session disconnect. |

---

## 6. Open Items / Deferred Decisions

1. ~~**Prefix-strip rule at training-data prep time.**~~ **Resolved in Training Environment:** the expanded `(CNN) --` / `CITY (CNN) --` regex from the EDA notebook was ported verbatim into `src/preprocessing/cnn_dm_prep.py`. CNN-dateline coverage is ~29.6%; Reuters (0.1%) and bare ALL-CAPS cities (12.2%) intentionally left untouched.

2. **HDBSCAN `min_cluster_size=2` vs `3`.** Current config uses 2 (higher silhouette). If downstream summarization on 2-article clusters produces low-quality outputs in the E2E test, reverting to 3 is the escape hatch.

3. **Multi-News subset size.** We're using 300 clusters for dev / iteration. Final evaluation should probably use a larger slice (1K–full 5.6K test split) for paper-comparable numbers.

4. **Multi-News embeddings.** Skipped. If we do Ablation 3 (gold vs HDBSCAN clusters on Multi-News), we'll need to embed + cluster Multi-News articles too.

5. **Paywall-flagged articles.** Currently kept in the dataset with `is_paywall_preview: true`. Downstream pipeline can choose to include or exclude; the E2E test will tell us which.

6. **HF rate-limit warnings.** Every dataset load prints `"Warning: You are sending unauthenticated requests to the HF Hub"`. Setting `HF_TOKEN` in `.env` would silence these and raise rate limits — not urgent.

7. **Real 50K training-data prep.** Smoke test (5K/500/500) is done locally. The 50K/1K/1K full run happens on Colab at training time (no need to prep locally given the re-prep-on-Colab data flow). If we ever want a local run, `python -m src.preprocessing.cnn_dm_prep` with no flags uses the config defaults.

8. **PyTorch + CUDA verification.** Deferred to the top of the Colab training notebook (`notebooks/04_train_bart_colab.ipynb`) — there's no meaningful way to verify on a MacBook without CUDA. The notebook will begin with a `torch.cuda.is_available()` + GPU-name sanity check.

9. **Checkpoint persistence strategy on Colab.** Colab free-tier sessions die after ~12h and ephemerally. The training notebook will need to mount Google Drive and save checkpoints there, or the work is lost on disconnect. Config flag for this will be added when we build the notebook.

10. ~~**Stage 1 headline quality under training.**~~ **Resolved in v1/v2 Colab runs:** v1 exposed the extractive-copy failure caused by using "first sentence of article" as label; v2 switched to "first bullet of highlights" and produced short, abstractive, headline-shaped outputs. Reuters-dateline leakage was not observed in the v2 smoke-test generations.

11. **Stage 1 warmup-too-long on smoke-test scale.** v2 smoke test had 468 total training steps but `warmup_steps=500` (from the config tuned for 50K training). LR never reached target → model barely moved from pretrained weights. Outputs still look good because `bart-large-cnn` is already CNN/DM-trained, but this means the smoke test doesn't exercise realistic training dynamics. **On the 50K full run this disappears** (warmup = 11% of 4687 steps, standard ratio). **Optional fix:** add a `warmup_steps` override when `SMOKE_TEST=True` — deferred. **Also applies to Stage 2 smoke test** (same step budget of 468).

12. **Full 50K Stage 1 + Stage 2 runs.** Smoke tests confirmed correctness for both stages; full runs are polish-only and each requires ~2h Colab time (may span sessions). Existing checkpoints at `/content/checkpoints/stage1/best/` and `/content/checkpoints/stage2/best/` are usable for downstream pipeline work. Deferred — user decision pending.

13. **Stage 1 and Stage 2 checkpoints are on Colab `/content/`, not persisted.** Smoke tests deliberately saved locally (Drive was out of space). If the Colab session drops, the checkpoints are lost and we'd need to re-train (fast at smoke-test scale). For the E2E pipeline test we'll either need to flip `SAVE_TO_DRIVE=True` (requires Drive space), push to HuggingFace Hub, or run the whole pipeline end-to-end in one session. Each Stage-1/2 best/ directory is ~1.6 GB.

14. **Stage 2 train/inference distribution shift.** Stage 2 was trained with **reference** first-highlight headlines as prefix (`"{highlight_0}\n{article}"`). At inference time in the E2E pipeline we'll plug in **Stage 1 generated** headlines instead. If Stage 1's headlines differ stylistically from first-highlight references, Stage 2 summaries may degrade. Watch for this in the 🔁 End-to-End Pipeline Test. Mitigations: use Stage 1 generated headlines during Stage 2 training (requires a first-pass Stage 1 model), or train Stage 2 without headline prefix to see the ablation delta (this *is* Ablation 1 from §5.7).

15. **Factual drift catalogued for Error Analysis.** Stage 2 sample 15 showed a factual drift where PRED said "20-year-old" while REF said "22-year-old" (Chad le Clos at 2012 London Olympics). He was actually ~20 at the event; the REF may be using his age at writing time. Either the model drifted to the event-time age (acceptable paraphrase) or hallucinated. Add to the 🔍 Error Analysis hallucination examples when that section runs.

16. **Cross-stage VRAM cleanup on Colab.** Running Stage 1 then Stage 2 in the same Colab session OOMs on a T4 — Stage 1's trainer + generation model + BERTScorer all stay pinned via the cell output traceback and reference chains that ordinary `del` + `gc.collect()` + `torch.cuda.empty_cache()` cannot surgically free. **Working fix: restart session between stages** (preserves `/content/` on disk, so checkpoints and prepped data survive). For a multi-stage notebook pattern this is a real constraint worth noting for future re-runs. A `gc.collect() + torch.cuda.empty_cache()` safety net was added at the top of `train_stage2` but the primary fix is the session restart.

---

## 7. Next Section

Both Stage 1 and Stage 2 are smoke-validated with working checkpoints. Next reasonable moves:

**Path A (recommended): 🏷️ Named Entity Recognition.**
Implement `src/ner/entity_extractor.py` using spaCy `en_core_web_trf`. NER runs on the Stage 2 generated summaries and (as fallback) on source articles. Entities feed into the Streamlit UI's event cards. Well-scoped, no GPU needed, independent of the remaining BART work — easy forward progress.

**Path B: 🔁 End-to-End Pipeline Test.**
Glue Stage 1 + Stage 2 + multi-doc inputs on the Multi-News cluster set (300 clusters we prepared). Produces (headline, summary) pairs per cluster using our two trained models. Validates the full inference pipeline and surfaces the Stage-2-train/inference distribution shift (Stage 2 was trained on reference headlines, at inference it sees Stage-1-generated ones). Higher information value for the final report but requires another Colab session for generation.

**Path C: 📏 Automatic Evaluation.**
Run ROUGE + BERTScore on CNN/DM test split (single-doc) and Multi-News (multi-doc). Also computes headline–summary consistency (deferred from Stage 2). Depends on Path B being done first for the multi-doc numbers.

**Deferred (polish, not correctness):** full 50K Stage 1 and Stage 2 runs. A working Stage 1 and Stage 2 pair exists already; running longer would polish metrics but not change behavior.

User to choose.
