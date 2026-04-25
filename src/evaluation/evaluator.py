"""
Automatic Evaluation Module
============================
Pipeline Stage: 📏 Automatic Evaluation (Section 6.1)

Computes the metrics that go in the final report from saved JSONs:
- **Single-doc** test metrics on CNN/Daily Mail (computed on Colab during
  `train_stage1`/`train_stage2`; reported here)
- **Multi-doc** metrics on Multi-News E2E pipeline outputs (ROUGE + BERTScore
  on summary vs reference summary)
- **Headline–summary consistency** (BERTScore F1 between generated headline
  and generated summary) on both CNN/DM test (chained) and Multi-News E2E
- **Final summary table** comparing single-doc vs multi-doc

Design split:
- The expensive work (generation) happens on Colab (`trainer.evaluate` on test
  split + `summarizer.run_*_pipeline` for chained CNN/DM and Multi-News E2E).
- This module loads the resulting JSONs and computes the additional metrics
  (BERTScore, consistency, ROUGE on E2E outputs) — all CPU-bound, runs
  locally in `notebooks/03_evaluation.ipynb`.

Multi-News headline-vs-reference is intentionally **NOT** scored: Multi-News
has no native headline labels and our `reference_headline` field is the first
sentence of the reference summary, which is a circular comparison target.
We do report headline-summary consistency, which is intrinsic and reference-free.
"""

import argparse
import json
import os
import re
from typing import Any

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- Sentence splitting (matches trainer.py for rougeLsum consistency) ----------

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _sentence_split_for_rouge_sum(text: str) -> str:
    """Split on sentence boundaries and rejoin with newlines (rougeLsum convention)."""
    if not text:
        return ""
    sentences = _SENTENCE_BOUNDARY_RE.split(text.strip())
    return "\n".join(s.strip() for s in sentences if s.strip())


# --- Metric primitives ----------------------------------------------------------


def compute_rouge(
    predictions: list[str],
    references: list[str],
    rouge_types: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute mean F-measure for each requested ROUGE variant.

    For `rougeLsum`, predictions and references are sentence-split and
    newline-joined first (otherwise rougeLsum degenerates to rougeL).
    """
    from rouge_score import rouge_scorer

    rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    need_split = "rougeLsum" in rouge_types

    totals = {rt: 0.0 for rt in rouge_types}
    n = max(len(predictions), 1)
    for pred, ref in zip(predictions, references):
        if need_split:
            pred_in = _sentence_split_for_rouge_sum(pred)
            ref_in = _sentence_split_for_rouge_sum(ref)
        else:
            pred_in, ref_in = pred, ref
        scores = scorer.score(ref_in, pred_in)
        for rt in rouge_types:
            totals[rt] += scores[rt].fmeasure
    return {rt: totals[rt] / n for rt in rouge_types}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "roberta-large",
) -> dict[str, float]:
    """Mean BERTScore precision / recall / F1 over the prediction-reference pairs."""
    from bert_score import score as bert_score_fn

    p, r, f1 = bert_score_fn(
        predictions, references, model_type=model_type, lang="en", verbose=False
    )
    return {
        "bertscore_p": float(p.mean().item()),
        "bertscore_r": float(r.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }


def compute_consistency(
    headlines: list[str],
    summaries: list[str],
    bertscore_model: str = "roberta-large",
) -> dict[str, float]:
    """
    Headline–summary consistency: BERTScore F1 treating the headline as the
    prediction and the summary as the reference. Reference-free; measures
    whether the two outputs agree with each other.
    """
    return {
        f"consistency_{k}": v
        for k, v in compute_bertscore(headlines, summaries, model_type=bertscore_model).items()
    }


# --- Aggregators for saved JSON artifacts ----------------------------------------


def evaluate_e2e_results(
    e2e_results: list[dict[str, Any]],
    bertscore_model: str = "roberta-large",
) -> dict[str, Any]:
    """
    Compute summary-vs-reference metrics + headline-summary consistency on
    Multi-News E2E pipeline outputs.

    Headline-vs-reference is intentionally skipped — Multi-News has no native
    headline label; our `reference_headline` is derived from the reference
    summary and would produce a circular score.
    """
    summaries = [r["generated_summary"] for r in e2e_results]
    refs = [r["reference_summary"] for r in e2e_results]
    headlines = [r["generated_headline"] for r in e2e_results]

    summary_rouge = compute_rouge(summaries, refs)
    summary_bert = compute_bertscore(summaries, refs, model_type=bertscore_model)
    consistency = compute_consistency(headlines, summaries, bertscore_model=bertscore_model)

    return {
        "n_clusters": len(e2e_results),
        "summary_vs_reference": {**summary_rouge, **summary_bert},
        "headline_summary_consistency": consistency,
    }


def evaluate_cnn_dm_chained(
    chained_results: list[dict[str, Any]],
    bertscore_model: str = "roberta-large",
) -> dict[str, Any]:
    """
    Compute single-doc metrics on the chained CNN/DM test pass.

    Reports:
      headline_vs_reference: Stage 1 generated headline vs CNN/DM derived headline
                             (first-bullet-of-highlights label)
      summary_vs_reference:  Stage 2 generated summary vs joined-highlights target
      headline_summary_consistency: BERTScore between the two generated outputs
    """
    headlines_pred = [r["generated_headline"] for r in chained_results]
    headlines_ref = [r["reference_headline"] for r in chained_results]
    summaries_pred = [r["generated_summary"] for r in chained_results]
    summaries_ref = [r["reference_summary"] for r in chained_results]

    headline_rouge = compute_rouge(headlines_pred, headlines_ref)
    headline_bert = compute_bertscore(headlines_pred, headlines_ref, model_type=bertscore_model)
    summary_rouge = compute_rouge(summaries_pred, summaries_ref)
    summary_bert = compute_bertscore(summaries_pred, summaries_ref, model_type=bertscore_model)
    consistency = compute_consistency(
        headlines_pred, summaries_pred, bertscore_model=bertscore_model
    )

    return {
        "n_examples": len(chained_results),
        "headline_vs_reference": {**headline_rouge, **headline_bert},
        "summary_vs_reference": {**summary_rouge, **summary_bert},
        "headline_summary_consistency": consistency,
    }


# --- Final summary table -------------------------------------------------------


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def compile_final_table(
    stage1_test_metrics: dict[str, Any] | None,
    stage2_test_metrics: dict[str, Any] | None,
    cnn_dm_chained_eval: dict[str, Any] | None,
    multinews_eval: dict[str, Any] | None,
) -> str:
    """
    Render a markdown summary table comparing single-doc (CNN/DM) vs multi-doc
    (Multi-News) evaluation results.
    """
    lines: list[str] = []
    lines.append("# 📏 Automatic Evaluation Summary\n")

    # --- Stage 1 + Stage 2 individual test metrics (from trainer.evaluate) ---
    if stage1_test_metrics or stage2_test_metrics:
        lines.append("## Per-stage CNN/Daily Mail test metrics (from `trainer.evaluate`)\n")
        lines.append("| Metric | Stage 1 (headline) | Stage 2 (summary) |")
        lines.append("|--------|--------------------|-------------------|")
        keys = sorted(
            set((stage1_test_metrics or {}).keys())
            | set((stage2_test_metrics or {}).keys())
        )
        for k in keys:
            if not k.startswith("final_test_"):
                continue
            s1 = (stage1_test_metrics or {}).get(k, "—")
            s2 = (stage2_test_metrics or {}).get(k, "—")
            lines.append(f"| `{k}` | {_fmt(s1)} | {_fmt(s2)} |")
        lines.append("")

    # --- Chained CNN/DM (single-doc) ---
    if cnn_dm_chained_eval:
        lines.append("## Chained Stage 1 → Stage 2 on CNN/Daily Mail test (single-doc)\n")
        lines.append(f"_n = {cnn_dm_chained_eval.get('n_examples', '?')} examples_\n")
        for section in ("headline_vs_reference", "summary_vs_reference", "headline_summary_consistency"):
            block = cnn_dm_chained_eval.get(section, {})
            if not block:
                continue
            lines.append(f"### {section.replace('_', ' ').capitalize()}\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in block.items():
                lines.append(f"| `{k}` | {_fmt(v)} |")
            lines.append("")

    # --- Multi-News E2E (multi-doc) ---
    if multinews_eval:
        lines.append("## Multi-News end-to-end pipeline (multi-doc, primary result)\n")
        lines.append(f"_n = {multinews_eval.get('n_clusters', '?')} clusters_\n")
        for section in ("summary_vs_reference", "headline_summary_consistency"):
            block = multinews_eval.get(section, {})
            if not block:
                continue
            lines.append(f"### {section.replace('_', ' ').capitalize()}\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in block.items():
                lines.append(f"| `{k}` | {_fmt(v)} |")
            lines.append("")

    # --- Single-doc vs Multi-doc summary comparison ---
    if cnn_dm_chained_eval and multinews_eval:
        lines.append("## Single-doc (CNN/DM chained) vs Multi-doc (Multi-News E2E) — summary metrics\n")
        cnn_summary = cnn_dm_chained_eval.get("summary_vs_reference", {})
        mn_summary = multinews_eval.get("summary_vs_reference", {})
        keys = sorted(set(cnn_summary) | set(mn_summary))
        lines.append("| Metric | CNN/DM chained (single-doc) | Multi-News E2E (multi-doc) |")
        lines.append("|--------|------------------------------|----------------------------|")
        for k in keys:
            lines.append(
                f"| `{k}` | {_fmt(cnn_summary.get(k, '—'))} | {_fmt(mn_summary.get(k, '—'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def save_evaluation_report(report: dict[str, Any], output_path: str) -> str:
    """Save an evaluation report dict to JSON (creates parent dirs as needed)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path


# --- Convenience: load all artifacts + compute everything ----------------------


def run_full_evaluation(
    stage1_test_path: str | None,
    stage2_test_path: str | None,
    cnn_dm_chained_path: str | None,
    multinews_results_path: str | None,
    config: dict,
    output_dir: str = "data/processed/evaluation_outputs",
) -> dict[str, Any]:
    """
    Load saved JSONs from the Colab session, compute the metrics that need
    further processing (Multi-News + CNN/DM-chained ROUGE/BERTScore +
    consistency), and write a final report JSON + markdown summary table.
    """
    bertscore_model = config["evaluation"]["bertscore_model"]
    report: dict[str, Any] = {}

    # Per-stage test metrics from trainer.evaluate (already metrics-only).
    if stage1_test_path and os.path.exists(stage1_test_path):
        with open(stage1_test_path) as f:
            report["stage1_test_metrics"] = json.load(f)
    if stage2_test_path and os.path.exists(stage2_test_path):
        with open(stage2_test_path) as f:
            report["stage2_test_metrics"] = json.load(f)

    # Chained CNN/DM (predictions saved; metrics computed here).
    if cnn_dm_chained_path and os.path.exists(cnn_dm_chained_path):
        with open(cnn_dm_chained_path) as f:
            chained = json.load(f)
        report["cnn_dm_chained_eval"] = evaluate_cnn_dm_chained(
            chained, bertscore_model=bertscore_model
        )

    # Multi-News E2E (predictions saved; metrics computed here).
    if multinews_results_path and os.path.exists(multinews_results_path):
        with open(multinews_results_path) as f:
            mn = json.load(f)
        report["multinews_eval"] = evaluate_e2e_results(
            mn, bertscore_model=bertscore_model
        )

    # Render summary table.
    report["summary_table_md"] = compile_final_table(
        report.get("stage1_test_metrics"),
        report.get("stage2_test_metrics"),
        report.get("cnn_dm_chained_eval"),
        report.get("multinews_eval"),
    )

    # Save.
    save_evaluation_report(report, os.path.join(output_dir, "evaluation_report.json"))
    with open(os.path.join(output_dir, "summary_table.md"), "w") as f:
        f.write(report["summary_table_md"])

    return report


# --- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compute final evaluation metrics from saved JSONs.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--stage1-test",
        default="data/processed/evaluation_outputs/stage1_test_metrics.json",
    )
    parser.add_argument(
        "--stage2-test",
        default="data/processed/evaluation_outputs/stage2_test_metrics.json",
    )
    parser.add_argument(
        "--cnn-dm-chained",
        default="data/processed/evaluation_outputs/cnn_dm_test_chained.json",
    )
    parser.add_argument(
        "--multinews-results",
        default="data/processed/pipeline_outputs/multinews_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/evaluation_outputs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    report = run_full_evaluation(
        stage1_test_path=args.stage1_test,
        stage2_test_path=args.stage2_test,
        cnn_dm_chained_path=args.cnn_dm_chained,
        multinews_results_path=args.multinews_results,
        config=config,
        output_dir=args.output_dir,
    )
    print(report["summary_table_md"])


if __name__ == "__main__":
    main()
