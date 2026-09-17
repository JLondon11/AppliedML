"""Exact production rebuild for AppliedML Pass 13 NLP Part II Figures 9, 11, 12.

This script does not invent metrics. It reads the repository-retained outputs of the
underlying executable experiments and produces the publication figures used in
Pass 13.

Upstream experiment sources:
- Fig 9: case_studies/bert_imdb_sentiment/shared_experiment/run_figures_08_09.py
- Fig 11: case_studies/enterprise_rag/figure_11_retrieval_curve/figure_11_retrieval_quality_topk.py
- Fig 12: case_studies/enterprise_rag/figure_12_hallucination_comparison/figure_12_hallucination_reduction_rag.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def figure_09():
    d = ROOT / "case_studies" / "bert_imdb_sentiment" / "figure_09_loss_curves"
    tr = pd.read_csv(d / "figure_09_training_loss.csv")
    ev = pd.read_csv(d / "figure_09_validation_loss.csv")
    best = ev.loc[ev["validation_loss"].astype(float).idxmin()]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(tr["step"], tr["training_loss"], lw=1.35, label="Training")
    ax.plot(ev["step"], ev["validation_loss"], marker="o", ms=4, lw=1.35, label="Validation")
    ax.axvline(int(best["step"]), ls="--", lw=.9)
    ax.scatter([int(best["step"])], [float(best["validation_loss"])], s=32, zorder=4)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=False)
    ax.tick_params(direction="out")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(d / "figure_09_bert_sentiment_loss_curves.svg", bbox_inches="tight")
    fig.savefig(d / "figure_09_bert_sentiment_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_11():
    d = ROOT / "case_studies" / "enterprise_rag" / "figure_11_retrieval_curve"
    m = pd.read_csv(d / "figure_11_metrics.csv")
    ks = m["k"].to_numpy()
    fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.35))
    ax = axs[0]
    ax.plot(ks, m["mean_recall"], marker="o", lw=1.5)
    ax.fill_between(ks, m["mean_recall"]-m["sem_recall"], m["mean_recall"]+m["sem_recall"], alpha=.16)
    ax.set_xscale("log")
    ax.set_xlabel("Retrieved depth, k")
    ax.set_ylabel("Mean evidence recall")
    ax.set_ylim(0, 1.02)
    ax.grid(False)
    ax = axs[1]
    ax.plot(ks, m["marginal_recall_gain"], marker="o", lw=1.5)
    ax.axhline(0, lw=.7)
    ax.set_xscale("log")
    ax.set_xlabel("Retrieved depth, k")
    ax.set_ylabel("Marginal recall gain")
    ax.grid(False)
    ax = axs[2]
    ax.plot(ks, m["mean_retrieved_whitespace_tokens"], marker="o", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Retrieved depth, k")
    ax.set_ylabel("Mean retrieved words")
    ax2 = ax.twinx()
    ax2.plot(ks, m["mean_irrelevant_fraction"], marker="s", lw=1.25, ls="--")
    ax2.set_ylabel("Mean irrelevant fraction")
    ax2.set_ylim(0, 1.02)
    ax.grid(False)
    for i, ax in enumerate(axs):
        ax.text(.5, -.23, f"({chr(97+i)})", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(d / "figure_11_retrieval_quality_topk.svg", bbox_inches="tight")
    fig.savefig(d / "figure_11_retrieval_quality_topk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_12():
    d = ROOT / "case_studies" / "enterprise_rag" / "figure_12_hallucination_comparison"
    metrics = json.loads((d / "figure_12_metrics.json").read_text())
    smean = metrics["standalone_unsupported_claim_rate"]
    sci = metrics["standalone_bootstrap_95ci"]
    rmean = metrics["rag_unsupported_claim_rate"]
    rci = metrics["rag_bootstrap_95ci"]
    miss = metrics["rag_retrieval_miss_rate"]
    genfail = metrics["rag_generation_failure_given_hit_population_rate"]
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.5))
    ax = axs[0]
    vals = [smean, rmean]
    lo = [smean-sci[0], rmean-rci[0]]
    hi = [sci[1]-smean, rci[1]-rmean]
    ax.bar([0,1], vals, yerr=np.array([lo,hi]), capsize=4, width=.58)
    ax.set_xticks([0,1], ["Standalone", "Retrieval-grounded"])
    ax.set_ylabel("Mean unsupported-term fraction")
    ax.set_ylim(0,1)
    ax = axs[1]
    ax.bar([0,1], [miss, genfail], width=.58)
    ax.set_xticks([0,1], ["Retrieval miss", "Generation failure\nafter retrieval hit"])
    ax.set_ylabel("Fraction of evaluated claims")
    ax.set_ylim(0,1)
    for i, ax in enumerate(axs):
        ax.tick_params(direction="out")
        ax.grid(False)
        ax.text(.5, -.23, f"({chr(97+i)})", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    fig.tight_layout(w_pad=2.6)
    fig.savefig(d / "figure_12_hallucination_reduction_rag.svg", bbox_inches="tight")
    fig.savefig(d / "figure_12_hallucination_reduction_rag.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_09()
    figure_11()
    figure_12()
