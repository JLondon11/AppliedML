"""Generate NLP Part II Figure 4 from real BERT-Tiny contextual embeddings.

Frozen inventory target:
  Embedding Geometry in Contextual Representation Spaces

Model:
  google/bert_uncased_L-2_H-128_A-2 (BERT-Tiny, hidden size 128)

All plotted coordinates are derived from actual last-layer contextual token
vectors. The script saves every extracted 128-dimensional vector, the PCA
coordinates, PCA components, and model revision. No point is manually placed.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
HERE = Path(__file__).resolve().parent
plt.rcParams["svg.fonttype"] = "none"

# Restrained scientific palette.
PALETTE = {
    "animals": "#43566B",
    "vehicles": "#71879B",
    "finance": "#476B63",
    "geography": "#8A6958",
}
BANK_FIN = "#476B63"
BANK_GEO = "#8A6958"
NEUTRAL = "#333333"

examples = [
    # Semantic neighborhoods for panel (a)
    ("dog_1", "animals", "dog", "The dog chased a ball across the field."),
    ("cat_1", "animals", "cat", "The cat slept quietly near the window."),
    ("wolf_1", "animals", "wolf", "A wolf moved silently through the forest."),
    ("car_1", "vehicles", "car", "The car stopped beside the station."),
    ("truck_1", "vehicles", "truck", "The truck carried equipment down the road."),
    ("bus_1", "vehicles", "bus", "The bus arrived at the city terminal."),
    ("train_1", "vehicles", "train", "The train crossed the bridge before dawn."),
    ("plane_1", "vehicles", "plane", "The plane landed safely at the airport."),
    ("loan_1", "finance", "loan", "The lender approved the loan after reviewing the application."),
    ("money_1", "finance", "money", "She transferred the money into a savings account."),
    ("credit_1", "finance", "credit", "The customer used credit to finance the purchase."),
    ("investment_1", "finance", "investment", "The investment produced a steady annual return."),
    ("river_1", "geography", "river", "The river flowed through the broad valley."),
    ("shore_1", "geography", "shore", "Waves reached the rocky shore at sunset."),
    ("water_1", "geography", "water", "Cold water moved rapidly around the stones."),
    ("stream_1", "geography", "stream", "The stream curved through the meadow."),

    # Polysemous bank contexts for panel (b)
    ("bank_fin_1", "bank_finance", "bank", "The bank approved the business loan yesterday."),
    ("bank_fin_2", "bank_finance", "bank", "She deposited money at the bank before work."),
    ("bank_fin_3", "bank_finance", "bank", "The bank offered customers a lower interest rate."),
    ("bank_fin_4", "bank_finance", "bank", "The bank reviewed the company's credit application."),
    ("bank_fin_5", "bank_finance", "bank", "Investors met with the bank to discuss financing."),
    ("bank_geo_1", "bank_geography", "bank", "They rested on the bank of the river."),
    ("bank_geo_2", "bank_geography", "bank", "Trees grew along the muddy river bank."),
    ("bank_geo_3", "bank_geography", "bank", "The canoe reached the bank before the storm."),
    ("bank_geo_4", "bank_geography", "bank", "Wildflowers covered the steep bank beside the stream."),
    ("bank_geo_5", "bank_geography", "bank", "The water rose above the grassy bank after the rain."),
]

def locate_target(offsets, sentence: str, target: str):
    # Find the first exact word occurrence, then select all wordpiece tokens
    # whose character spans overlap that occurrence.
    lower = sentence.lower()
    start = lower.index(target.lower())
    end = start + len(target)
    idx = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if max(a, start) < min(b, end):
            idx.append(i)
    if not idx:
        raise RuntimeError(f"Could not locate target {target!r} in {sentence!r}")
    return idx

def extract():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    rows = []
    with torch.no_grad():
        for context_id, category, target, sentence in examples:
            enc = tokenizer(
                sentence,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=128,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            target_idx = locate_target(offsets, sentence, target)
            out = model(**enc, output_hidden_states=True, return_dict=True)
            vec = out.last_hidden_state[0, target_idx, :].mean(dim=0).cpu().numpy()
            token_ids = enc["input_ids"][0, target_idx].tolist()
            pieces = tokenizer.convert_ids_to_tokens(token_ids)

            row = {
                "context_id": context_id,
                "category": category,
                "target_word": target,
                "sentence": sentence,
                "wordpieces": " ".join(pieces),
                "hidden_size": int(vec.shape[0]),
            }
            row.update({f"h{i:03d}": float(v) for i, v in enumerate(vec)})
            rows.append(row)

    return pd.DataFrame(rows)

def main():
    df = extract()
    hidden_cols = [c for c in df.columns if len(c) == 4 and c[0] == "h" and c[1:].isdigit()]
    X = df[hidden_cols].to_numpy(float)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    df["pc1"] = Z[:, 0]
    df["pc2"] = Z[:, 1]

    info = HfApi().model_info(MODEL_ID)
    revision = info.sha

    df.to_csv(HERE / "figure_04_bert_tiny_contextual_vectors.csv", index=False)

    pca_df = pd.DataFrame(
        pca.components_,
        index=["PC1", "PC2"],
        columns=hidden_cols,
    )
    pca_df.insert(0, "explained_variance_ratio",
                  pca.explained_variance_ratio_)
    pca_df.to_csv(HERE / "figure_04_pca_components.csv")

    metadata = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "hidden_size": int(len(hidden_cols)),
        "num_contexts": int(len(df)),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "torch_version": torch.__version__,
    }
    (HERE / "figure_04_model_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))

    # (a) Contextual semantic neighborhoods.
    ax = axes[0]
    for category in ["animals", "vehicles", "finance", "geography"]:
        sub = df[df.category == category]
        ax.scatter(
            sub.pc1, sub.pc2, s=34,
            color=PALETTE[category],
            label=category,
        )
        for r in sub.itertuples(index=False):
            ax.annotate(
                r.target_word, (r.pc1, r.pc2),
                xytext=(4, 4), textcoords="offset points", fontsize=7.5
            )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.5, -0.20, "(a)", transform=ax.transAxes,
            ha="center", va="top", fontsize=12)

    # (b) Same lexical token in distinct contexts.
    ax = axes[1]
    bf = df[df.category == "bank_finance"]
    bg = df[df.category == "bank_geography"]
    ax.scatter(bf.pc1, bf.pc2, s=38, color=BANK_FIN, label="financial bank")
    ax.scatter(bg.pc1, bg.pc2, s=38, color=BANK_GEO, marker="^", label="river bank")

    cf = bf[["pc1", "pc2"]].mean().to_numpy()
    cg = bg[["pc1", "pc2"]].mean().to_numpy()
    ax.scatter(*cf, s=80, facecolors="none", edgecolors=BANK_FIN, linewidths=1.3)
    ax.scatter(*cg, s=80, facecolors="none", edgecolors=BANK_GEO, linewidths=1.3)
    ax.plot([cf[0], cg[0]], [cf[1], cg[1]],
            color=NEUTRAL, linewidth=1.0, linestyle="--")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.5, -0.20, "(b)", transform=ax.transAxes,
            ha="center", va="top", fontsize=12)

    # (c) Relational geometry from actual contextual centroids.
    ax = axes[2]
    centroids = {}
    for category in ["animals", "vehicles", "finance", "geography"]:
        sub = df[df.category == category]
        c = sub[["pc1", "pc2"]].mean().to_numpy()
        centroids[category] = c
        ax.scatter(*c, s=55, color=PALETTE[category])
        ax.annotate(category, c, xytext=(5, 5),
                    textcoords="offset points", fontsize=8)

    ax.scatter(*cf, s=55, color=BANK_FIN, marker="s")
    ax.scatter(*cg, s=55, color=BANK_GEO, marker="^")
    ax.annotate("bank / finance", cf, xytext=(5, -12),
                textcoords="offset points", fontsize=8)
    ax.annotate("bank / river", cg, xytext=(5, -12),
                textcoords="offset points", fontsize=8)

    # Actual displacement directions in the same PCA space.
    ax.annotate("", xy=centroids["finance"], xytext=cf,
                arrowprops=dict(arrowstyle="->", color=BANK_FIN, linewidth=1.0))
    ax.annotate("", xy=centroids["geography"], xytext=cg,
                arrowprops=dict(arrowstyle="->", color=BANK_GEO, linewidth=1.0))
    ax.annotate("", xy=centroids["vehicles"], xytext=centroids["animals"],
                arrowprops=dict(arrowstyle="->", color=NEUTRAL, linewidth=1.0))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.text(0.5, -0.20, "(c)", transform=ax.transAxes,
            ha="center", va="top", fontsize=12)

    for ax in axes:
        ax.grid(False)
        ax.tick_params(direction="out", width=0.8)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(HERE / "figure_04_embedding_geometry.svg", bbox_inches="tight")
    fig.savefig(HERE / "figure_04_embedding_geometry.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
