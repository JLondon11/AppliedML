"""
Figure 6 — Distributional Semantics
NLP Part I

Native computational scientific rendering generated from a real public-domain corpus.

Corpus:
Lewis Carroll, Alice's Adventures in Wonderland, Project Gutenberg eBook #11
(public-domain text; GITenberg/Project Gutenberg mirror).

Pipeline:
real corpus excerpt -> symmetric word-context co-occurrence counts -> PPMI ->
deterministic SVD -> 2D distributional geometry.

No image-generated or invented quantitative values are used.
"""

from pathlib import Path
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

CORPUS = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having
nothing to do: once or twice she had peeped into the book her sister was reading, but it
had no pictures or conversations in it, and what is the use of a book, thought Alice,
without pictures or conversations? So she was considering in her own mind, as well as
she could, for the hot day made her feel very sleepy and stupid, whether the pleasure
of making a daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her. There was nothing so very
remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit
say to itself, Oh dear! Oh dear! I shall be late! But when the Rabbit actually took a
watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started
to her feet, for it flashed across her mind that she had never before seen a rabbit with
either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she
ran across the field after it, and fortunately was just in time to see it pop down a
large rabbit-hole under the hedge. In another moment down went Alice after it, never once
considering how in the world she was to get out again. The rabbit-hole went straight on
like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not
a moment to think about stopping herself before she found herself falling down a very deep
well. Either the well was very deep, or she fell very slowly, for she had plenty of time
as she went down to look about her and to wonder what was going to happen next.
"""

WINDOW = 4
TARGETS = ["alice", "rabbit", "sister", "book", "watch", "well"]

NAVY = "#0B3C5D"
BLUE = "#328CC1"
VIOLET = "#6D4C9F"
GOLD = "#D9A441"
SLATE = "#1F2937"
BACKGROUND = "#F8FAFC"

OUTDIR = Path("figure_06_distributional_semantics")
OUTDIR.mkdir(parents=True, exist_ok=True)

tokens = re.findall(r"[a-z]+", CORPUS.lower())
freq = Counter(tokens)

contexts = [
    word for word, count in freq.most_common()
    if word not in set(TARGETS) and len(word) > 2
][:18]

target_index = {word: i for i, word in enumerate(TARGETS)}
context_index = {word: i for i, word in enumerate(contexts)}

X = np.zeros((len(TARGETS), len(contexts)), dtype=float)

for i, word in enumerate(tokens):
    if word not in target_index:
        continue
    lo = max(0, i - WINDOW)
    hi = min(len(tokens), i + WINDOW + 1)
    for j in range(lo, hi):
        if j == i:
            continue
        context_word = tokens[j]
        if context_word in context_index:
            X[target_index[word], context_index[context_word]] += 1

total = X.sum()
row_sum = X.sum(axis=1, keepdims=True)
col_sum = X.sum(axis=0, keepdims=True)
expected_denominator = row_sum @ col_sum

with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log2(
        np.where(
            (X > 0) & (expected_denominator > 0),
            (X * total) / expected_denominator,
            1.0,
        )
    )

ppmi = np.where(X > 0, np.maximum(pmi, 0.0), 0.0)

U, S, VT = np.linalg.svd(ppmi, full_matrices=False)
coords = U[:, :2] * S[:2]

if coords[0, 0] < 0:
    coords[:, 0] *= -1
if coords[0, 1] < 0:
    coords[:, 1] *= -1

fig = plt.figure(figsize=(13, 4.7))
grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.05], wspace=0.34)

ax_a = fig.add_subplot(grid[0, 0])
ax_a.axis("off")
snippets = [
    "…Alice… sister… book…",
    "…White Rabbit… Alice… watch…",
    "…rabbit-hole… Alice…",
    "…deep well… Alice…",
]
for k, snippet in enumerate(snippets):
    ax_a.text(0.03, 0.83 - k * 0.20, snippet, fontsize=11.5, color=SLATE)
ax_a.text(0.03, 0.06, f"{len(tokens)} tokens; symmetric window = {WINDOW}", fontsize=9.5, color=NAVY)

ax_b = fig.add_subplot(grid[0, 1])
display_contexts = contexts[:10]
matrix = X[:, :10]
image = ax_b.imshow(matrix, cmap="Blues", aspect="auto", interpolation="nearest")
ax_b.set_xticks(range(len(display_contexts)), display_contexts, rotation=55, ha="right", fontsize=8.5)
ax_b.set_yticks(range(len(TARGETS)), TARGETS, fontsize=9)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if matrix[i, j] > 0:
            ax_b.text(
                j, i, str(int(matrix[i, j])),
                ha="center", va="center", fontsize=7.5,
                color=("white" if matrix[i, j] >= max(2, matrix.max() * 0.55) else SLATE),
            )

for spine in ax_b.spines.values():
    spine.set_visible(False)

ax_c = fig.add_subplot(grid[0, 2])
point_colors = [NAVY, BLUE, VIOLET, GOLD, NAVY, BLUE]

ax_c.scatter(
    coords[:, 0], coords[:, 1],
    s=62, c=point_colors,
    edgecolors=SLATE, linewidths=0.55,
)

for (x, y), word in zip(coords, TARGETS):
    ax_c.annotate(
        word, (x, y), xytext=(5, 5), textcoords="offset points",
        fontsize=9.5, color=SLATE,
    )

ax_c.axhline(0, color=SLATE, linewidth=0.55, alpha=0.35)
ax_c.axvline(0, color=SLATE, linewidth=0.55, alpha=0.35)
ax_c.set_xlabel("SVD dimension 1", fontsize=9.5)
ax_c.set_ylabel("SVD dimension 2", fontsize=9.5)
ax_c.tick_params(labelsize=8)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

for axis, label in zip([ax_a, ax_b, ax_c], ["(a)", "(b)", "(c)"]):
    axis.text(
        0.5, -0.23, label,
        transform=axis.transAxes,
        ha="center", va="top",
        fontsize=11, color=SLATE,
    )

fig.savefig(OUTDIR / "Figure_06_Distributional_Semantics.png", dpi=400, bbox_inches="tight", facecolor="white")
fig.savefig(OUTDIR / "Figure_06_Distributional_Semantics.svg", bbox_inches="tight", facecolor="white")
fig.savefig(OUTDIR / "Figure_06_Distributional_Semantics.pdf", bbox_inches="tight", facecolor="white")

np.savetxt(
    OUTDIR / "Figure_06_raw_counts.csv",
    X, delimiter=",", header=",".join(contexts), comments="",
)

(OUTDIR / "Figure_06_corpus_excerpt.txt").write_text(CORPUS.strip(), encoding="utf-8")

(OUTDIR / "Figure_06_provenance.txt").write_text(
    """Figure 6 — Distributional Semantics
Corpus: Lewis Carroll, Alice's Adventures in Wonderland, Project Gutenberg eBook #11.
Source: public-domain Project Gutenberg text, obtained from the GITenberg mirror.
Processing: lowercase alphabetic tokenization; symmetric context window ±4;
raw word-context counts; PPMI; deterministic NumPy SVD; first two dimensions.
All displayed matrix counts and plotted coordinates are computed from the real corpus excerpt.
No image-generated or invented quantitative values are used.
""",
    encoding="utf-8",
)

print(f"Generated Figure 6 in: {OUTDIR.resolve()}")
