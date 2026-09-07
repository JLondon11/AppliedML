"""Generate NLP Part II Figure 3: Byte-Pair Encoding (BPE).

Reproducible scientific rendering. The corpus is explicitly defined below;
all plotted values are computed from the BPE merge procedure rather than
invented or manually placed.
"""
from collections import Counter
from pathlib import Path
import csv
import matplotlib.pyplot as plt

CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
N_MERGES = 8
OUT = Path(__file__).resolve().parent

def symbols(word):
    return list(word) + ["</w>"]

def pair_counts(vocab):
    counts = Counter()
    for seq, freq in vocab.items():
        for a, b in zip(seq[:-1], seq[1:]):
            counts[(a, b)] += freq
    return counts

def merge_pair(vocab, pair):
    out = {}
    a, b = pair
    merged = a + b
    for seq, freq in vocab.items():
        seq = list(seq)
        new, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i+1] == b:
                new.append(merged); i += 2
            else:
                new.append(seq[i]); i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + freq
    return out

def total_symbols(vocab):
    return sum(len(seq) * freq for seq, freq in vocab.items())

def run_bpe():
    vocab = {tuple(symbols(w)): f for w, f in CORPUS.items()}
    history = []
    states = [vocab]
    for step in range(1, N_MERGES + 1):
        counts = pair_counts(vocab)
        pair, frequency = counts.most_common(1)[0]
        before = total_symbols(vocab)
        vocab = merge_pair(vocab, pair)
        after = total_symbols(vocab)
        history.append((step, pair[0], pair[1], pair[0]+pair[1],
                        frequency, before, after, before-after))
        states.append(vocab)
    return history, states

def word_tokens(vocab, word):
    for seq, freq in vocab.items():
        if "".join(seq).replace("</w>", "") == word:
            return list(seq)
    raise KeyError(word)

def main():
    history, states = run_bpe()

    with (OUT / "figure_03_bpe_merges.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","pair_left","pair_right","merged_token",
                    "pair_frequency","symbols_before","symbols_after",
                    "symbol_reduction"])
        w.writerows(history)

    checkpoints = [0, 1, 2, 4, 8]
    with (OUT / "figure_03_bpe_tokenization.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["merges_applied","word","token_sequence_length","tokens"])
        for k in checkpoints:
            for word in CORPUS:
                toks = word_tokens(states[k], word)
                w.writerow([k, word, len(toks), " ".join(toks)])

    steps = [r[0] for r in history]
    freqs = [r[4] for r in history]
    after = [r[6] for r in history]
    reductions = [r[7] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
    ax = axes[0]
    ax.plot(steps, freqs, marker="o", linewidth=1.5)
    ax.set(xlabel="Merge step", ylabel="Selected pair frequency")
    ax.grid(alpha=.2)

    ax = axes[1]
    ax.plot([0]+steps, [total_symbols(states[0])]+after, marker="o", linewidth=1.5)
    ax.set(xlabel="Merges applied", ylabel="Corpus symbol count")
    ax.grid(alpha=.2)

    ax = axes[2]
    for word in CORPUS:
        lengths = [len(word_tokens(states[k], word)) for k in checkpoints]
        ax.plot(checkpoints, lengths, marker="o", linewidth=1.3, label=word)
    ax.set(xlabel="Merges applied", ylabel="Token sequence length")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)

    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        ax.text(.5, -.25, label, transform=ax.transAxes, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(OUT / "figure_03_bpe.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure_03_bpe.svg", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
