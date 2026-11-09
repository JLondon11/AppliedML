# Chapter 6 — Autonomous scientific discovery, real NCI computational diagnostic (Pass 24)

**Run:** `python run.py --budget 50`, then `python test_outputs.py`. Dependencies: RDKit, numpy, scikit-learn, matplotlib.

Genuine molecular structures: bundled RDKit `Data/NCI/first_5K.smi`, SHA-256 `91e71c015f14939837f2943dcc904f7c87e5a3a0124d82b05c28ad2f23004def`. After filtering, 4,886 unique structures (3,908 surrogate training, 978 held-out test). Random forest property surrogate (35 trees, depth 12), 512-bit Morgan fingerprints. Held-out mean absolute error **0.13436**; **R² 0.38227** for deterministic, RDKit-calculated penalized QED.

Graph-edit generator (not a neural SMILES decoder) proposes atom substitution, atom addition or terminal-atom deletion; generated structures are sanitized, deduplicated and excluded from the source corpus. Three seeds 11,23,37, with 50 selected valid candidate evaluations per arm/seed (300 total). **The surrogate-guided method scores up to eight mutation proposals per selection versus one for the random control; the number of final chemically evaluated candidates is equal, but compute budgets are not.**

| Method | Best score mean ± SD | Mean candidate score ± SD | Mean distinct Murcko scaffolds ± SD |
|---|---:|---:|---:|
| Surrogate-guided | 0.8421 ± 0.1075 | 0.5746 ± 0.0736 | 17.67 ± 5.77 |
| Random edits | 0.8502 ± 0.0107 | 0.4770 ± 0.0193 | 24.33 ± 5.51 |

Surrogate guidance improves mean candidate scores but **not mean best-of-budget** scores, and it reduces scaffold diversity. No activity, synthesizability, toxicity, biological efficacy, or scientific discovery is experimentally established; QED/MW/logP are computed descriptors.

Pass 24 deliverables contain the executed output tree `experiment/results`: manifest, all 978 held-out NCI predictions, 300 canonical SMILES candidates and scores, summary, generated chemical structure images and scientific plots. The GitHub source is organized with this chapter section. Full chapter Master QA status remains **HOLD (10 partial, 6 missing, 0 fully certified)**.

### Source-code correctness

The complete code and checker are committed here, but the exact GitHub checkout has not been independently rerun. The local identical experiment logic and output checker completed, and the resulting Chapter 6 PDF compiled successfully to 119 pages.
