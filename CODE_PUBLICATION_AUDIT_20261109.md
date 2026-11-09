# GitHub code publication audit — 2026-11-09

Repository: `JLondon11/AppliedML` (main). This is a **repository inventory**, not a claim that all experimental models have run or that every manuscript unit is certified.

A recursive GitHub tree inspection found **203 code files** with extensions `.py`, `.ipynb`, or `.sh`. Of these, **189 files** are under the book's `chapters/` directory (including three shared runtime/production scripts); 14 are elsewhere. These counts do not validate correctness, provenance, or completeness.

## Chapter-family directory coverage

| Chapter family | Current code files | Audit limitation |
|---|---:|---|
| `chapters/foundations_machine_learning` | 8 | Unit-by-unit completeness not established |
| `chapters/deep_learning` | 11 | Part I has 7 experiment scripts and 3 figure/production scripts; Part II has 1 figure generator; named Part I experimental gaps remain |
| `chapters/computer_vision` | 50 | Unit-by-unit completeness not established |
| `chapters/generative_ai_part_i` | 42 | Unit-by-unit completeness not established |
| `chapters/generative_ai_part_ii` | 5 | Unit-by-unit completeness not established |
| `chapters/nlp_part_i` | 5 | Unit-by-unit completeness not established |
| `chapters/nlp_part_ii` | 46 | Unit-by-unit completeness not established |
| `chapters/reinforcement_learning` | 1 | Only a figure generator is evident; all experiment units require detailed audit |
| `chapters/scientific_ai` | 17 | Unit-by-unit completeness not established |
| `chapters/systems_engineering_and_mlops` | 1 | Only a figure generator is evident; all experiment units require detailed audit |
| `chapters/_shared` | 3 | Shared infrastructure, not an experimental unit |

**No chapter is declared 100% code-publication compliant from counts alone.** Also, not every listed file is necessarily executable or an experimental implementation.

## Chapter 2: verified directory presence

`chapters/deep_learning/part_i/experiments/` contains these seven committed experimental entrypoints:

- `digits_cpu_pilot/run.py`
- `digits_threeway_cpu/run.py`
- `gradient_flow_digits/run.py`
- `gradient_flow_multiseed_digits/run.py`
- `sunspots_lstm_gru/run.py`
- `sunspots_recurrent_gradients/run.py`
- `sunspots_sequence_length_ablation/run.py`

These genuine scikit-learn Digits and NGDC sunspot experiments **do not replace** missing named-dataset MNIST, CIFAR-10, UCI HAR, hurricane satellite/HURDAT2, Stanford Dogs, style-transfer, or optical-flow experiments.

Chapter 2 figure/production entrypoints visible in GitHub:

- `figure_generation/generate_figures.py`
- `production_pass14/generate_closure_figures.py`
- `production_pass14/resolve_remaining_figures.py`

All require scientific inspection; a generic figure generator is not evidence that named experiments are completed.

## Immediate open issues

1. Build a manuscript-to-GitHub inventory for **every named application and case study** across every chapter; identify absent scripts and uncommitted code.
2. Commit any finished local-only code with verified paths, including figure generators, tests, and relevant result artifacts. Full dataset copies should be handled according to license and repository size.
3. Re-execute named experiments on their actual datasets and produce scientifically correct output figures/metrics.
4. Require section-specific README, dataset partition counts (in scientific prose), run commands, test evidence, explicit figure linkage, and commit SHA.
5. Avoid silently accepting image placeholders, fabricated outputs, experimental claims unbacked by code, or old LaTeX sources with code-listing regressions.

**Status: PARTIAL repository coverage verified; universal code-publication certification NOT MET.**
