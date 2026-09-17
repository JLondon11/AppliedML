# Figure Code Index

This index records the canonical chapter-level entry points for figure-generation code used by the AppliedML book production pipeline.

| Book chapter | Canonical generator / code location |
|---|---|
| Computer Vision Part I — Foundations | `chapters/computer_vision/part_i/foundations/figure_generation/generate_figures.py` |
| Computer Vision Part II — Autonomous Systems | `chapters/computer_vision/part_ii/autonomous_vehicles/figure_generation/generate_figures.py` |
| Deep Learning Part I | `chapters/deep_learning/part_i/figure_generation/generate_figures.py` |
| Deep Learning Part II | `chapters/deep_learning/part_ii/figure_generation/generate_figures.py` |
| Foundations of Machine Learning | `chapters/foundations_machine_learning/figure_generation/generate_figures.py` |
| Generative AI Part I | `chapters/generative_ai_part_i/figure_generation/generate_figures.py` |
| Generative AI Part II | `chapters/generative_ai_part_ii/figure_generation/generate_figures.py` plus section-specific scaling-law scripts |
| NLP Part I | `chapters/nlp_part_i/figure_generation/generate_figures.py` plus figure-specific scripts under `chapters/nlp_part_i/figures/` |
| NLP Part II | `chapters/nlp_part_ii/figure_generation/generate_figures.py` plus section-specific experiment directories |
| Reinforcement Learning | `chapters/reinforcement_learning/figure_generation/generate_figures.py` |
| Scientific AI | `chapters/scientific_ai/figure_generation/generate_figures.py`, `chapters/scientific_ai/code/`, and application-specific directories |
| Systems Engineering & MLOps | `chapters/systems_engineering_and_mlops/figure_generation/generate_figures.py` |

## Current production overrides

The exact Pass 12 Generative AI Part I generators for figures 3, 10, 11, 14, 20, and 24 are in:

`chapters/generative_ai_part_i/figure_generation/pass12/generate_pass12_figures.py`

These overrides take precedence over older chapter-level fallback logic for those figure numbers.

## Provenance rule

Code location alone is not sufficient for Master-QA acceptance. For an empirical figure, the repository must also identify the dataset or published numerical source, preprocessing/splitting protocol, model/algorithm, seed where applicable, and the figure number/caption it reproduces. Figures based on externally published artwork or unavailable protocol-matched results remain HOLD/BLOCKED rather than being replaced with synthetic values.

## Section placement

Where a figure has a section-specific experiment directory (for example NLP Part II scaling laws, BERT sentiment, enterprise RAG, interpretability, or Scientific AI ophthalmology), that directory is the canonical source. Shared chapter-level generators are retained as orchestration/fallback entry points for deterministic non-empirical figures.
