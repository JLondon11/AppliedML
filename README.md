# AppliedML

Source repository for the Applied Machine Learning book.

## Figure-generation policy

All code used to generate book figures must live in this repository under the correct chapter. Figure-generation code is organized under chapter-specific `figure_generation`, `code`, or section-specific figure directories. Empirical figures must retain dataset / experiment provenance; externally published artwork or results must not be replaced by fabricated numerical values.

Primary generator locations:

- `chapters/computer_vision/part_i/foundations/figure_generation/`
- `chapters/computer_vision/part_ii/autonomous_vehicles/figure_generation/`
- `chapters/deep_learning/part_i/figure_generation/`
- `chapters/deep_learning/part_ii/figure_generation/`
- `chapters/foundations_machine_learning/figure_generation/`
- `chapters/generative_ai_part_i/figure_generation/`
- `chapters/generative_ai_part_ii/figure_generation/`
- `chapters/nlp_part_i/figure_generation/`
- `chapters/nlp_part_ii/figure_generation/` plus section-specific experiment directories
- `chapters/reinforcement_learning/figure_generation/`
- `chapters/scientific_ai/code/`, `chapters/scientific_ai/figure_generation/`, and section-specific application directories
- `chapters/systems_engineering_and_mlops/figure_generation/`

The Pass 12 Generative AI Part I remediations (figures 3, 10, 11, 14, 20, and 24) are reproduced by:

`chapters/generative_ai_part_i/figure_generation/pass12/generate_pass12_figures.py`

These include the actual compact VAE training run on the public scikit-learn digits dataset used for the accepted latent interpolation, ELBO, and latent-neighborhood-purity figures.

## Production rule

A figure must not be marked reproducible merely because a generic plotting script exists. The generator, dataset/protocol, seed, and frozen-inventory figure number must agree with the production artifact. Dataset-specific, clinical, benchmark, or simulator-result figures remain provenance-gated until their exact source or protocol-matched computation is available.
