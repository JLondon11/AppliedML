# Chapter 6 committed-source evidence review — 2026-11-12

## Examined source (default branch)
- `figure_generation/pass12/generate_pass12_figures.py` — imports the VAE, replacement, DDPM, and figure-composition modules. Advertises Figure 03/10/11/14/20/24 outputs.
- `figure_generation/pass12/run_vae_evidence.py` — loads actual scikit-learn handwritten-digit observations; asserts 1,347 training / 450 validation / 0 testing; measures ELBO and 10-NN latent-label purity; writes weights, metrics and figures.
- `figure_generation/pass12/real_data_figures.py` — loads the same digits dataset; evaluates VAE reconstruction and injected-noise robustness on validation images; fits a separate conditional denoiser. Explicitly disclaims DDPM equivalence for that denoiser.
- `figure_generation/pass12/ddpm_digits_experiment.py` — real 1,797-digit source; asserts 1,257 train / 270 validation / 270 independent test examples, with disjoint indices; trains a noise predictor, chooses checkpoint by validation MSE, reports held-out timestep-wise epsilon MSE against zero prediction, samples images.
- `figure_generation/pass12/compose_ddpm_figure24.py` — composites saved real test digits, generated DDPM samples, and stored numerical test metrics.

## Supported source-code observations, not runtime certification
1. Training and partitioning logic exists for both VAE and DDPM; these source-inspected studies are candidates for proportional **Level A** after actual output/figure and scientific claim checks.
2. The VAE has validation data but no separate held-out test subset. Its demonstration must not be advertised as independent test performance.
3. The DDPM has independent held-out test noise-MSE measurements with a zero-predictor reference. This does not quantify sample distribution fidelity or establish model superiority on image-generation benchmarks.
4. The source for Figure 20 is a digit VAE noise-robustness plot; the production checkpoint reports industrial-defect Figure 6.20. **Blocking numbering/provenance ambiguity:** reconcile exact source, manuscript print number, image hash and caption. Do not silently reuse the digit plot as defect evidence.
5. An automated mechanical audit script now exists at `figure_generation/audit_pass12_evidence.py`. It validates expected JSON and image artifacts once generated, and explicitly does not confer scientific certification.

## Not yet verified
- No Pass-12 output bundle has been independently executed and assessed in this review.
- Current Pass-37 manuscript PDF/TeX and print figure assets are not accessible from the brief status record.
- Industrial-defect, synthetic-driving and concrete-mixture task-specific runtime evidence and downstream validation remain unconfirmed.
- Source-code existence does not certify all current chapter unit experiments.

## Next evidence gates
1. Run `python chapters/generative_ai_part_i/figure_generation/pass12/generate_pass12_figures.py` in a locked environment, retaining full logs, metrics and hashes.
2. Run `python chapters/generative_ai_part_i/figure_generation/audit_pass12_evidence.py --outputs chapters/generative_ai_part_i/figure_generation/pass12/outputs --report audit_pass12.json`.
3. Independently inspect image panels and measured metrics, then reconcile all output files against the actual current manuscript captions.
4. Evaluate remaining application-specific models and record requirement-by-requirement evidence links before assigning PASS.
