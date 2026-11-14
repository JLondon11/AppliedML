# Exact GitHub-head source execution — 2026-11-13

## Source identity
The source was read from `main:chapters/generative_ai_part_i/figure_generation/vae_pca_real_digits.py` and reconstructed byte-for-byte in the execution environment. Its Git blob SHA-1 was independently computed as `666cac09a9ec5b127f2ba7d89f109c6b10065df4` and matched the GitHub API blob SHA.

## Execution
Command: `python vae_pca_real_digits.py --out output --seed 19 --epochs 35`. Execution completed successfully using Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0, and torch 2.10.0+cpu. Real sklearn handwritten digits: source 1,797; training 1,257; validation 270; independent test 270. Selected epoch 35. Test-index SHA256 `bfe1805d067fe13271534f968cdf891b227872c039fbc2151f7772fedd394b93`.

| Quantity | Exact head rerun | Archived evidence |
|---|---:|---:|
| VAE test MSE | 0.05387162044644356 | 0.05387162044644356 |
| PCA test MSE | 0.05312391743063927 | 0.05312391743063927 |
| Mean paired PCA minus VAE MSE | -0.0007477071485482156 | -0.0007477067993022501 |
| Bootstrap CI lower | -0.001515885314438492 | -0.0015158850699663162 |
| Bootstrap CI upper | 0.000004024867928365255 | 0.000004024946656500377 |

Comparing independent-test prediction rows, the maximum absolute differences were: VAE per-image MSE 1.49e-8, PCA per-image MSE 0, latent coordinates 4.77e-7. Maximum validation-history MSE difference 7.45e-9. All five output files had **different** SHA-256 hashes from the earlier ZIP, including the PNG and checkpoint. These differences are compatible with numerically equivalent execution but do not prove a specific hardware/software cause. **Exact output-byte identity is NOT certified.**

## Scientific conclusion and gate
A single-seed held-out experiment showed VAE reconstruction MSE slightly higher than PCA, with paired bootstrap interval including zero. It does **not** demonstrate VAE superiority. Source identity is **PASS**, numerical re-execution is **PASS within 1e-6**, bytewise output identity is **NOT PASS**, and current manuscript figure mapping and full R9 asset delivery are **PENDING**. Earlier `HEAD_REPLAY_2026-11-12.md` descriptions of exact matching artifacts concern a separately reconstructed/archived program, not this literal GitHub-source execution.

Figure 6.20 in the current manuscript and the industrial-defect, driving, and concrete-mixture application evidence remain outside this experiment's certification.
