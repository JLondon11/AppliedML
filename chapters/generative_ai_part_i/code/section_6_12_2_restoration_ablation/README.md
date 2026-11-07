# Section 6.12.2 — Photographic denoising ablation (Pass 18)

The completed experiment evaluates a time-conditioned versus unconditioned epsilon-prediction CNN on genuine scikit-image photographs, with algorithmically added Gaussian noise. Four separate photographs train the networks; camera and coins are held out.

Experiment: seeds 20261106, 20261107, 20261108; 65 training steps per model; retained signal alpha-bars 0.90 and 0.75; Gaussian baseline sigma 0.8. The six network trainings completed after a larger training budget exceeded the runtime limit. Results and checkpoint files are packaged with Chapter6_RealData_Certification_Pass18.zip.

| Model | Mean PSNR dB | Mean SSIM |
|---|---:|---:|
| Noisy input | 9.928 | 0.129 |
| Gaussian smoothing | 16.989 | **0.318** |
| Unconditioned CNN | 17.352 | 0.291 |
| Conditioned CNN | **17.684** | 0.308 |

This is a single-step noise-prediction ablation, **not** iterative DDPM sampling or scientific microscopy restoration. The same two held-out images are used across seeds; they are not independent datasets. No clinical or cellular measurement validity is claimed.

Run the corresponding `run.py` with numpy, scipy, matplotlib, scikit-image and torch installed; it emits `results/metrics.csv`, `summary.csv`, `manifest.json`, plots and six model checkpoints. Full Chapter 6 Master QA remains HOLD; see the 16-unit certification matrix in Pass 18.
