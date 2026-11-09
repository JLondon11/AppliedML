# Chapter 9: real-data DDPM experimental evidence

The actual experiment uses sklearn's 1,797 real 8×8 handwritten-digit images. Stratified, disjoint partitions contain **1,257 training**, **270 validation**, and **270 independent test** samples. Normalize the image intensities from [0,16] to [-1,1].

The unconditional DDPM has 100 diffusion steps, an epsilon-prediction MLP with 66-256-256-64 dimensions, and two periodic timestep features. Training ran for 45 epochs using Adam at 0.001, minibatch size 128. Select the checkpoint using validation epsilon-MSE averaged over timesteps 10, 25, 50, 75, and 100. Test metrics use unseen held-out images; sampling uses ancestral DDPM reverse diffusion from Gaussian latent noise.

| Forward timestep | Trained epsilon MSE | Zero-epsilon baseline MSE |
|---|---:|---:|
| 10 | 0.709731 | 0.999047 |
| 25 | 0.307815 | 0.992306 |
| 50 | 0.122999 | 0.998605 |
| 75 | 0.080936 | 1.019122 |
| 100 | 0.081337 | 0.981715 |

Unweighted mean over five measured timesteps: **0.260564 vs. 0.998159**. This is an epsilon-prediction comparison, **not** an image-generation-quality comparison. Several generated images have visible artifacts. The experiment does not evaluate DDIM, score-based SDE methods, or SOTA generators. Figure 24 is eligible for *verified experimental provenance* but **not yet for full quality certification**.

## Reproduction

From this directory:

```bash
python ddpm_digits_experiment.py --out outputs/ddpm --epochs 45
python compose_ddpm_figure24.py outputs/ddpm outputs/figure_24.png
```

Dependencies: NumPy, scikit-learn, PyTorch, Matplotlib. All generated outputs and per-run versions are stored under `outputs/ddpm/` when executed; GitHub Actions publishes this directory as an artifact.

## Suggested scientific caption for Figure 9.24

**Real-data DDPM inference on handwritten digits.** Left: representative real images from an independent held-out test partition (270 images). Center: sixteen unconditional samples produced by the trained 100-step DDPM reverse-diffusion process. Right: held-out noise-prediction mean-squared error at five forward timesteps, compared against a zero-prediction baseline. The compact DDPM was trained on 1,257 real images for 45 epochs, with a separate 270-image validation set used for checkpoint selection. Generated images exhibit visible imperfections; these results do not establish DDIM or score-based diffusion performance.
