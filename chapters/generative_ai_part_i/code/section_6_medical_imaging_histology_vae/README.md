# Chapter 6 — Medical imaging synthesis: measured real tissue VAE

**Executed model:** convolutional variational autoencoder (24-dimensional latent). **Actual source:** `skimage.data.immunohistochemistry()`, one real 512×512 RGB immunohistochemistry microscopy image. This is not a radiology, patient-level, or clinical benchmark.

The tissue image is divided by non-overlapping **spatial parent regions**, then subdivided into overlapping 32×32 RGB patches at stride 16 within each region: 589 training patches from image rows 0–319, 93 validation patches from rows 320–383, and 217 test patches from rows 384–511. No test patch pixels are present in training/validation regions. Since **all** patches originate from one image, they are not independent patients or independently collected tissue samples.

Run:
```bash
python run.py --epochs 12
python test_outputs.py
```
Requires Python, PyTorch, NumPy, scikit-image and Matplotlib. Seed: 20261109, 12 epochs, validation-selected final checkpoint. Output includes genuine source checksum, saved checkpoint, all 217 held-out per-patch metrics, 128 prior-sample feature records, training trace, and two scientifically labeled figures. The exact executed outputs and figures are in `Chapter6_Generative_AI_PartI_Pass22.zip`.

| Observed quantity | Value |
|---|---:|
| Held-out mean reconstruction PSNR | 20.1757 dB |
| Held-out mean reconstruction SSIM | 0.2927 |
| Held-out mean reconstruction pixel MSE | 0.010447 |
| Nearest training patch MSE (descriptive comparator) | 0.010598 |
| Prior generated samples | 128 |

**Negative result:** Posterior-mean reconstructions blur fine tissue structures and unconditional samples are largely featureless. Reconstruction PSNR does not assess unconditional generation. This single-tissue experiment cannot establish clinical plausibility, privacy protection, patient-level generalization, pathological label fidelity, or downstream utility. It adds **partial executed evidence** to Medical Imaging Synthesis, but full Master QA remains HOLD.
