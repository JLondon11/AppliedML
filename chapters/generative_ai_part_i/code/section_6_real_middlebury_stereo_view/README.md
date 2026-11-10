# Chapter 6 — Conditional Stereo-View Synthesis from Genuine Middlebury Data

**Executed task:** Learn a conditional generative distribution of the right-camera RGB image conditioned on a left-camera RGB crop, on **one real calibrated rectified stereo pair** from the Middlebury 2014 Motorcycle benchmark, available as `skimage.data.stereo_motorcycle()`. Official source and acquisition: Scharstein et al. (2014), https://vision.middlebury.edu/stereo/data/scenes2014/ and https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.stereo_motorcycle. This is **not an autonomous-driving video dataset**, detector augmentation, adverse-weather generator, or deployment test.

Partition: rows 0–319 training, 320–399 validation, 400–499 test; 32×32 patch / stride 16; **855/180/225** patches, drawn from just one scene. Overlapping crops are restricted to the same source region, not across partitions.

Model: paired-training conditional convolutional VAE; encoder receives both left and right views; decoder receives the left image and an eight-dimensional latent. Mean RGB reconstruction error plus `0.0005 * KL`; 220 optimizer steps, Adam learning rate 0.001, selected on validation error. For held-out inputs, evaluate the mean of **8 unconditional latent-prior draws** from the CVAE decoder; do not use right-view pixels for inference. Real-valued baseline: copying the left-view crop.

| Method | Mean PSNR (dB) | Mean SSIM |
|---|---:|---:|
| Copy real left view | **23.157** | 0.540 |
| CVAE predictive mean | 14.408 | **0.620** |

The CVAE improves SSIM on 188/225 test patches and PSNR on only 46/225. Blurring causes a major PSNR loss. It is not high-fidelity generative view synthesis. The training checkpoint was selected at the last step; convergence is unverified. No independent stereo scene or real autonomous-driving trajectory was evaluated. This is *within-scene* task-specific evidence, not generalization.

Run with torch, numpy, scikit-image and matplotlib:

```bash
python run.py --steps 220 --output results
python test_outputs.py
```

Outputs: source-photo hashes, test-patch predictions / metrics, training trace, learned checkpoint, a real-vs-generated gallery, and a paired PSNR/SSIM figure. The Pass 41 chapter ZIP also includes a richer real-scene-with-disparity plot and input-gradient-selected test patch gallery from the executed checkpoint.

**Citation:** D. Scharstein et al., “High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth,” GCPR 2014.

**Status:** Experimental run complete for this narrow scope; **scientific generalization / synthetic-driving certification HOLD**.
