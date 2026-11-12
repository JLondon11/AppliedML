# Chapter 6 — Pass 48 scientific unit certification

**Full application certification: HOLD, 0/14.** Clean independent model-training reproducibility is **CERTIFIED (scoped)** for units **3, 6 and 13** only, from the exact previously packaged Chapter 6 source and true source datasets. The original 109-page Pass 45 manuscript remains unchanged. Full per-unit external validity is not certified.

## Independent reruns

* **Unit 3: genuine Middlebury single-pair stereo conditional VAE.** Fresh 220-step train; spatial split 855/180/225 patches. Genuine left/right image SHA-256 agrees with original. All 225 test patch metrics agree with original saved metrics to within 3e-6. CVAE mean PSNR 14.4082 dB (copy-left reference 23.1570 dB); SSIM 0.61985 (copy-left 0.54001). Failures retained: poor PSNR, one parent scene, correlated patches, no out-of-scene validation.
* **Unit 6: genuinely scanned 1797 handwritten digits.** Fresh 25-epoch VAE, 35-epoch conditional and unconditional latent diffusion, split 1257/270/270. All 2000 generated-image classifier labels agree; max per-class probability difference below 4e-6. Conditional class agreement 77.8%, unconditional 9.4%; genuine digit test classifier accuracy 95.185%. Not pretrained Stable Diffusion or natural-language-to-image.
* **Unit 13: genuine Mauna Loa CO2 measurements.** Fresh 65-epoch training for three model seeds, split 1145/254/306 chronological windows. All 12 benchmark RMSE/MAE scores match and all 7344 saved forecast rows reproduce. Mean CVAE RMSE 0.4868 ppm versus ridge 0.6673 ppm. Nominal 90% prior-predictive bands cover only 20.25%; **raw probabilistic calibration fails**.

## Pass 48 new strictly later-period calibration test

The already-selected models use training and checkpoint selection periods ending by 1995. Post-hoc absolute-residual interval halfwidths were fitted per forecast horizon on 149 windows with all targets ending in 1998 and independently evaluated on 150 later windows starting after December 31, 1998. Observed original interval coverage in 1999-2001 was 23.36%; observed residual-band interval coverage was **93.39%**, mean interval width **1.705 ppm**. Three model seeds and eight leads; all results from measured genuine observations. Forecast windows within the calibration/evaluation intervals are overlapping/dependent, so **no exchangeability or finite-sample conformal coverage guarantee** is claimed. This supplementary result is not integrated into the Pass 45 chapter and does not certify general climate modeling.

## Remaining external scientific gates

No genuine BDD100K downstream multi-scene object detection evaluation; no physical laboratory testing of the six generated concrete formulas; single-parent sources in several other studies; molecular targets and clinical external outcome validation outstanding. Other 11 current units retain earlier evidence checks but **were not freshly retrained in Pass 48**. Figure appearance or saved-output checksum alone does not confer full unit certification. Never substitute invented outputs or synthetic fixtures for real held-out evidence.

The complete matching source scripts, fresh checkpoints, saved genuine test predictions, temporal calibration evidence, PDF, and exact 14-unit matrix are packaged as `Chapter6_Pass48_Reproducibility_and_Certification.zip` in the accompanying user delivery.
