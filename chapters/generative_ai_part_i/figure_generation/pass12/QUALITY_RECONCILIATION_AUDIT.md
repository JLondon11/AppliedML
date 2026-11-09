# Chapter 9 — Independent quality-evidence reconciliation

The local real-data DDPM/DDIM quality archive was audited against its own unaggregated measurements. **12 per-seed records** (4 sampler configurations × 3 seeds), each containing seven scientific diagnostics, reconciled with all reported means and sample standard deviations.

The quality and sampling evidence reference the same checkpoint SHA-256:

`a3f573596d581370325a400e34e9d07945e2953c0c1bf7d4f730ffff3140e53d`

The checkpoint bytes in the local evidence archive agree with this SHA. Eight existing numerical tests and three additional reconciliation tests passed locally, including two deliberate-corruption rejection tests. GitHub Actions has been updated to independently run `tools/scientific_qa/verify_quality_evidence.py`; **a successful remote workflow run is not yet confirmed**.

## Grounding and scientific limitations

The experiments use the 1,797 real images in `sklearn.datasets.load_digits`, split into 1,257 training, 270 validation and 270 independent test images. The classifier trained only on the training partition achieved approximately 96.3% independent-test accuracy. Its learned features support a **domain-specific Fréchet diagnostic, not canonical FID**. The feature distances demonstrate substantial distribution mismatch between generated and held-out images.

A nearest-training proximity threshold cannot rule out memorization; confident classifier predictions cannot establish visual realism; and raw pixel range violations remain significant. Three seeds are insufficient for broadly generalizing method rankings. The evidence supports a real-data-trained diffusion model comparison, but **not full generated-image-quality certification**.

## Reproduction

```bash
python tools/scientific_qa/verify_quality_evidence.py \
  --results chapters/generative_ai_part_i/figure_generation/pass12/outputs/quality \
  --sampling chapters/generative_ai_part_i/figure_generation/pass12/outputs/ddim_comparison \
  --checkpoint chapters/generative_ai_part_i/figure_generation/pass12/outputs/ddpm/best_ddpm.pt
```

The evidence verifier checks that the source checkpoint and the quality/sampling reports share the same SHA-256, that every expected method/seed has exactly one record, that numerical measurements are finite, and that the aggregated results reproduce the raw CSV arithmetic. Scientific interpretation, perceptual quality, and chapter PDF integration require separate review.
