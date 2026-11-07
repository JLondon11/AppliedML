# NEU-DET three-seed benchmark (executed 2026-11-07)

## Dataset integrity
Source: Kaggle NEU steel-surface defect database (user-uploaded `archive (2).zip`). Source SHA-256 `2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f`. 1,800 original 200x200 grayscale images, 300/class. The identical decoded image `train/images/patches/patches_105.jpg` was excluded because it duplicates `patches_101.jpg`. Final disjoint split: train 1080, validation 359, test 360. Kaggle original validation partition reserved as test; the original train portion was split into grouped blocks for validation. Only real training data were used to train the GAN.

## Evaluated models
- Same lightweight CNN architecture, real images only
- Same CNN, training-time random horizontal flips
- Same CNN, real training images + 120 generated samples/class from conditional DCGAN
- Generator 300 optimizer steps/seed, classifier 12 epochs/seed, fixed validation-balanced-accuracy checkpoint selection
- Three training seeds: 20261106, 20261107, 20261108
- No test-based hyperparameter selection.

## **Measured results**, held-out 360 real test images

| Arm | Test accuracy, mean ± SD across seeds | Macro-F1 mean ± SD |
|---|---|---|
| Real only | 0.7194 ± 0.0774 | 0.7066 ± 0.0901 |
| Horizontal flip | 0.7000 ± 0.0528 | 0.6910 ± 0.0551 |
| Conditional GAN augmented | 0.5269 ± 0.0811 | 0.5004 ± 0.0754 |
| Size-matched real-image repeats and flips | 0.7963 ± 0.0243 | 0.7875 ± 0.0284 |

Paired GAN-minus-real accuracy difference: -0.1926 (about **-19.26 percentage points**), negative for all three seeds. A nominal t-based 95% interval over three seed differences is [-0.2070, -0.1782], **not** a robust population uncertainty interval: seeds share the same finite test set and are not independent datasets. Mean handcrafted feature distances from real data are large, and samples show repetitive defects/textures; neither the proxy nor sample appearance is equivalent to validated FID/KID or expert judgments.

The preceding 60-step pilot yielded the opposite ranking: 0.4528 real-only vs 0.5222 GAN. The replicated experiment therefore supersedes the pilot for the reported protocol. **Do not claim GAN augmentation improves defect inspection here.**

## Provenance and code
`run_multiseed.py` in this directory implements the replicated benchmark. The fully executed local source (`run_multiseed.py` and `neu_reference.py`), all checkpoints, seed-level predictions, manifest, training histories, plots and per-class analysis are in the user deliverable `Chapter6_NEU_Replicated_Benchmark_Pass13.zip`. The repository runner imports the corresponding core architectures from `run.py`; algorithms and defaults are matched. An independent clean-clone rerun of the repository runner/CI has not yet been completed.

Run:
```bash
python chapters/generative_ai_part_i/code/section_8_8_2_neu_defect_benchmark/run_multiseed.py --data-root /path/to/NEU-DET
```

## Scientific limitations
Only three seeds and a single fixed held-out split. No convergence guarantee for a 300-step conditional GAN. The horizontal-flip arm is not matched in training sample count to GAN augmentation, so gains/losses may be confounded by optimization exposure. No established pre-trained generative quality evaluator, expert labels on generated samples, external test set, architecture ablation, or compute-matched non-GAN synthetic-control arm. Because generating valid defect images is a key case-study requirement, full Master QA **HOLD** remains appropriate. This is a completed measured comparison for the stated protocol, not general evidence about GAN augmentation.

## Matched-size non-generative control (completed)
The additional control draws 120 training photographs per class with replacement from the original training partition, applies random horizontal flips to half the extra copies, and appends 720 real-derived examples, matching the GAN arm's added sample count. Under three matched seeds and the same 12 classifier epochs, the test accuracy was 0.8167 / 0.8028 / 0.7694; mean 0.7963, sample SD 0.0243. Macro-F1 mean 0.7875, sample SD 0.0284. This is **26.94 percentage points higher** than the GAN arm. Detailed results and scientific LaTeX report are in the Pass 13 deliverable ZIP. The conclusion remains specific to this short GAN training budget; generator fidelity, training convergence, and generality remain unverified.
