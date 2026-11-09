# Chapter 6 — Cross-domain synthetic data evaluation (Pass 21)

## Purpose and limitations

This executed diagnostic distinguishes **real NEU steel-defect photographs** from **real texture photographs** using (A) genuine NEU source training images or (B) the conditional GAN outputs from the previously executed Chapter 6 NEU benchmark as source training images. It is not an executed multi-domain generative transfer model, medical/driving experiment, or full generative-fidelity certification.

- Genuine NEU test: 360 held-out defect images; original 1,080 NEU training images available, 120 randomly sampled for this control.
- Genuine target: scikit-image brick, grass and gravel 512×512 photographs, partitioned into 120 upper spatial tiles for training and 72 bottom spatial tiles for testing.
- Training per arm: 120 source images + 120 real texture images, equal image counts.
- Three previously trained GAN seeds, 20261106 / 20261107 / 20261108; the real-source reference uses the **same fixed** NEU source training subset, not three independent baseline seeds.
- Feature-engineered grayscale statistics and standardized logistic regression; held-out domain balanced accuracy and ROC AUC.
- Original Kaggle archive source-image SHA-256 checks guard against split confusion.

## Measured scores

| Source training examples | Balanced accuracy | Domain ROC AUC |
|---|---:|---:|
| Genuine NEU photos (fixed reference) | 0.9611 | 0.9961 |
| GAN seed 20261106 | 0.6333 | 0.4002 |
| GAN seed 20261107 | 0.6514 | 0.6010 |
| GAN seed 20261108 | 0.6958 | 0.7057 |
| GAN mean ± sample SD | 0.6602 ± 0.0322 | 0.5689 ± 0.1552 |

The result is a negative generator-domain fidelity diagnostic: training with saved synthetic defect images did not generalize to recognizing genuine held-out defect photographs as the source domain. It is not an independent replication of the source GAN experiment or evidence of general augmentation efficacy.

## Run

Requires numpy, Pillow, scikit-image, sklearn, torch, matplotlib.

```bash
python run.py --neu-zip "/data/archive (2).zip" --gan-results /data/NEU_pass13/results --output results
python test_outputs.py
```

The Pass 21 chapter deliverable ZIP includes the locally executed implementation, full numerical test predictions, generated figures and manifest, plus three saved GAN-output tensors inherited from the executed NEU benchmark. Source data and checkpoint provenance are documented in the original NEU Pass 13 package. The GitHub runner is algorithmically equivalent but has not itself undergone an independent clean-checkout execution.

## Certification

The cross-domain application advances from no evidence to **partial executed real-data evidence**. Full certification remains **HOLD**: only a source-versus-texture diagnostic was run, not actual task-matched generative augmentation across medical, industrial, and driving domains; target test tiles originate from just three parent photographs, yielding correlated evidence.
