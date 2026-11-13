# Chapter 6 — BDD100K multi-scene road-object crop augmentation

**Status: SCIENTIFIC HOLD (data and executed benchmark missing).** The full Python runner **is committed** at `chapters/generative_ai_part_i/code/section_6_driving_bdd_multiscene_validation/run.py`, with its input-integrity fixture tests at `test_input_gates.py`. The previous statement that the runner had not been committed was outdated. **Committed source is not equivalent to a completed real-data experiment.**

## Required real dataset
The code expects genuine BDD100K 2018 detection-label JSON files and matching training and validation photographs. Neither genuine multi-scene execution logs nor actual source-derived scores are established in the currently reviewed evidence. Dataset references: https://github.com/bdd100k/bdd100k and https://dl.cv.ethz.ch/bdd100k/data/ .

## Precisely implemented method
The runner reads authentic annotated object boxes for car, person and traffic light; crops photographs to 64×64; separates train/validation/test by parent frame; rejects image-pixel duplicates across subsets; trains a class-conditional convolutional VAE on original training crops; compares classifiers trained with real-only, real-bootstrap and generated-CVAE augmentation. Three model seeds are included. It can save split manifests, trained CVAE checkpoints, held-out test probabilities, per-seed accuracy/balanced accuracy/macro F1 and actual generated crop/example figures.

**Scope limit:** The target is held-out **road-object crop classification**, **not** whole-scene synthetic driving generation, object detection mAP or deployment safety. Passing fixture-based input tests proves only input-gate behavior. It is not evidence of a real driving experiment.

## Execution and evidence gate
Example (replace paths only with genuine downloaded datasets):

```bash
python chapters/generative_ai_part_i/code/section_6_driving_bdd_multiscene_validation/run.py \
  --train-images /path/to/bdd100k/images/100k/train \
  --val-images /path/to/bdd100k/images/100k/val \
  --train-labels /path/to/bdd100k/labels/bdd100k_labels_images_train.json \
  --val-labels /path/to/bdd100k/labels/bdd100k_labels_images_val.json \
  --output /path/to/bdd_real_result
```

Before scientific certification, independently verify input hashes, original split counts and frame separation, actual training logs, per-sample real held-out predictions, model-generated crop quality, source-linked figures, quantitative controls, and limitations. Claims of downstream **detection** improvement require an additional real held-out detection mAP experiment with whole-scene synthesis where applicable. Do not report absent metrics as measured results.
