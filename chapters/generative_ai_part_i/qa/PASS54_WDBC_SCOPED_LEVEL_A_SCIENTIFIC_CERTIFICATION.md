# Chapter 6 — Pass 54: original Wisconsin cytology conditional-VAE certification

**Decision:** `CH6-D07 SCIENTIFIC_CERTIFIED`, **Level A**, limited to an offline single-modality, historical original-data **conditional-VAE augmentation diagnostic**. This does **NOT** certify multimodal precision medicine, a clinical synthetic patient record generator, treatment response, privacy preservation, or clinical deployment. The parent precision-medicine application and latest full manuscript **remain HOLD**.

## Data, protocol and execution

The original Wisconsin Diagnostic Breast Cancer dataset (`sklearn.datasets.load_breast_cancer()`) contains **569 actual historical fine-needle cytology feature measurements**, with 30 real observed features per record and original diagnostic labels. Independently executed original-data SHA-256 of concatenated float32 feature and int64 label arrays:

`f86286b06e0eb4fbbb699d6ba5df7f1432e65642369ae111d0e47b47d02338bc`.

Fixed stratified seed 20261109 gives **341 training / 114 validation / 114 held-out test** genuine records. The StandardScaler is fit only on training records and transformed features are clipped to [-8,8]. Three distinct conditional-VAE training seeds **11, 23, 37** each ran **35 epochs**, with validation-selected checkpoints (all selected epoch 35). This is therefore **not** evidence of training convergence, out-of-hospital generalization or independent-seed test populations.

For each seed, 341 model-generated numerical feature vectors (same training class counts) were used in the synthetic augmentation arm. An original-real-only classifier, a real-plus-matched-bootstrap classifier and a real-plus-CVAE classifier were trained. All three arms were evaluated on the **same 114 original held-out records**, giving **nine model-test evaluations = 1,026 prediction rows**, not 1,026 independent records. Other retained results: 1,023 generated numerical feature rows, 105 original per-seed epoch records, three trained CVAE checkpoints and a measured experimental figure. Code uses original real-data sklearn and PyTorch implementations, not mock scientific inputs.

## Actually observed original test performance

| Model training arm | Mean genuine-held-out ROC AUC (3 seeds) | Mean genuine-held-out balanced accuracy |
|---|---:|---:|
| Real-only | **0.9993386243386243** | **0.9930555555555557** |
| Real + matched real resampling | 0.9986772486772487 | 0.9755291005291006 |
| Real + CVAE-generated numerical records | 0.98070987654321 | 0.9470899470899471 |

**Negative outcome:** the synthetic-augmentation arm **underperformed** the original-only baseline and matched genuine-data bootstrap on this historical split. Three seeds change model initialization but **do not** provide three distinct cohorts; the real-only classifier is deterministic and its same score was repeated across the seeds. No robust population-level confidence interval or clinical effect claim follows from this experiment.

## Independent checks and code storage

The 1,026 per-record held-out probabilities and truth labels were independently reaggregated, reproducing every original per-seed recorded ROC AUC and balanced-accuracy value. The source SHA, 341/114/114 unique split, three training histories (105 rows), 1,023 generated numerical records, three trained original-model checkpoints and the two-panel original-test figure were inspected. Pass 54 source rerun completed locally.

The following source is committed to GitHub:
- `code/section_6_precision_medicine_cytology_cvae/run.py` — existing full authentic original-data CVAE experiment.
- `code/section_6_precision_medicine_cytology_cvae/independent_cytology_pass54.py` — independently executed original dataset replay code and plotted outputs.
- `code/section_6_precision_medicine_cytology_cvae/verify_pass54_original_data.py` — fail-closed source split, original-test score, saved generated record, and checkpoint audit.
- `code/section_6_precision_medicine_cytology_cvae/pass54_original_test_metrics.csv` — all nine genuine held-out seed/arm source scores.
- `qa/CH6_SCOPED_SCIENTIFIC_VS_PRODUCTION_REGISTER_20261113.csv` — separately scoped certificate record.

Actual trained weights, large per-record generated vectors, all 1,026 predictions and figure files are retained in the separately supplied Pass 54 experimental evidence package; they must not be falsely described as pre-existing files in GitHub. Source code and the measured score table **are** in GitHub. The saved original-source dataset is available through sklearn.

The new figure `wdbc_original_test_ablation.png/.pdf` shows seed-wise original held-out ROC AUC and balanced accuracy, with retained means and professional muted color encoding. The figure uses actual recorded run values and contains no invented scientific bars. Production integration and current manuscript figure numbering remain unchecked.

## Proportional R1–R10

| Requirement | Decision | Scope/evidence |
|---|---|---|
| R1 genuine data provenance | PASS | Original public WDBC measured source, SHA, labeled 569 records |
| R2 split/leakage | PASS | Stratified unique original 341/114/114 IDs; original-train-only fitting |
| R3 actual executed model | PASS | Three genuinely trained tabular CVAEs plus nine trained/evaluated logistic classifier instances |
| R4 retained actual outputs | PASS | 1,026 original held-out per-patient score rows, 1,023 model-generated vectors, checkpoints |
| R5 method-appropriate controls | PASS | Original-only and class-balanced genuine-real bootstrap arms |
| R6 measured errors/limitations | PASS | Actual ROC AUC, balanced accuracy, macro-F1 and per-seed negative result; shared-test caveat |
| R7 domain checks | PASS only for original-record offline diagnostic | Genuine original cytology feature/classifier evaluation; **not** clinical efficacy |
| R8 actual scientific figure | PASS for scoped experiment | Source-derived two-panel original held-out result and generator |
| R9 full experiment source storage | PASS | Original full model, independent replay, verifier and measured per-seed CSV committed to GitHub |
| R10 honest scoped conclusions | PASS | Negative single-modality augmentation result, no claims about clinical safety or multiomics |

**Certification result:** Level A **scientific certification for the exact restricted offline experiment**, not certification for the broader precision-medicine case study or clinical deployment.

## Updated Chapter 6 accounting

**Seven scientifically certified narrowly defined experiments: six Level A and one Level B.** This count is not seven of the fourteen broad case studies certified. Broader driving perception (actual BDD100K detection studies), prospective novel-concrete physical strengths, the entire medically relevant multicomponent application and final manuscript figures remain separate evidence/production gates.

**Production approval: HOLD** until exact current compiled chapter assets, figure/caption integration and editorial QA have been independently inspected.
