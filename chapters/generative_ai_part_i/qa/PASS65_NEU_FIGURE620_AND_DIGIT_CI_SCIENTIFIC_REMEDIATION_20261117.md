# Chapter 6 — Pass 65: original NEU science, Figure 6.20 provenance and digit-generator CI remediation

**Date:** 2026-11-17 (Arizona local). **Decision:** Real source-derived scientific presentation and actual GitHub Actions failure analysis/repair completed. Twelve existing narrowly scoped scientific certificates remain separate from fourteen broader parent-study scientific decisions. **Final Chapter 6 manuscript production remains HOLD** without the latest actual compiled PDF and LaTeX.

## 1. Original handoff carefully interpreted

The directly inspected user-supplied `Proceed Chapter 6.txt` reports a historical Pass37 *109-page* compiled chapter with industrial-defect Figure 6.20 on **printed page 67** beside its caption and narrative; problems begin on **printed page 101**. The named Pass37 PDF, TeX and ZIP files are **not present** in the current accessible sandbox and are not in the connected repository. The note provides credible **historical placement evidence**, **not contemporaneous verification** of the latest compiled Figure 6.20 image source, experimental lineage or page.

The current `quality_assurance/chapter6_evidence_manifest.json` explicitly identifies a source-domain conflict: a provisional **handwritten-digit** `figure_20.png` script candidate conflicts with the manuscript's reported **NEU industrial-defect** study. Filename or figure-number resemblance is not scientific provenance. That candidate remains unverified and is NOT used as a replacement.

## 2. Fix an actually false chapter experiment README

The main `code/section_8_8_2_neu_defect_benchmark/README.md` still stated the **Pass10 provisional experiment was unexecuted**, although the same committed folder contains `REPLICATED_BENCHMARK_PASS13.md` and 12 real source-seed rows representing a completed three-seed four-arm original-data experiment. It was corrected to describe **what actually ran**, distinguishing original Pass10 software fixtures from legitimate Pass13 empirical results.

Authentic NEU-DET photographed source: **1,800 original 200×200 grayscale defect photographs**, six classes. One duplicate original source image was excluded from development, leaving **1,080 training / 359 validation / 360 test source photographs**. Original source archive SHA256 `2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f`. Three seeds **20261106, 20261107, 20261108** with the same 360 held-out real test images and four classifier training arms:

| Original experimental arm | Observed test accuracy mean | Observed macro-F1 mean |
|---|---:|---:|
| Original photos only | 0.71944444 | 0.70655892 |
| Horizontal flips | 0.70000000 | 0.69098885 |
| Conditional GAN-augmented | **0.52685185** | **0.50038539** |
| Sample-size-matched re-used original photos and flips | **0.79629630** | **0.78747116** |

The conditional GAN **reduced held-out accuracy by 19.26 percentage points** versus original-only for this short-trained original-data experiment. That result must not be framed as successful generative improvement. The same test dataset is reused across seeds; no 4,320-independent-test-image claim. Original genuine defects and model-generated defect images are scientifically distinct.

## 3. New relevant, non-plot scientific original-data figure

A new scientific **dataflow/experimental protocol diagram with real measured score table** was executed from `replicated_seed_metrics_pass13.csv` using newly committed `code/section_8_8_2_neu_defect_benchmark/make_neu_original_study_diagram.py`. It depicts the original photographed NEU source, duplicate control and dataset splits, four training arms on the original test set, and independently re-averaged genuine per-seed held-out accuracy and macro F1. It is neither a generic bar graph nor synthetic defect imagery, nor an invented benchmark.

- Source original measured CSV SHA256: `1f74ab0c8067612a072c2eff5554b971682ce851bd4c7ca5eb585fbb5f163d9e`.
- Actual locally generated, visually inspected vector figure: `neu_real_experiment_protocol.pdf/.svg/.png`, original staged PDF SHA256 `b0cd35390c911a280a255a412a861fc2084c030ec333adf6ed57aec25cdec5e8`.
- The real generated PDF is exactly **468 pt = 6.5 inches wide**. Actual measured minimum selectable font **8.1 pt**, no off-page text and clean visual rendering.
- Original paper/source figure PDFs and CSV bundled for user delivery in `Chapter6_Pass65_Neu_Digit_CI_Scientific_Remediation.zip`.
- `.github/workflows/chapter6_neu_real_science_protocol.yml` **actually ran and passed** as GitHub Actions run **44209396903**, checking source measurements, source hashes, correct clinical/production disclaimers, output PDF/PNG/SVG and text legibility.

The evidence manifest now lists the diagram as an **unnumbered candidate industrial-defect figure**. It is **NOT asserted to be the manuscript's existing Figure 6.20**. It cannot replace required genuine model photo/result galleries where the manuscript discusses visual defect fidelity. Final printed page placement remains unverified.

## 4. Correct real digit-model verification, not model data

The original independent real handwritten-digit trained model-rerun failed because its plotting code asserted that **all numerical re-executions must exactly match archived original conditional 778/1000 and unconditional 94/1000 scored images**. The actual GitHub log of failed run **43973163203**, digit job **133670286149**, proved that the complete real model training and prediction QA had passed, but the fresh measured generator run produced **779/1000** conditionally correct and **94/1000** unconditional. The plotting script failed at the hard-coded numerical assertion. The difference of one label is a legitimate numerical-rerun difference; it is not evidence that the model failed scientific evaluation.

- Original `figure_6_44_publication.py` has been modified to calculate observed totals **from its actual 2,000 freshly generated prediction rows**, retaining archived 778/94 only as *explicit optional* strict replay controls.
- New `derive_original_run_class_counts.py` derives actual per-arm/per-digit counts, each over 100 genuine executed output images, and checks them against the current run's JSON manifest. The workflow uses this **current-run CSV rather than the archived Pass27 class table**.
- GitHub Actions run **44209077569** (commit `325d98960c...`) has already **fully succeeded** for *both* tissue and digit jobs after the first hard-coded-count fix, confirming that the original model science is executable.
- A later source-consistency run **44209130138** exposed a **second hard-coded 778/94 assertion and caption** in `plot_verified_original_class_counts.py`. Its log confirmed genuine generated run results **779/94**, successful prediction checks and successful Figure6.44 building, followed by failure of the stale classwise count figure.
- That second script now reads the measured current-run class summary, checks per-class/overall numerical reconciliation, prints correct updated caption counts, and optionally validates the exact archived numbers only when explicitly requested.
- A new `test_current_run_count_figure.py` regression test uses a software-only altered count fixture to ensure that 779 is accepted for a new source and 778 can still be checked against archived original data. **Fixtures are code tests, never claimed as measured model results.** The true GitHub Actions tissue-and-digit workflow now runs the regression before genuine training.
- As of this Pass65 record, newer GitHub Actions workflow runs from these latest corrections may still be active/queued; do not call those new specific runs successful until their final conclusions are independently verified.

## 5. Scientific versus production decision

**Scientific status:** The previously certified twelve scoped Chapter 6 original-data demonstrations remain correctly registered. NEU CH6-D04 remains a narrowly titled **Level B certified negative GAN-augmentation benchmark**. Label-conditioned original-digit diffusion remains a focused Level A study; **neither** newly drawn diagrams nor CI improvements count as independent new scientific case studies. Broader individual application titles still require matching original domain evidence.

**Production status:** NEU study README now accurately presents original real-data evidence, the new scientific candidate diagram is source-verifiable and print-legible, and automated figure QA has an actually confirmed successful run. But the latest compiled full chapter PDF/LaTeX, 48-figure publication mapping, Figure 6.20 original image identity, all tables/listings and end-of-chapter pagination remain unverified in this session. **Final Master QA HOLD remains correct.**
