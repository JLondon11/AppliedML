# Strict case-study/application real-data figure audit — 2026-11-09

## Scope and evidence standard

This pass inspected the **current complete GitHub file tree** at `JLondon11/AppliedML` and registered recognizable experiment and case-study code groups. Its goal is to distinguish **committed code**, **committed numerical records**, **committed result figure files**, and the separate, harder scientific tests of genuine dataset provenance, measured model execution and manuscript placement. Real photographs are genuine data. Actually generated model output images and their measurements are valid when properly identified; invented observations or invented performance numbers are never acceptable.

**This is a source-tree coverage audit, not a completed line-by-line audit of every manuscript, model run, or image pixel.** Not all manuscript chapters are currently represented by standalone code directories, and outputs archived outside the unit folders may require separate lookup.

## GitHub-backed findings

| Evidence signal among 54 discoverable code-bearing units | Count |
|---|---:|
| Registered code-bearing units | 54 |
| Unit-directory README found | 35 |
| Numerical results or provenance files found under unit path | 22 |
| Result figure binaries/vector assets found under unit path | 6 |
| Units with README + numerical result files + result figure assets all visible in the same unit path | 0 |

The counts reflect **directory evidence only** and must **not** be interpreted as 54 failed scientific studies or as a true measure of every manuscript section. Files may reside in external chapter ZIPs, Actions artifacts, distinct figure directories or other paths. Conversely, a committed figure alone does not prove that it was generated from authentic measurements.

Machine-readable path evidence and per-unit gaps: `qa/REAL_DATA_CASE_STUDY_FIGURE_EVIDENCE_AUDIT_20261109.json`.

## Targeted manual spot checks

**Chapter 5 — Segment Anything/COCO:** The committed README identifies original COCO validation photographs and annotation masks, a named checkpoint, measurement counts, paired IoUs, asset hashes and a reported successful CI execution. This is comparatively strong evidence, but the full independently reconstructed figure-to-source audit and compiled manuscript certification were not repeated in this pass.

**Chapter 6 — diffusion digits:** The README identifies genuine scikit-learn Digits images, a 1,257/270/270 split, actual model-generated images and measured conditional/unconditional output scores. Generated samples are legitimate generative-model outputs when labeled correctly; however, README states the full figure, artifact and checkpoint package is in a release ZIP rather than committed entirely to GitHub. Independently checking the archived outputs against the manuscript remains necessary.

**Chapter 9 — BERT IMDb confusion matrix:** The figure-generator source trains a DistilBERT classifier using named IMDb train/test records and derives confusion-matrix values from predictions. This is stronger than a hard-coded matrix, but source inspection alone is not proof that the script completed and the displayed manuscript image matches saved real predictions.

**Chapter 6 — NEU photographs and GAN output domain comparison:** NEU steel-defect photos and texture photos are real data. A GAN trained on authentic imagery produces real **model outputs**, which may legitimately be evaluated and compared with genuine held-out photographs, if source and generated images are clearly distinguished. This experiment is **not prohibited simply because it uses generated outputs**; its scientific execution, data splits and figures still require verification.

## Required certification checks for each application and case study

1. Identify its actual manuscript section, title and all tables and figures.
2. Confirm a correctly cited, authentic dataset, its license or access source and unchanged identifiers or hashes.
3. Verify actual executed model/algorithm, fixed code/version, random seeds, split sizes and held-out evaluation controls.
4. Check prediction records, confusion matrices, time series and metrics numerically against saved outputs; never infer from captions.
5. Re-run the figure script using measured data, compare outputs to manuscript images, and inspect legends/units/captions scientifically.
6. Confirm code, figure generators, result files and approved figure assets exist in GitHub or in explicitly documented, independently accessible versioned run archives. GitHub source publication is mandatory.
7. Certify only after every requirement passes; until then mark the unit **HOLD**.

## High-priority gaps

- Chapter 2's MNIST, CIFAR-10, HAR, hurricane-intensity, dog-breed, optical-flow and style-transfer units still lack their complete named-dataset evidence, even though authentic Digits and sunspot supporting studies exist.
- Complete artifact and manuscript-to-code mapping remains incomplete for all chapters. A code-group registry does not establish how many actual applications or case studies occur in the manuscript.
- Sections that currently point to release ZIPs or CI-only figures must have those outputs retrieved and checked; if missing, publication status stays HOLD.
- All other chapter units not enumerated in the 54-group directory discovery remain **NOT YET INVENTORIED**, never silently certified.

**Decision:** Global scientific and publication certification **HOLD**. No universal all-chapters compliance claim is warranted.
