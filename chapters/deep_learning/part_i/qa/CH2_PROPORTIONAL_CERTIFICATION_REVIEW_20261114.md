# Chapter 2 — proportional R1–R10 scientific reassessment (2026-11-14)

**Governing rule:** repository-root `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` (A/B/C science levels, scientific and production approvals separately). The previous 2026-11-09 zero-of-eleven strict audit is retained unchanged as historical evidence. Its rule requiring every chapter unit to use a uniform experiment protocol is not the current policy.

## Chapter identification reconciliation

The AppliedML canonical manifest names `chapters/deep_learning/part_i` **Chapter 2 — Deep Learning Part I**. The older `All_Chapters_Complete_12.zip` stores the same subject in `all_chapters_delivery_v2/03_Deep_Learning_Part_I/`, whose number differs. Never use the archive's `03_` folder number to rename the canonical Chapter 2 or overwrite another subject's manuscript. The latest canonical source still requires explicit reconciliation before production release.

## Independently executed genuine experiment

The real NGDC annual sunspot observations distributed by `statsmodels.datasets.sunspots` contain **309 observations (1700–2008)**, with exact float32 SHA-256 `9bcbe9dcdc8c75ef925a2f360ab677c2d84a17c6e65a8cf6644be24b82c5be09`. The 12-year input windows have target-year chronological partitions **178 train (1712–1889), 60 validation (1890–1949), 59 held-out testing (1950–2008)**. Only 12 *previously recorded* annual observations are input for each forecast; this is one-step, observed-history forecasting, **not** a recursive multi-year simulation.

I independently executed the archived source `ch2_sunspots_lstm_gru/run.py` for six real PyTorch models: LSTM/GRU × initialization seeds 17/42/84, each with 16 hidden units, validation-driven early stopping and three real-output figures. The run produced 354 held-out predictions, six genuine PNG/PDF figure assets, all validation loss histories, original target years and original observed targets.

The independently computed results from the new run are:

| Architecture | Three-seed test RMSE mean | Sample SD | Description |
|---|---:|---:|---|
| LSTM | 20.6795657 | 0.4830048 | Actual selected checkpoints, 59 original test years per seed |
| GRU | 20.6677302 | 2.1250781 | Same original dataset/partition |
| Persistence | 33.1750065 | N/A | Copies previous observed year's count |

The actual forecast values were rechecked against the saved test metrics; the source dataset digest, 309 original observations, chronological partitions and seedwise aggregates all passed the independent verifier. Compared with the prior saved reference run, the maximum test RMSE discrepancy was **1.535×10⁻⁶** and the maximum prediction difference **1.526×10⁻⁵**, attributable to minor numeric backend variability. No claim is made of byte-identical execution. Source and evidence archive are available as `Chapter2_LSTM_GRU_Real_Sunspots_20261109.zip`; the full experiment implementation is also available under `chapters/deep_learning/part_i/experiments/sunspots_lstm_gru/run.py` in GitHub (same protocol, refactored source), with a committed scoped numerical validator at `chapters/deep_learning/part_i/qa/validate_scoped_sunspots_evidence.py`.

## R1–R10 scientific decision — Chapter 2 Unit CH2-01

**Original title:** Application: Time Series Prediction with LSTM Using PyTorch.

**Assigned Level:** **B** — complete offline experimental application, scientifically certified **only** for *one-step forecasting of recorded annual sunspot counts using a 12-year observed input history and an LSTM model*. This is not certification for other generic time series or forecast horizons. The equal-split GRU model is a measured comparator, not the core requirement of CH2-01.

| Criterion | Decision | Specific basis |
|---|---|---|
| R1 | PASS | 309 authentic NGDC annual observations, original record range and numeric SHA-256 verified |
| R2 | PASS | Chronological train/validation/test target-year split; train-only normalizers; no future target leakage; prior observed input years allowed |
| R3 | PASS | Six actual neural-network executions including 3 LSTM runs, fixed stated training protocol |
| R4 | PASS | Saved 59 genuine independent-period LSTM forecasts per seed, prediction years, all targets and training histories |
| R5 | PASS | Actual computed persistence control, plus matched GRU comparison on same data and dates |
| R6 | PASS | Actual MAE/RMSE, seedwise spread (three random initializations), error time series, honest small-sample limitations; no speculative confidence intervals |
| R7 | PASS | Real chronological observational dataset with offline forecasting scope; no unclaimed physical/clinical deployment validation demanded |
| R8 | PASS | Executed real-observation forecast overlay, multi-seed RMSE and training/validation histories; inspected raster plot and confirmed PNG/PDF artifacts |
| R9 | PASS | Rerunnable complete PyTorch source and scientific figures; saved complete result arrays, source/checkpoint-selection protocol and all seeds in archived execution package; matching GitHub implementation and executable verification code available. No external licensed model checkpoint is needed to rerun; storage of six separate `.pt` state files is not required to inspect actual predictions or regenerate the figure. Final release asset archival is a separate production gate |
| R10 | PASS | Result strictly limited to NGDC annual series and one-step observed-history forecasting; no GRU architectural superiority claim |

**Scoped scientific decision: CH2-01 SCIENTIFIC_CERTIFIED — LEVEL B, OBSERVED ANNUAL SUNSPOT FORECASTING ONLY.** Manuscript-production status: **HOLD**. The available ZIP does not verify the latest canonical Chapter 2 section prose, exact inserted figure assets/page placement, or the final production PDF. A broad unqualified forecasting claim would exceed this certificate.

## Other original units

**10/11 original named application/case-study units remain scientific HOLD** as they stand. Their required exact titled tasks remain to be executed; the existence of authentic sunspot regression, initial recurrent-gradient traces and 8×8 Digits outputs does not certify HAR LSTM classification, hurricane-satellite wind estimation, CIFAR-10 CNN, 28×28 MNIST CNN, Stanford Dogs transfers, original content/style transfer, video optical flow, LSTM/GRU *sequence classification*, trained deep recurrent vanishing-gradient analysis, or fine-grained transfer ablations.

Their proportional Level B R1–R10 review is registered in `CH2_PROPORTIONAL_R1_R10_REASSESSMENT_20261114.csv`. All unresolved criteria are marked **NOT VERIFIED**, not blanket FAIL for missing Level C features. Scoped standalone Level A subexperiments (e.g. initial-gradient response on actual sunspot sequences) should be registered separately only after source/figures and scoped titles are independently reviewed. They cannot silently certify unchanged broader titles.

**Scientific register:** 1/11 scoped original application titles have a passing **B-level science certificate**; 10/11 original units HOLD. **Production approval: 0/11.** The chapter and book are **not** scientifically/production complete.

## Follow-up priorities

1. Reconcile the newest complete canonical Chapter 2 source and figure directory with `03_Deep_Learning_Part_I` archive without corrupting chapter numbering. Integrate the certified NGDC-LSTM Figure(s) with dataset counts and honest scope; rebuild PDF and visually inspect.
2. Execute authentic UCI HAR LSTM/GRU **classification**, which can cover two named units and produce genuinely new classification outputs.
3. Execute actual 28×28 MNIST and CIFAR-10 CNN experiments, not replacement 8×8 Digits, then image predictions and error analysis.
4. Complete the remaining authentic astronomy/hurricane, dog-breed, image-style, video-flow and recurrent-gradient task-specific experiments with their own A/B/C scope assessments.

**Non-negotiable:** no fabricated data, invented predictions, unrelated dataset substitution or quiet change of an unchanged case-study title. No missing PDF build check may be confused with a scientific R1–R10 failure, but production approval remains a separate condition.
