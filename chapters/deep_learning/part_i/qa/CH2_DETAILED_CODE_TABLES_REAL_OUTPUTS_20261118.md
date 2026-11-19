# Detailed Chapter 2: full code, substantive table, and actual-output scientific pass
**Date:** 2026-11-18. **Authoritative source:** the 166-page, smaller-font, detailed `Deep_Learning_Part_I_InProgress.tex` edition, not the distinct 118-page short edition. Governing science standard: proportional R1–R10 with scopes A/B/C.

## Exact source audit and manuscript changes

The detailed source has been rebuilt locally with 2 complete pdflatex passes to a **118-page scientific editing draft**. The page-count reduction follows purposeful removal of 65 verbose/nonessential code listings and 23 generic tables plus 45 historical editorial code-fix notes. The main technical sections, mathematical derivations, named applications and case studies have not been swapped for the shorter edition. The current PDF SHA256 is `b9a537bbe081ec99cfc698cd978a8653a1c38e1eb4301dd59f59142ff0c6385f`.

- **Printed code:** from 176 blocks (~2,805 source lines) to **111 focused blocks (~851 lines)**. Exactly 65 long, import-only, output-only, notebook-only or invalid code fragments were archived separately with the original text. Each remaining printed block is <=25 lines and passed Python AST syntax parsing; none of these 111 fragments are claimed complete, independently executed application programs solely from syntax checks.
- **Printed commentary:** 45 version-to-version correction notes were archived outside the book, preventing editorial remediation history from masquerading as textbook narrative. Original experiment outcomes whose sources were not actually executed were rewritten as experimentally conditional, especially UCI HAR LSTM/GRU comparison and the Stanford Dogs fine-tuning ablation.
- **Tables:** 31 generic/redundant/unsubstantiated tables replaced with **8 technically substantive** tables: **5 built directly from saved authentic NGDC sunspot and original Digits PyTorch CNN results**, **2 based on independently checked exact parameter/output-dimension formulas**, and **1 reporting scientifically scoped domain inference limits**. Actual 450-image original Digits class supports and six CNN errors were recalculated from original per-image labels and predictions. Table figures are not fabricated.
- **Figures:** all existing 71 original/previously integrated figure environments were preserved, and two new authentic-data results were inserted, giving **73 total**. **42** have complete external assets, **2** are native TikZ, **28** still lack some original artwork, and **1** remains empty/commented. The compiled PDF still has **35 fallback placeholder references**. The release is **HOLD** until the missing original empirical outputs and assets are recovered or generated from actual data. Earlier Jordan/Elman and unfolding figures were inserted and visually inspected; architecture diagrams do not pass empirical R8 by themselves.

## Two new original-data experiment scopes, actually executed

### Scoped Level B — original PyTorch CNN image classification
Executed a two-convolutional-layer PyTorch CNN on **1,797 original observed scikit-learn 8×8 Digits images**, with subject/image-ID-disjoint 1,077 train, 270 validation, 450 test images, training-only model/checkpoint selection, actual saved full test probabilities, labels, confusion matrix, and macro-F1. Test: **444/450 correct, accuracy 0.9866667, macro-F1 0.9865809**, selected validation epoch 20 of 35. Matched logistic regression using the same original training and held-out test image identities scored **0.9622222**, with no general superiority claim. The genuine four-panel output figure is integrated as **Figure 2.30** (PDF page 57). The **8×8 Digits are not the named MNIST 28×28 dataset**, and this study does not certify the separate MNIST/CIFAR applications.

Full executable source and original-data metric verifier:
`chapters/deep_learning/part_i/experiments/detailed_real_digits_pytorch_cnn/run.py` and `test_outputs.py`. The local source experiment was run twice with byte-identical evidence JSON and independently verified complete held-out metrics. CI workflow has been authored, but no passing online workflow result was confirmed.

### Scoped Level A — initialized recurrent-input-gradient sensitivity
Executed **18 actual recurrent initialization/automatic-gradient computations**, on **309 original NGDC annual sunspots** (1700–2008), for tanh RNN, LSTM and GRU at three seeds and two input lookbacks, with 24 true observed training windows per run and exact input-lag-gradient arrays saved. Median oldest/newest gradient ratio for 36-year windows: RNN **2.7658e-9**, LSTM **4.7630e-8**, GRU **3.0208e-8**. A true-results figure is integrated into the accurately retitled `Case Study: Initial Input-Gradient Attenuation on Observed Sunspot Sequences`. This is a sensitivity-at-initialization experiment, not a trained long-horizon-memory benchmark or comparison of classification accuracies.

Full executable source and independent integrity verifier:
`chapters/deep_learning/part_i/experiments/detailed_real_sunspots_gradient/run.py` and `test_outputs.py`. Two local runs gave byte-identical saved numeric evidence arrays.

### Previously supported, distinct scoped extension
An independently executed NGDC annual sunspot **one-step forecasting** LSTM/GRU example remains part of this detailed chapter's evidence (309 genuine observations; 178/60/59 window split, six recurrent fits, measured held-out forecast errors). It **does not** validate the separate airline-passenger model's original empirical outcome or UCI HAR classification.

## Scientific certification status under proportional R1–R10

The **12 original named detailed-chapter application/case-study headings** have now received exact-title evidence-level assessment. **Two narrowly scoped units have scientific A/B pass evidence** (original 8×8 PyTorch Digits CNN, and original annual-sunspot initialization-gradient mechanism), with **10 other original named task-level experiments HOLD**. Every remaining named application lacking original dataset-executed algorithm, per-observation results or original result figure is marked `HOLD`, rather than converting architecture diagrams, synthetic fixtures or pasted code into fake experimental evidence.

Outstanding datasets/results include original airline forecasts, raw UCI HAR LSTM and matched LSTM-vs-GRU sequence classification, actual storm-tagged cyclone intensity imagery and independent evaluation, named **MNIST 28×28** and original CIFAR-10 studies, Stanford Dogs pretrained inference and fine-tuning ablation, actual image-style and real-video style-transfer output plus temporal evaluation. Workflow proposals at the end of the chapter do not count as executed real-data experiments.

The corresponding full **12-unit CSV science matrix**, **73-figure CSV asset ledger**, **8-table scientific-basis audit CSV**, and **111-listing syntax audit CSV** are in the downloadable local package.

## Remaining release holds

Current manuscript still has **35 printed fallback figure placeholders** and **11 overfull boxes**, and many original empirical experiments are unexecuted or their authorized data/weights unavailable. **Full detailed chapter scientific certification HOLD; complete print/figure quality HOLD; book Master QA HOLD.**

Complete revised LaTeX/PDF/binary figure assets/observed numerical results are in `Chapter2_Detailed_Code_Tables_RealData_Remediation_20261118.zip`, but publication of the entire ZIP contents as the canonical GitHub manuscript/figure binary tree has **not** been confirmed. The new real-data source programs and workflow are committed; GitHub Actions success is not confirmed.
