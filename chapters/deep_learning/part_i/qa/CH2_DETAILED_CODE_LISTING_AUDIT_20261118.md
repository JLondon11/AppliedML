# Chapter 2 detailed edition — audited printed Python code and runtime scope

**Date:** November 18, 2026. **Authoritative branch:** detailed/small-type Deep Learning Part I; original input `Chapter2_Detailed_AirPassengers_RealData_Remediation_20261118.zip`, not the separate short Pass 7 edition.

## Actual verified changes
- All **111** printed code listings extracted from the actual LaTeX source, all at most **24 lines**, all now parsed by the Python AST. Previously, five failed syntax because of unexpected indentation.
- Source-sensitive repairs to the original UCI HAR demonstration: use authentic **128 × 9 inertial time windows** (not 561 static feature summaries masquerading as sequence steps); prevent subject/label columns from becoming model inputs; fit/select weights on subject-disjoint development groups; use correct sparse categorical label loss. The original UCI HAR model has **not** been trained or evaluated here; no experiment result inferred.
- Repaired deprecated pandas feature-reading examples and the MNIST test-data-loader ordering, computed dog-classification match aggregation, and separated incompatible validation loss and accuracy axes.
- Replaced deprecated and unverified YouTube `pafy` source downloading code with an authorized local-video input check. No real video inference is claimed.
- Preserved earlier verified original AirPassengers, observed sunspots and original 8×8 Digits PyTorch results. No neural-network metrics were changed in this pass.
- **All 73 figure environments and eight substantive tables remain in the rebuilt manuscript.**

## Production evidence
Two completed `pdflatex -halt-on-error` compilations; **119-page PDF**, SHA256 `d82ef4daee076372f1a687be3824572a3654ec0489e6883e01e98aa390a08fb9`. No undefined citations/references. Visual inspection of the changed HAR, video and MNIST listing pages **27, 31, 59** found legible source listings. The existing document has **34 visible missing-image placeholders** and **11 overfull hbox warnings**, still release blockers.

## Exact scientific limits
- **AST parsing is syntax checking, not standalone runtime or scientific certification.** Several printed examples depend on preceding snippets or unavailable original files.
- TensorFlow is not installed in this local environment; original HAR training and inference on complete real UCI data remain unexecuted and **HOLD**.
- The legacy artistic style transfer still uses `K.placeholder`, `K.gradients`, legacy Keras backend variables, and an undefined L-BFGS evaluator; this is not certified as an up-to-date full implementation. Original-style output images and checkpoints are not independently verified.
- Original MNIST 28×28, CIFAR-10, Stanford Dogs, real hurricane imagery, video action and style-transfer original datasets and outputs still require full specific experiments, source IDs, ground-truth evaluations and generated-result images. They are not certified from short code examples.
- Full scientific scope remains **3 of 12 locally supported named units**, inherited unchanged from the prior detailed-chapter evaluation; full chapter science and production **HOLD**.

## Source artifact
A verified local ZIP `Chapter2_Detailed_Pass_Code_Listing_Audit_20261118.zip` includes corrected full TeX/PDF, original figures, full listings ledger `qa_code_listing_full_audit_20261118.csv`, untouched source listings `qa_code_listing_original_pre_audit_111.json`, regression validator and execution JSON, two build logs, and this QA record. ZIP CRC PASS, 177 members. PDF and binary graphic assets in the ZIP are **not confirmed committed to GitHub**. The source-level check `chapters/deep_learning/part_i/qa/verify_detailed_ch2_printed_listings.py` is committed independently.

**Verdict:** printed code brevity, parsing and source structure **PASS**; modern style-transfer runtime, full datasets/model execution and overall chapter scientific/production approval **HOLD**.
