# Chapter 2 — Pass 7 scholarly bibliography and page-production QA (2026-11-15)

## Basis and verified edits

Reopened the actual user-supplied `Chapter2_Pass6_Scientific_Certification_Remediation_20261115.zip`; extracted the complete LaTeX, all figures, numerical evidence and existing PDF. Made **two externally checked** bibliography corrections directly in the manuscript source:

1. `sagawa2020`: full original author roster and ICLR title **Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization**; linked ICLR OpenReview `https://openreview.net/forum?id=ryxGuJrFvS`.
2. `zhang2019rmsnorm`: corrected from preprint-only to actual **NeurIPS 2019** published proceedings and link `https://proceedings.neurips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html`.

The exact citation keys are preserved. No model predictions, data, or measured figures were altered in this pass.

## Actual manuscript and publication checks

| Check | Finding |
|---|---|
| Bibliography entries | **50** |
| Citation commands | **114** |
| Missing cited bibliography keys | **0** |
| Uncited bibliographic items | **4**: `devlin2019`, `dosovitskiy2021`, `han2015`, `kirillov2023` — editorial cleanup candidates |
| Figure environments | **35**, all linked original PDF graphics present |
| Tables; code listings | **5**; **23** |
| LaTeX labels | **112**, no duplicate labels |
| Source->output evidence gate | **PASS** on original observed source arrays and saved experimental predictions |
| Full LaTeX build | **Two successful pdflatex passes**, **118-page PDF** |
| Final LaTeX warnings | **No** undefined citations/references, overfull boxes, duplicate page-anchor warnings |
| Entire PDF sparse-page scan | **0 unexpected sparse body pages**, excluding intended title |
| Visual check | Contact-sheet of selected 14 pages, including figure/case-study and reference pages; not all pages fully human-inspected |

The numeric gate re-executed from the Pass 6 archive reconciles authentic Digits/Diabetes data and models' **108 digit 3/8 predictions, 3 actual 8×8 convolution responses, 5,400 twelve-seed predictions, 450 ten-class digit predictions, 111 diabetes regression predictions**, and seven result figures with 21 PNG/PDF/SVG assets. These checks verify saved-output integrity, **not fresh model retraining**.

The observed-data Level A/B cases retain their prior scientifically scoped **5/5** certifications (2A, 3B). The complete chapter figure balance is 7 output-figure groups and 28 explanatory or analytic graphics, with no claim that every conceptual figure reflects measured experiments.

## Deliverable and holds

Rebuilt final PDF SHA256 `0ca3813b8b48caf39fc9a8b8a6dcbc36417a584f55656e91c29b446c8162d1a5` (118 pages). Complete source, PDF, figures and evidence in session ZIP `Chapter2_Pass7_Scholarly_Production_QA_20261115.zip` (155 ZIP members, CRC integrity PASS, ~22 MB). Full QA report included in ZIP. Two corrected bibliography references independently corroborated by original publication records.

**Not yet certified:** a publisher-metadata check of each of the 50 reference entries; complete manual 118-page print proof; and a confirmed canonical GitHub push of the latest full LaTeX/compiled PDF/binary figure assets. The source/graphic assets are distributed in ZIP **not** claimed present in GitHub.

**Chapter 2 scoped scientific certification: 5/5 RETAINED. Chapter production: PARTIAL ACCEPT / HOLD. Entire book Master QA: HOLD.**
