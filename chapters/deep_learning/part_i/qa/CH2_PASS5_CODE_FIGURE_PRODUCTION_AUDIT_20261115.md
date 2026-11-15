# Chapter 2 — Pass 5 complete-source figure, table and code audit
Date: 2026-11-15. Source: actual user-supplied `Chapter2_Pass4_All5_RealData_20261114.zip` (`Chapter2_Pass3/Chapter_2_Deep_Learning_Part_I_REVISED_QA_PASS2.tex`).

## Inspected and actually built
- 35 figure environments and **35/35 corresponding readable single-page PDF assets**; seven genuine observational/model-result figures (2.5, 2.19, 2.31–2.35), 28 conceptual/analytic or architectural scientific figures. Do not falsely call explanatory plots empirical results.
- Five tables checked; representative CNN output counts and 24,704/74,112/98,816 RNN/GRU/LSTM parameter counts independently validated against formulas with bias conventions.
- 23 Python listing environments; all 23 parse successfully using `ast.parse`. Six amended routines passed additional actual functional checks.
- 112 labels; no duplicate names.
- **Nine repaired listings:** gradient-norm clipping `lst:clip`; SGD momentum `lst:momentum`; mixup `lst:mixup`; training-only pipeline `lst:pipeline`; 2D convolution `lst:conv`; ensemble uncertainty `lst:ensemble`; actual compute-budget metadata `lst:budget`; manifest validation `lst:manifest`; protocol-matched classifier comparison `lst:baseline`. All now use declared function inputs and avoid unbound top-level globals; explanatory snippets are not falsely passed off as full standalone applications.
- Replaced unbreakable inline path `\texttt{code/real\_digits\_nonlinear\_2d.py}` with breakable `\path{code/real_digits_nonlinear_2d.py}`; removed 123.7 pt overfull line.
- Disabled hyperref page anchors only in title/TOC frontmatter and restored them at `\mainmatter`; removed duplicated `page.i` PDF destination.
- **Two full successful LaTeX passes:** final PDF 118 pages, SHA-256 `ced984f6f1a1b17fc6817b50a1379f15b27b27558d711c479445268e5f2b4e69`. Last build had no overfull hboxes, no duplicate-anchor warnings, no undefined citations/references.
- Re-rendered and inspected PDF pages 52 and 98, demonstrating Figure 2.19/Table 2.1 and Figure 2.35/seed study with their relevant prose/listing.
- Complete updated source, binary figure PDFs, numerical experiment archives and the new machine-readable per-figure report delivered in **`Chapter2_Pass5_Production_Code_Figure_Audit_20261115.zip`**. ZIP CRC/integrity check passed.

## Honest certification decisions

Five original-data empirical units retain existing scoped 2A/3B scientific certifications; **this production audit creates no new certificates**. All 35 graphics exist and are correctly linked, but a full literature-level fact check of every paragraph, print-size scrutiny of each of the other 33 PDF pages containing figures, and bibliography review are not completed by this pass.

**Chapter 2 PRODUCTION: PARTIAL ACCEPT / HOLD** until complete proofing and final canonical GitHub manuscript/binary archive with confirmed CI. This GitHub report is committed, but the revised full LaTeX and binary PDF and figure sources remain in the delivered ZIP; their GitHub release publication is not verified. Whole-book Master QA HOLD.
