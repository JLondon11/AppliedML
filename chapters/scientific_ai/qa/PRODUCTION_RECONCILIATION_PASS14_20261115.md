# Chapter 11 — Production reconciliation, Pass 14 (2026-11-15)

The current independently inspected **177-page PDF** is NOT paired with the recoverable historical LaTeX manuscript in the source archive. Do not silently overwrite it.

## Frozen inventory findings

- Current PDF: **68** consecutively numbered figures (11.1–11.68) and **zero** visible literal `??` strings in extracted text.
- Historical `All_Chapters_Complete_10.zip` LaTeX source: **54** figure environments and original figure assets.
- Caption-based source-to-current mapping: **52 high-confidence identity matches**; the old 11.3 and 11.4 have substantially replaced captions, so their assets require separate matching. **14 additional current-edition figures** are not represented in the historical list.
- Figures 11.3 and 11.4 are listed as revised identity pending asset-level verification, **not certified by unchanged names**.

## Confirmed collision

- Current PDF **Figure 11.54**: a **Domain-Scale Scientific AI Platform** architecture diagram on PDF page **139**.
- Current PDF **Figure 11.68**: a **Multi-scale sensor and observation assimilation architecture** on PDF page **165**.
- The GitHub `code/figure_11_54_real_rod_digital_twin` measured heated-rod result figure was assigned Figure 11.54 in **another manuscript build**. None of these current PDF captions corresponds to that measured laboratory figure. **Current-edition integration is unverified.**
- Experimental evidence for the heated-rod lab remains valid within its recorded limited scope. Do not broaden its claims to climate, clinical, or industrial deployment.

## Separate scientific/production gates

- Scientific Digital Twins broader application: SCIENTIFIC_HOLD.
- Nuclear segmentation plus nucleus-level cancer classification: SCIENTIFIC_HOLD. Authentic image-derived candidate masks and separate patient WDBC predictions are unpaired; no benchmark annotated nuclei masks or paired cell labels.
- Current chapter: PRODUCTION_HOLD until the exact 177-page PDF matching editable LaTeX source, figure binaries and per-number captions are reconciled and a clean PDF build verified.
- Preserve prior historical audit snapshots. This record **does not** assert a new final manuscript build, newly executed sepsis/CT/materials studies, or a complete GitHub release.

## Files in local Pass 14 evidence ZIP

`current_figure_inventory.csv`: exact extracted caption inventory for 11.1–11.68;
`historical_to_current_figure_crosswalk.csv`: 54 old LaTeX identities and tentative mappings, flagged when unverifiable;
`unmatched_current_figures.csv`: 14 new current figure identities;
`figure_54_pdfpage_139.png` and `figure_68_pdfpage_165.png`: direct PDF page render verification.

Dataset/model/citation certifications must follow `SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md` and `CODE_PUBLICATION_STANDARD.md`.
