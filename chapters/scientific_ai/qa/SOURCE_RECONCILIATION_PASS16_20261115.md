# Chapter 11 — Source Reconciliation Pass 16 (2026-11-15)

## Verified outcome

A new local recovery package reconstructs a coherent **68-figure asset library**, corresponding to the directly inspected 177-page chapter PDF. The figure identities derive from Pass 15's PDF image-content matching, not from caption similarity alone.

- Current figure identifiers: `11.1` through `11.68` (continuous; 68 of 68).
- 52 exact visually decoded historical asset matches from `All_Chapters_Complete_10.zip`, confirmed previously by pHash Hamming 0 and normalized correlation 1.0.
- 16 unmatched current-edition image objects recovered from PDF (not assumed to be independently generated experimental data).
- Each recovered/current asset is assigned `FiguresCurrent/Figure_11_NN.png`, has the original matching PDF page, a source class and a SHA-256 checksum.
- All 68 output file hashes pass the local `verify_current_assets.py` check.
- A 34-page two-figure-per-page visual proof PDF was generated and page 1 rendered for inspection.
- The recoverable historical editable LaTeX source (54 figure environments) was preserved in a separate `historical_source` directory to prevent false current-edition attribution.
- Local deliverable: `Chapter11_Source_Reconciliation_Pass16.zip`, containing all 68 assets, provenance ledger, verification and asset construction scripts, historical TeX, LaTeX figure path map, and visual proof.

## Hard limitations

**The current 177-page 68-figure editable manuscript remains unavailable.** The package is an asset reconstruction and **not a recovered exact 68-figure LaTeX manuscript**. Its image binaries have not been published into the GitHub canonical `chapters/scientific_ai/figures/` release; this QA Markdown record does not assert otherwise. A historical image content match does not certify experimental provenance, source code, or validity.

Figure 11.54 in the current PDF is the domain-scale platform diagram. Figure 11.68 is a multiscale sensor assimilation architecture. Neither equals the measured heated-rod results figure from the separate experiment build. Do not merge by number only.

## Separate approvals

- Scientific evidence for the broader named Scientific Digital Twins application: **HOLD**.
- Annotated nuclear segmentation plus nucleus-level classification: **HOLD**.
- Chapter 11 production release: **HOLD**, until exact current source or controlled substantive reconstruction, numbered caption/source integration, full PDF rebuild, citation/figure/table layout QA, and documented Github publication.
- Complete chapter Master QA: **HOLD**.

The asset ledger and verified outputs are available as a user-delivered local ZIP. This registry entry does not claim that ZIP or binary assets are already in GitHub.
