# Figure Caption and Panel-Reference QA

## Hard rule
Every figure must have a detailed caption. Every labeled panel in a multi-panel figure must be explicitly referenced and described in the main caption.

## Mandatory checks
1. Identify every panel label visible in the figure: (a), (b), (c), and so on.
2. Verify that each label appears in the main caption.
3. Verify that each panel receives a substantive description, not merely a label mention.
4. Verify that panel descriptions follow the visual order of the panels.
5. Verify that differences in datasets, methods, conditions, metrics, axes, scales, or visual encodings across panels are explained.
6. Verify that non-obvious colors, line styles, markers, uncertainty bands, error bars, symbols, and annotations are defined where necessary.
7. Verify that empirical captions include sufficient provenance or experimental context to interpret the result.
8. Reject captions that are generic, one-line summaries when the figure requires more detail.
9. Reject any figure in which a visible panel has no corresponding caption description.
10. Final acceptance must use the compiled PDF/figure integration, not source text alone.

## Release criterion
- Undescribed labeled panels: 0
- Panel-label/caption mismatches: 0
- Materially under-detailed captions: 0
- Unresolved caption defects: 0


## Heatmap color-scale checks
11. Every heatmap or scalar color field must include a visible quantitative colorbar or legend.
12. The legend/colorbar must identify the represented quantity and show interpretable numerical values or range.
13. Units must be included where applicable.
14. Reject heatmaps whose colors cannot be quantitatively interpreted from the figure itself.

## Duplicate-figure checks
15. Compare all figure assets within each chapter for exact duplicates.
16. Review visually similar figures for near-duplication or redundant information.
17. Reject duplicates that differ only by filename, export format, crop, resolution, or minor styling.
18. Retain similar figures only when they have a clear distinct pedagogical purpose.

## Additional release criteria
- Heatmaps without quantitative color legend/colorbar: 0
- Exact duplicate figures within a chapter: 0
- Unresolved near-duplicate figures within a chapter: 0
