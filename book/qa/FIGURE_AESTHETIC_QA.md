# Book-Wide Figure Aesthetic QA

A figure cannot receive final ACCEPT on scientific validity alone. The final rendered asset must also pass visual-design QA at publication scale.

## Required checks

For every final figure verify:

1. **Composition** — compact, balanced, and free of accidental empty space.
2. **Typography** — axes, ticks, annotations, legends, and panel labels remain legible after LaTeX scaling.
3. **Palette** — restrained, premium, scientifically meaningful, and not simply the plotting-library default.
4. **Hierarchy** — the intended scientific comparison is visually dominant.
5. **Panels** — multi-panel artwork has consistent sizing, alignment, spacing, and visual weight.
6. **Legends/colorbars** — quantitative scales are readable and never overlap data or neighboring panels.
7. **Density** — no spaghetti-plot clutter, label collisions, or excessive annotation.
8. **Consistency** — the figure fits the book's visual language without making sequential figures look identical.
9. **Print robustness** — sufficient contrast on white paper and reasonable grayscale/color-vision robustness.
10. **Scientific restraint** — no decorative effects, icons, gradients, or infographic grammar unless scientifically necessary.

## Status rules

- **ACCEPT** — scientifically correct and visually publication-ready.
- **REVISE** — scientifically correct but visually weak, cluttered, poorly spaced, inconsistent, or insufficiently legible.
- **REPLACE** — the visual form itself is scientifically or aesthetically inferior to a materially better representation.
- **BLOCKED** — final artwork or underlying data/computation is unavailable for inspection.

Final acceptance requires direct inspection of the rendered asset, not merely its source code.
