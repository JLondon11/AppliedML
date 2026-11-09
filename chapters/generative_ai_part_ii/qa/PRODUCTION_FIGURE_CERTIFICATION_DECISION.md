# Chapter 9 — Production Figure Certification Decision (2026-11-09)

**Canonical chapter:** 9, Generative AI Part II  
**Frozen inventory:** 11 production figures  
**Final certification:** **HOLD, 0/11 certified**

## What has actually passed

The connected GitHub Actions job successfully runs the frozen-inventory decoder, the scientific-provenance unit tests, and the report-generation/upload stages. The final `--strict --production` gate fails, as intended, because no figure has a fully documented experimental record with committed output and measurements. The upstream Chapter 8 digits VAE/DDPM/DDIM workflow has separately passed on GitHub; its results must not be silently reassigned to Chapter 9.

### Mandatory production evidence, per figure

Each final figure requires (1) its precise frozen-figure number and caption; (2) a named real dataset and an authoritative dataset source URL; (3) disjoint training, validation and test counts; (4) an actual named method and complete executable Chapter 9 generator; (5) measured numerical results stored as machine-readable JSON; (6) a final committed PNG/SVG/PDF asset; (7) SHA-256 integrity verification of source, numerical results and rendered figure; and (8) independent scientific/visual manuscript review. Reusing an unrelated example, analytic curve, synthetic illustration or default shared generator is insufficient.

The new production validator checks requirements 1–7 where machine-verifiable; it **cannot** alone prove that images are scientifically faithful or that captions are integrated into the PDF. Those still require human inspection and the canonical compiled manuscript.

## Specific blockers found

- `chapters/generative_ai_part_ii/qa/figure_evidence.json` contains no certified experiment entries.
- The canonical `figures/` and `manuscript/` production directories are absent from the current repository tree.
- The chapter-level generator delegates to generic fallback renderers. These are ineligible for real-data certification.
- The existing scaling-law Figure 4 code plots published analytical-fit predictions and counterfactual allocations, not directly observed new model-training outcomes. It must be labeled as such and remains HOLD against a strict measured-real-experiment criterion.
- Chapter 8 real-digits experiments are not automatically Chapter 9 figure evidence.

## Executable gates

```bash
python -m unittest discover -s tools/scientific_qa -p 'test_chapter9_*.py' -v
python chapters/generative_ai_part_ii/qa/audit_figure_evidence.py \
  --generator chapters/generative_ai_part_ii/figure_generation/generate_figures.py \
  --evidence chapters/generative_ai_part_ii/qa/figure_evidence.json \
  --root . --report chapter9_figure_audit.json --production --strict
```

The strict command currently exits nonzero because no figure has complete evidence. A failed *release gate* is expected; a failed *unit test* would be an implementation defect.

## Closure order

First synchronize the actual Chapter 9 manuscript and frozen captions. Next implement and execute the figure-specific real-data model/dataset protocols and commit their measured outputs, figures and checksums. Then perform per-figure scientific and publication-scale visual inspection, run the strict GitHub gate, verify a passing hosted run, and compile/inspect the manuscript PDF. Do not change the certification count until all evidence exists.

**Decision:** Do not certify figures 1–11. Production figure certification is a verified HOLD, not a claim of experimental completion.
