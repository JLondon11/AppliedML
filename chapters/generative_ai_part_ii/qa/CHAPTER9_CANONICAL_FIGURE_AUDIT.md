# Canonical Chapter 9 — real-data figure certification audit

**Book chapter:** 9, Generative AI Part II (`chapters/generative_ai_part_ii`).  
**Frozen production figure count:** 11, according to `RELEASE_MANIFEST.md` and `book/CHAPTER_RELEASE_STATUS.csv`.

## Findings from the main-branch repository tree

1. The canonical Chapter 9 source uses `figure_generation/generate_figures.py`, which delegates to a shared generic rendering runtime. That runtime offers deterministic synthetic matrices, analytical surfaces, default architectures, and other illustrative fallbacks. **Such a plot does not establish measured outputs from the model or real dataset named by its frozen figure.**
2. No accepted production figures are currently committed under `chapters/generative_ai_part_ii/figures/`. The canonical `manuscript/` directory and Chapter 9 PDF are not present either. Therefore figure placement, final captions, and manuscript cross-references remain unverified.
3. Section-specific code for the scaling-law Figure 4 exists at `scaling_laws/figure_04_compute_quality_pareto_frontier/figure04_chinchilla_frontier.py`. It uses a **published parametric fit** and computes theoretical allocations and counterfactual curves; these are **not new measurements from trained models**. It is useful scholarly analysis but **does not meet a strict real-experiment-only evidence requirement** without separate empirical validation.
4. The earlier digits VAE/DDPM/DDIM work is stored under `chapters/generative_ai_part_i/`, which the current repository identifies as **Chapter 8**. The provenance cannot be transferred to canonical Chapter 9 by number alone.
5. Existing Chapter 9 evidence record starts empty, deliberately. No final figure is certified solely from the presence of plotting code.

## Enforced certification requirement

`qa/audit_figure_evidence.py` decodes the actual frozen Figure 9 inventory, detects unknown or duplicate figure records, checks each recorded figure and metrics file exists in the repository, verifies both SHA-256 digests, and rejects generic fallback generators. Its `--strict` option fails while any of the 11 figures remains on HOLD.

The new GitHub Actions workflow generates a complete JSON discrepancy report and uploads it even when the strict certification gate fails. An expected failing check means certification is **blocked**, not that empirical results were performed.

## Remediation priorities

- Recover and commit the correct Chapter 9 LaTeX manuscript and frozen captions before changing figure numbers or placement.
- For each of the 11 figures, implement the actual named model/dataset protocol, retain train/validation/test sizes, save measured CSV/JSON and the final rendered output, and record both hashes.
- For scaling-law Figure 4, distinguish published analytical-fit predictions from actual training measurements, or obtain an independently documented empirical dataset before calling it an experimentally measured plot.
- Inspect every final figure at publication size and match it to the manuscript paragraph, caption, model identity and statistical claim.
- Rerun the strict inventory and published GitHub workflow; certify only after all 11 have verified evidence and compiled manuscript integration.

**Present status: Chapter 9 HOLD.** This audit introduces reliable enforcement but does not declare all eleven figures experimentally verified.
