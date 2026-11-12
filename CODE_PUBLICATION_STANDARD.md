# Mandatory source-code publication standard — all chapters

Every application, case study, numerical experiment, simulation, benchmark, and code-generated scientific figure in the textbook must have its **complete executable source code committed to this repository**. A notebook screenshot, LaTeX code excerpt, zip attachment, or locally executed script does **not** satisfy the GitHub requirement.

## Required structure

Use a canonical path under `chapters/<chapter_slug>/` or the existing established chapter layout. Each scientific unit must have:

1. **Complete executable source** organized by named chapter section and experiment; no placeholders standing in for the actual implementation.
2. **README** with exact installation and execution commands, dataset sources and licenses, random seeds, training/validation/testing sizes, split criteria, runtime environment and limitations.
3. **Figure-generation code** that reproduces every empirical figure shown in the manuscript from real datasets and actual outputs. Conceptual architecture illustrations may be non-data renderings but must be labeled accordingly.
4. **Machine-readable numerical results**, including measured metrics, model predictions or their grounded provenance, training histories where pertinent, figure manifests, and verification checks.
5. **Committed figures** in source-friendly SVG plus PNG/PDF when reasonable, or where binary-size/licensing constraints apply, explicit verified artifact hashes and an accessible retrieval mechanism. Never claim a binary is on GitHub unless it is present.
6. **Section linkage** connecting each manuscript figure/table/listing to the code entrypoint, input dataset, output filename, and research references.
7. **Validation evidence**, including executed checks and a clear status (executed, independently reproduced, pending, blocked).

## Acceptance criteria

A unit is **not code-publication certified** if the GitHub executable implementation is absent, depends on undisclosed local inputs, lacks the figure-generation script for its claimed empirical figures, or cannot be tied unambiguously to the manuscript. Do not substitute different datasets for an experiment by relabeling outputs. Incomplete units must remain marked **pending**, not certified. Ensure one chapter's production state cannot silently revert to older code-heavy source versions.

## Current scope

This policy covers **all chapters**, including every application and case study. The current repository contains substantial code, but this is not proof of unit-level completeness. See `CODE_PUBLICATION_AUDIT_20261109.md` and the Chapter 2 strict evidence matrix for known gaps.

Future PR/release checklist: run the code, regenerate scientific outputs, confirm dataset/partitions/citations, verify all files in the actual GitHub tree, and record the commit SHA. Keep manuscript code listings few and substantive; complete implementations belong in GitHub.

## Non-negotiable real-data / no-fabricated-results requirement (2026-11-09)

**All scientific measurements, result figures, benchmark tables, predictions, training curves, confusion matrices, ablation analyses, and numerical comparisons in the published book must be produced by executed code from authentic, sourced real-world datasets or genuine published experimental observations.** Never invent, simulate, sample an arbitrary probability distribution, or hand-draw numerical values for an empirical result. A plotting script or visually polished image alone cannot prove scientific provenance.

- **Admissible:** authentic real photographs (including steel-defect, texture, medical and satellite photographs), original collected images, video, text, observations, public research datasets, their legitimate preprocessing/augmentation used for real-data model training, and actual trained-model predictions/evaluations on authentic held-out observations. Model-generated predictions are permitted when they are *actual outputs from a model applied to real data*, not invented or manually assembled; identify them precisely.
- **Not admissible as empirical data:** fabricated labels/observations, random or hand-specified metric arrays, toy synthetic targets, illustrative confusion matrices/ROC curves/training histories, arbitrary numbers described as measurements, and arbitrary simulated observations falsely presented as measurements. Trained-model-generated images may be legitimate experimental outputs or explicitly labeled inputs in a generative-model evaluation; they must not be described as photographs directly captured in the real world.
- **Explanatory architectures and mathematical derivations:** allowed as illustrations of concepts, but they must not be presented as observations, datasets, training outcomes, or benchmark results. Equations derived algebraically are not empirical data. A numerical mathematical demonstration must be clearly identified as a mathematical example, never as measured real-data evidence; when uncertain, omit the illustration rather than implying an experiment.
- **Published results quoted from research:** may be accurately cited and labeled as *published external findings* only if the underlying study used authentic real-world data and the values are checked against the source. They must not be described as locally generated experiment outputs.
- **Generative-model chapters:** model-generated outputs can be discussed as the subject of analysis, but must never be mislabeled as genuine observations or used as substitute training/evaluation ground truth in a real-data-certified experiment. Authentic model-generated outputs may appear as empirical evidence of a model's behavior when actually generated by executed code trained on authentic data, identified as generated output, and evaluated with measured metrics on real held-out data. Controlled synthetic-versus-real comparisons are acceptable when the research question explicitly concerns generated outputs, with distinct provenance for each experimental arm.
- **Mandatory chain of evidence:** every retained empirical figure/table requires dataset name, primary citation or download provenance, real observation count, chronological/group/stratified split sizes, actual execution command, source path committed to GitHub, recorded metric or prediction file, generator code path, generated output path and commit/retrieval hash, plus verification that chart values match measured output records.
- **Exclusion rule:** if dataset origin, real-data status, run provenance, output matches, or GitHub source publication cannot be verified, remove the empirical claim/figure/table from the publication draft and mark the unit **HOLD**. Do not insert placeholder graphics, guess numbers, or count that unit as certified.

This rule overrides older documentation that accepted simulated data, synthetic evaluation sets, representative diagnostics, or illustrative numerical plots as scientific results.

## Proportional scientific experimental certification (2026-11-12)

The book-wide **R1–R10 scientific certification** policy is defined in [SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md](SCIENTIFIC_EXPERIMENT_CERTIFICATION_STANDARD.md). It uses Level **B** by default for named applications and case studies, Level **C** for designated research/operational benchmark claims, and Level **A** for explicitly limited scientific demonstrations. Levels determine the **appropriate experimental protocol**, not whether authentic execution, genuine real data, actual results figures, or complete code/evidence are required. `N/A — JUSTIFIED` is allowed only for scientifically inapplicable criteria, with independent justification—not for inaccessible inputs or unfinished code. The present source-code publication and non-fabrication obligations remain fully binding.
