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

