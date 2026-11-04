# Chapter 5 — Manuscript Master QA pass (2026-11-03)

Manuscript: **Chapter 5, Computer Vision Part II — Autonomous Systems**.

## Verified edited manuscript build
- Compiled `Chapter_5.tex` with `pdflatex` in two final passes: **96 pages**.
- **45** figure environments; **12** table environments; **11** named applications/case studies.
- Build log: **no overfull hbox warnings, unresolved citations or undefined cross-references**.
- Figure asset preflight found no missing real figure files (excluded `Figures/#1`, which is a LaTeX macro parameter).
- All PDF pages exceeded the 850-character coarse text-density screening threshold; formerly sparse page 89 now has substantive, chapter-specific research exercises.

## Source edits completed
1. Removed redundant pre-bibliography heading, avoiding a largely empty page before the bibliography.
2. Rewrote capstone setup to correct a broken sentence and refer to the actual `JLondon11/AppliedML` book repository.
3. Replaced legacy, nonexistent exercise reference directory claims with a statement of available project tooling and the need for student implementations.
4. Removed unsupported numerical prevalence and generic heavy-occlusion miss-rate assertions; distinguished CityPersons benchmark labels from CrowdHuman implementation input.
5. Expanded end-of-chapter research exercises with experimentally testable perception, tracking/planning, segmentation, occlusion and fastMRI work, specifying real-data evidence and reproducibility.

## Still blocked from full Master QA approval
The named-model inference experiments are NOT certified. Original real input files and matching model checkpoints remain unavailable for PointPillars/KITTI, YOLO UAV/VisDrone, TuSimple ONNX lane detection, CrowdHuman pedestrian detection, Meta SAM/COCO and trained fastMRI models. The neural-rendering workflow requires real posed images, matched configurations and GPU training; the production autonomous-driving stack requires genuine recorded telemetry. Classical stereo experiments, analytical reconstructions, tests on generated fixtures, and published-data replots are supporting evidence, not substitutes.

No fabricated inference results, tables, or figures have been inserted. **Chapter 5 remains Master QA NOT APPROVED** for these applications.

A complete source/PDF/figure package was generated locally for the revised manuscript. The GitHub directory includes code manifests and case-study scripts; this report does not claim the local manuscript was committed to GitHub.
