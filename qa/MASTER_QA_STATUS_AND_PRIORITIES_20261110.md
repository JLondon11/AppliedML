# Master QA status — 2026-11-10

**Decision: GLOBAL HOLD — no chapter or book-wide scientific publication certification is supported by current evidence.**

This report consolidates the authentic manuscript archive audit and the live GitHub source tree. It does **not** conflate code presence, a generated schematic, or a figure caption with scientifically validated measured outputs.

## Chapter 11: verified source and exact manuscript audit

Local release: `Scientific_AI_Chapter11_RealData_Integrated_20261110_FINAL.zip`, containing `11_Scientific_AI/Scientific_AI_InProgress.tex` and the 158-page PDF. The local PDF SHA-256 was independently recomputed as `33678a626e56ba1436f10d16e1b73fc1e87f11a549a7e7062d6b43ce1f0b0431`.

| Actual-source indicator | Verified count |
|---|---:|
| Figure environments | 54 |
| Tables | 46 |
| Code-listing environments | 18 |
| Named application and case-study headings | **15** |
| Named units with at least one figure inside the section | **15** |
| Named units with a caption explicitly describing actual measured results | **1** |
| Named units with schematic/workflow figures but no explicit results figure | **14** |
| Missing literal figure assets in the supplied archive | **0** |

The **one named unit with a measured results graphic** is the Wisconsin Diagnostic Breast Cancer classical machine-learning comparison (Figure 11.8). It contains real 455 development / 114 held-out observations and measured classifier results; separate sources document its executions. The fully revised Figure 11.3 on real annual sunspots and Figure 11.11 on eight observed aluminum-rod thermocouple histories are additional **real-data figures**, but neither closes the absent result-figure requirement for the fourteen other explicitly titled application/case-study units.

**Important method limitation:** The 15 figure coverage decisions classify LaTeX sections and their figure captions, not the scientific content of every figure pixel, table, model run or external paper. A diagram can still be a useful explanatory figure; it simply cannot replace an actual application results figure. The sepsis overview and its nested translational case study share a pipeline graphic and are separate named headings, not two independently proven experiments.

The exact 15-unit registry with figure basenames, evidence status and outstanding action is committed at `chapters/scientific_ai/qa/CH11_APPLICATION_FIGURE_EVIDENCE_MATRIX_20261110.json`.

## Chapter 11 experiment priorities (do not claim completion before execution)

**Critical clinical/image units:** authentic fundus Grad-CAM predictions with original image and saliency overlay; pneumonia CT volume classification using genuine volumes and held-out outputs; sepsis time-series prediction on real patient records with documented temporal/center splits and measured ROC, calibration or alert burden; HPA organelle classification using real microscopy and predicted labels; nuclei segmentation against genuine annotated images, including masks and error analysis; focal cortical dysplasia detection on real subject scans with suitable masks or subject-level evidence.

**Research and discovery units:** actual antibiotic-activity measurements/chemical comparisons, safely interpreted against source publications; properly matched multimodal precision oncology cohorts and measured predictions (do not combine unpaired patients to fabricate a multimodal experiment); actual experiment-planning decisions from recorded experimental outcomes; measured autonomous-laboratory and materials-discovery results if suitable authentic sources exist; data-assimilation error metrics from actual observed biomedical/industrial twin measurements.

**Remaining topical diagrams** may be retained and improved as conceptual figures, but must be accompanied by actual experimental result figures or the unit stays HOLD.

## Other book chapters: precise supported evidence

- **Chapter 2:** A prior curated registry tracked 13 units, of which 9 were blocked and 4 had supporting execution artifacts; 0 were fully certified. Real Digits and sunspots experiments must not be relabeled as original MNIST, CIFAR-10, HAR or other named-dataset studies.
- **Chapter 6:** The supplied `Proceed Chapter 6.txt` confirms **Figure 6.20 placement QA accepted** in a *reported* Pass 37, printed page 67, before problems starting page 101, in a 109-page PDF. The referenced Pass 37 PDF/TEX/ZIP files are **not present** in this current local workspace, so their bytes and figure pixels could not be independently inspected here. Chapter 6 overall scientific QA remains HOLD.
- **Chapter 11:** Real-data replacements Figures 11.3, 11.8 and 11.11 are integrated in the inspected archived PDF. 14 named applications/case-studies still require actual model-result figures; full chapter certification HOLD.
- **Other chapters:** Recent GitHub inventory and partial execution reports document substantial code, but no complete per-manuscript-unit, dataset-to-figure certification was established in this pass. Their global status remains NOT CERTIFIED, not automatically failed or approved.

## GitHub publication gap

The live GitHub tree contains many complete source scripts, but the chapter-family canonical release folders required by `RELEASE_MANIFEST.md` are not collectively complete. In particular, no chapter-family directory scanned has a populated `manuscript/` directory; none has an `instructor_solutions/` directory; and Chapter 11's final revised 158-page PDF and modified LaTeX source are currently supplied as local archive deliverables, not verified on GitHub. Executable code checked into GitHub is necessary but insufficient for complete release approval.

## Scientific figure acceptance gate

Every application and case study requires (1) authentic primary dataset/photographs or observed records, (2) correctly executed model and named split/counts, (3) saved predictions and measured metrics, (4) scientifically appropriate publication-quality result graphics, (5) matching manuscript figure/caption/discussion, (6) complete committed runnable code, (7) figure/data source lineage and asset hashes, and (8) resolved PDF references and visual layout. Genuine outputs from a trained generative model are permitted if explicitly distinguished from original data.

**Next review priority:** For Chapter 11, the 14 named units with no observed-output figure are the dominant scientific deficit. Update the actual manuscript only after real experiment outputs exist, then rerender/verify the PDF. Also retrieve the true current Chapter 6 Pass 37 package before claiming independent verification of that release.

**Overall Master QA: HOLD. Figure 6.20 placement (Chapter 6): ACCEPT based on provided continuation report, not re-executed. Chapter 11 packaging and Figure 11.3/8/11 replacements: PARTIAL ACCEPT. Universal case-study real-data figure requirement: NOT MET.**
