# Book-wide scientific experimental certification standard (2026-11-12)

## Governing purpose

This policy applies to every named application, case study, research benchmark and technical demonstration throughout the graduate engineering textbook. It **retains requirements R1–R10**, scales their implementation to the scientific claim, and prohibits invented data, fabricated predictions, fabricated figures, and unsupported conclusions. It supplements, without weakening, `CODE_PUBLICATION_STANDARD.md`.

A unit's **title and claims must agree with its experimental scope**. Changing an assessment level does not turn incomplete evidence into a passing experiment. No empirical unit is certified solely by a README, a publication figure, a code stub, previously published aggregate results, or unit tests.

## Scientific depth designations

| Level | Scientific role | Minimum empirical scope |
|---|---|---|
| A — Scientific demonstration | Explicitly titled, focused technique or numerical demonstration | Authentic named observational data; executed relevant algorithm; retained outputs; a measured-result figure; an appropriate reference/check or metric; faithful limited conclusion |
| B — Experimental application/case study (DEFAULT) | Complete named task, usually offline evaluation | Authentic task-matched data and executed model; retained predictions; task-appropriate evaluation, credible baseline/reference, meaningful errors, publication-quality output figures, traceable executable source and conclusions limited to offline task |
| C — Research benchmark/flagship study | Comparative, transfer, safety, deployment or broad-generalization claims | Rigorous predefined evaluation, independent test/domain where applicable, matched baselines, uncertainty/robustness, domain/operational validity and detailed failure analysis; source and artifact traceability |

Level is **not** a quality grade. A fully evidenced Level A demonstration is certified **at Level A** but does not establish a Level B/C application. Levels do not waive ethical or scientific validity; C is required where the title inherently promises cross-domain, production or system-level outcomes. Every existing application/case-study defaults to B unless a documented scope judgment raises it to C. Downgrading an application to A solely because required experiments are inaccessible is prohibited; a separately titled and genuinely limited demonstration may qualify for A.

## The ten requirements and proportional interpretations

| ID | Book-wide condition | Acceptable proportional implementation |
|---|---|---|
| R1 | Authentic data and documented provenance | Actual observed records, original real photographs, licensed measured source; distinguish model-generated outputs from direct observations |
| R2 | Suitable disjoint experimental partition and leakage controls | Frozen pretrained inference can use a documented independent evaluation sample without new training; state pretrained source overlap when known/unknown; grouped/subject/scene/temporal partitions as appropriate; no invented splits |
| R3 | Executed named relevant model/algorithm | Real executed task-specific implementation; pretrained inference is allowed if method and task match |
| R4 | Actual outputs | Save real predictions, reconstructions or action traces, with source correspondence |
| R5 | Valid reference or baseline | Mandatory **matched comparator** for relative-superiority, benchmark or comparative title claims; otherwise appropriate literature/analytic/physical reference or defensible control can satisfy, with explanation |
| R6 | Appropriate quantitative metrics and error characterization | Use repeatability intervals for stochastic/generalization claims; deterministic convergence, residual, tolerance and sensitivity for physics/numerical tasks; avoid meaningless confidence intervals |
| R7 | Domain validation consistent with claims | Offline authentic dataset evaluation is valid for an offline task; flight/vehicle/clinical validation becomes mandatory when deployment, safety or physical performance is actually claimed |
| R8 | Publication-quality real-output scientific figure | Original data/model-output plots, source-linked, visually verified in PDF; explanatory diagrams do not substitute. Minimize bar charts, prefer overlays, temporal traces, error maps, spatial views, interval/pair plots and Pareto curves when apt |
| R9 | Complete code and evidence traceability | Commit runnable implementation, configs, exact checkpoint source/sha, data access and split manifest, measured results and figure-generation code; restricted datasets/checkpoints may be linked with a verified authorized retrieval procedure and hash rather than redistributed |
| R10 | Conclusions covering only what was actually tested | Title, narrative, limits and conclusions must accurately match the whole promised task; offline testing never certifies untested deployment |

## Certification decision rules

Per requirement: `PASS`, `PARTIAL`, `FAIL`, `NOT VERIFIED`, or `N/A — JUSTIFIED`. The **N/A category is permitted only when the underlying requirement truly does not apply to the defined methodology**, never because data, GPU, licenses, time, checkpoints, or experiments are missing. In the R1–R10 table, R1/R3/R4/R8/R9/R10 remain compulsory for empirical application units; R2/R5/R6/R7 must still be evaluated and normally take PASS under an explicitly appropriate implementation rather than N/A. A legitimate N/A records a rationale and independent reviewer acceptance.

A unit is **certified at its assigned level** only if all applicable requirements are PASS, every N/A is independently justified, original measurements/figures/code were checked, and the final manuscript is consistent. PARTIAL, FAIL or NOT VERIFIED prevents full certification. Code-only execution tests do not change empirical FAIL to PASS. An otherwise fully passed status matrix is only a candidate for scientific independent review, not an automatic certificate.

A larger-scope claim forces an upgrade to its corresponding evidence level. For example, an offline UAV detector can satisfy Level B with genuine image inference and honest offline scope; an 'autonomous drone' title that claims tracking/control cannot pass until actual tracking/control results exist. An original MRI offline reconstruction can pass without clinical outcome claims; clinical benefit cannot be inferred. 'Zero-shot SAM' must report prompt/box conditions and not imply prompt-free performance.

## Required evidence ledger for each empirical unit

Document unit ID/title/level; dataset identity/license/source/hash; authentic observed sample count; pretrained training provenance and any dataset overlap; training/validation/test or frozen-inference partition and leakage controls; model/checkpoint hash, environment, seed and command; per-observation measured outputs; baseline/reference; metric definition/errors/uncertainty; claim-specific domain checks; figure generator/source table/hash and visually inspected final PDF page; code commit SHA; precise scope of conclusions and limitations. Include reviewer and review date. External source access/licensing may be documented instead of shipping protected data, but execution evidence remains mandatory.

## Engineering figure standard

Choose graphics by scientific question, not a bar-chart quota. Prefer genuine model output over ornamental schematics for every named empirical case study. Use consistent readable type and axes/units, genuine photographic or signal/point-cloud views, restrained accessible premium palette, sample counts, uncertainty when relevant, no misleading cropping, and figure-specific scientific captions. Multiple independent panels are appropriate when they each answer a question. Audit at final printed size, inspect raster and vector assets and calculate plotted-value reconciliation where possible.

## Publication governance and transition

1. Retain older R1–R10 matrices as **strict historical audit snapshots**; do not retroactively rewrite their status values to claim newly certified units.
2. Assign A/B/C with reasons and title/claim-level scope for all current units, starting with Chapter 5. Keep the default B. Maintain independent evidence and issue registers.
3. Review R2, R5, R6 and R7 for evidence appropriateness; record justification for alternative protocol. A less resource-intensive protocol may pass **only with actual correct evidence**, never via waiver.
4. Commit evidence artifacts and run reproducibility/QA checks, then rebuild/inspect manuscript figures and citations. Update certification to PASS only when independently inspected.
5. Preserve `HOLD` for every unit with actual model/dataset/benchmark/output gaps. Book-wide release is not certified until every required empirical unit at its justified level is complete.

**Non-negotiable:** genuine data, authentic model execution and output, scientifically meaningful figures, full code traceability, and defensible conclusions remain mandatory for every named empirical application or case study.
