# Chapter 6 — Proportional scientific certification (R1–R10)

## Scope and authoritative evidence
This policy applies to every empirical case study, application, and designated technical demonstration in Chapter 6 (Generative AI: Part I). It does not itself certify any scientific result or establish that a manuscript revision has been built. Certification requires inspected source artifacts, actual execution evidence, and a successful GitHub commit for newly generated code.

## Levels
- **A — Scientific demonstration:** a focused real-data, executed algorithm with actual outputs, appropriate quantitative checks, accurate scope, and a publication-quality real-results figure.
- **B — Experimental application/case study (default):** application-specific real-data execution, quantitative evaluation, appropriate reference/baseline, error analysis, and traceable figures.
- **C — Research-grade benchmark:** independently evaluated and sufficiently broad comparative experiments, uncertainty/robustness, and domain checks needed to support general performance claims.

Levels represent breadth of demonstrated evidence, not honesty or visual quality. A certified A unit cannot make C-level claims.

## Requirements and applicability
| ID | Requirement | Application |
|---|---|---|
| R1 | Authentic dataset and provenance | Mandatory for empirical units; record dataset identity, source, access/version/license where appropriate, and observed samples. |
| R2 | Split design and leakage control | Method-dependent; document train/validation/test counts when training is used, or justify a fixed independent evaluation protocol when not. |
| R3 | Executed named model/algorithm | Mandatory; diagrams or methodology descriptions alone do not satisfy it. |
| R4 | Actual predictions, generations, outputs or action traces | Mandatory, retained with execution context. |
| R5 | Credible baseline/reference | Mandatory for comparative or improvement claims; otherwise document the technically meaningful reference or justify N/A. |
| R6 | Metrics, uncertainty and errors | Mandatory in method-appropriate form; deterministic convergence/residual/sensitivity checks may replace repeated stochastic runs. |
| R7 | Domain-specific validation | Matched to the claims; offline benchmark evidence does not certify physical deployment or clinical/industrial outcomes. |
| R8 | Publication-quality figure from actual experimental results | Mandatory; verify exact numerical provenance, captions, labels and figure-code linkage; prefer varied, informative graphic forms, avoid gratuitous bar charts. |
| R9 | Complete code, configurations, traceability | Mandatory; full code and plotting scripts must be committed in this repository; record environment, model/checkpoint and seeds when applicable. External restricted artifacts require documented acquisition and immutable identifiers rather than unlicensed redistribution. |
| R10 | Conclusions matched to verified scope | Mandatory; no extrapolation from demonstrations to deployment/general benchmarking without evidence. |

Record each R1–R10 as **PASS**, **PARTIAL**, **FAIL** or **N/A — Justified** with inspectable evidence. N/A is not allowed on grounds of difficulty, cost or inconvenience. For empirical units the fundamental R1/R3/R4/R8/R9/R10 require PASS; R6 must be substantively satisfied in its applicable form. An applicable PARTIAL or FAIL blocks full certification at the selected level.

## Chapter 6 special restrictions
- Industrial-defect generation: genuine defect data, generative model outputs, measurable fidelity/diversity and suitable downstream anomaly/defect-evaluation evidence if improvement is claimed. Figure placement QA does not imply scientific certification.
- Synthetic driving: genuine multi-scene driving data and a held-out downstream perception experiment are necessary to claim perception benefit. Offline evaluation must not be characterized as safe autonomous deployment.
- Concrete mixture generation: model suggestions and measured physical material properties are different quantities. Physical-performance claims require independent specimen testing; without them, limit conclusions to computational design or measured training-data prediction.
- Generative image/model demonstrations: a real training/inference experiment may be A if claims and title are narrowly scoped. Do not imply benchmark-wide generative model superiority without adequate baselines and validation.

## Unit evidence record
For each named study create a row with unit/section identifier, title, target A/B/C, actual data source, counts, model and run command, metrics, uncertainty/errors, reference, domain checks, output files, exact figure number(s), figure-generation command, related GitHub paths, commit SHA, claim boundary, and a verdict for each R1–R10. Do not invent missing section identifiers or values.

## Initial review boundary
Repository README identifies `chapters/generative_ai_part_i/figure_generation/pass12/generate_pass12_figures.py` as a generator for Figures 03, 10, 11, 14, 20 and 24 based on scikit-learn digit experiments. This is evidence of *a committed generator*, not by itself proof that later print-figure numbering, current manuscript images, exact experiment outputs, or all other case studies have been independently certified. Require source–asset–manuscript checks and actual run logs before marking PASS.

## Release gate
A release is scientifically certified only when every chapter unit has an explicitly assigned level, complete and supported requirement verdicts, the figure/source/caption mapping has been verified against the current manuscript, all requisite code is on GitHub with known commit SHAs, and every applicable requirement has PASS. Keep production-placement QA separate from scientific-evidence QA.
