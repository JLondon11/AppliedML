# Book Reproducibility Policy

1. All code-generated figures must have committed generating code.
2. Randomized experiments must set and document seeds/random states.
3. External data/models must document source, version, acquisition, preprocessing/split protocol, and license/access restrictions.
4. Source-derived published evidence must retain exact provenance and must not be represented as a newly generated experiment.
5. Code must not depend on OpenAI sandbox-specific paths.
6. Student chapters and instructor solutions are versioned separately.
7. Every chapter must end with Summary and End-of-Chapter Problems.
8. Every named Case Study/Application must document objective/context, data/inputs, method/implementation, evaluation, interpretation/limitations, and reproducibility/deployment.
9. Each case study/application requires an executable, versioned model-specific implementation in GitHub; every model-output figure and numerical results table must be traceable to executed code, documented real dataset samples, model weights/checkpoint, configuration, evaluation protocol and results. Complete implementation lives in GitHub, not printed as a code dump.
10. For every case study, use substantive motivation, then Problem, Objective, Dataset, Model, Metrics, Results, Benchmark Comparison, Analysis, and Conclusions. All named models and datasets require appropriate first-mention references and links.
11. Classify evidence honestly: (A) named-model real-dataset inference; (B) controlled or simulated code-executed algorithm; (C) code replots of published measurements; (D) original conceptual figures. Only (A) satisfies the named-model inference gate. No fabricated experiments or cross-protocol ranking.
12. A case study cannot be marked approved unless the verified implementation, model/dataset provenance, actual generated output figure, quantitative results table, and source-level scientific analysis all exist. Keep statuses BLOCKED/PARTIAL for missing data, weights, licensing, evaluation or verification.
13. Store application-specific code under chapter/section/case-study directories, with README, environment lock/requirements, dataset and checkpoint manifests, runnable scripts, generated metrics, provenance, and a continuous-verification manifest.
