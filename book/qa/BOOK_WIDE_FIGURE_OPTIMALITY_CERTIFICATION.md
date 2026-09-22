# Book-Wide Scientific Figure Optimality Certification Matrix

## Hard requirement

Every figure in every chapter must be individually reviewed to determine whether a materially better scientific figure can be used. Final chapter acceptance requires every figure to reach **ACCEPT** after remediation and inspection of the final rendered output.

## Current repository certification state

The repository presently exposes only a subset of final rendered figure assets. Therefore chapters without their complete final figure sets cannot be certified yet.

| Chapter | Current certification state | Required action |
|---|---|---|
| Foundations of Machine Learning | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Deep Learning Part I | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Deep Learning Part II | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Computer Vision Part I | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Computer Vision Part II | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Natural Language Processing Part I | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Natural Language Processing Part II | PARTIALLY AUDITED | Figures 1--12 have been reviewed at source/final-asset level where available; remediated figures require regenerated-render inspection before final ACCEPT. Audit any additional chapter figures not present in the repository. |
| Generative AI Part I | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Generative AI Part II | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Reinforcement Learning | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |
| Scientific AI | PARTIALLY AUDITED | Figure 14 has been reviewed/remediated; all remaining chapter figures require individual scientific-optimality review. |
| Systems Engineering and MLOps | NOT YET CERTIFIED | Synchronize all final figure assets or compiled chapter PDF; audit every figure individually. |

## Per-figure audit record

For every figure, record:
1. chapter and figure number;
2. section/subsection;
3. title and frozen-inventory requirement;
4. figure classification: theory, algorithm, application, case study, benchmark, diagnostic, architecture, or other;
5. provenance class: conceptual computation, real public data, reproduced experiment, simulator output, published fit, etc.;
6. current visual form;
7. alternative scientific representations considered;
8. whether uncertainty/distribution should be shown;
9. whether the current form uses excessive infographic/flowchart grammar;
10. heatmap/colorbar compliance where applicable;
11. panel/caption compliance;
12. duplicate/near-duplicate check;
13. decision: ACCEPT / REVISE / REPLACE / NOT YET CERTIFIED;
14. required remediation;
15. final rendered-output reinspection result.

## Final release gate

A chapter fails figure QA if even one figure remains REVISE, REPLACE, or NOT YET CERTIFIED.

The final book passes only when every chapter has:
- all figures individually audited;
- all remediations completed;
- all regenerated assets visually inspected;
- all captions and panel descriptions aligned;
- all heatmaps quantitatively legended;
- no duplicate or redundant figures;
- every figure at final status **ACCEPT**.
