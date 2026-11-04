# Chapter 1 Master-QA closure audit — real-data case studies/applications
Date: 2026-11-04

## Scope
This audit enforces the project requirement that every Chapter 1 case study/application contain output/results figures generated from real data by committed executable code.

## Verified repository state
| Requirement | Status | Evidence |
|---|---|---|
| Complete empirical code organized by chapter/section | PASS | `code/case_studies/` |
| Medical-diagnosis real-data implementation | PASS | `medical_diagnosis_wdbc.py` |
| Medical measured metrics committed | PASS | `results/medical_diagnosis_wdbc/metrics.csv` |
| Edge-quantization real-data implementation | PASS | `edge_digits_quantization.py` |
| Edge measured metrics committed | PASS | `results/edge_digits_quantization/metrics.csv` |
| Weather real-data implementation | PASS-CODE | `weather_seattle.py` |
| Weather measured artifacts committed | HOLD | external dataset could not be materialized in current execution runtime |
| C-MAPSS real-data implementation | PASS-CODE | `aerospace_cmapss_fd001.py` |
| C-MAPSS measured artifacts committed | HOLD | external dataset could not be materialized in current execution runtime |
| Canonical Chapter 1 manuscript source present | FAIL/BLOCKER | `manuscript/` absent |
| Result figures inserted into manuscript | BLOCKED | manuscript source absent |
| Every case-study figure referenced/discussed | BLOCKED | manuscript source absent |
| Canonical `figures/` release directory | FAIL/BLOCKER | absent |
| Section-by-section case-study inventory from manuscript | BLOCKED | manuscript source absent |

## Measured results already reproduced
### Wisconsin Diagnostic Breast Cancer
- Logistic regression: accuracy 0.976608, balanced accuracy 0.971890, F1 0.981481, ROC-AUC 0.994597.
- Random forest: accuracy 0.959064, balanced accuracy 0.954731, F1 0.967442, ROC-AUC 0.995765.

### Optical handwritten digits — edge quantization
- Float64 model: accuracy 0.974074; coefficient storage 5120 bytes.
- Symmetric int8 coefficients: accuracy 0.972222; coefficient storage 640 bytes.
- Storage reduction: 8x.
- Absolute accuracy decrease: 0.001852.

## Noncompliant historical material
Older Chapter 1 production scripts contain numerical reconstructions/simulation fallbacks, including a Lorenz-96 weather panel and simulated degradation trajectories. These remain valid only as conceptual/numerical illustrations. They must not be used as the empirical output/results evidence for a case study when the surrounding text claims real-world data.

## Required closure actions
1. Migrate the current Chapter 1 LaTeX manuscript into `chapters/foundations_machine_learning/manuscript/`.
2. Inventory every heading containing Case Study, Application, Example/Application, deployment workflow, or domain-specific application.
3. Map each such subsection to a committed real-data implementation.
4. Execute each implementation and retain metrics, predictions, provenance, and publication figures.
5. Move/duplicate accepted publication figures into canonical `figures/<section>/` locations.
6. Insert each output/result figure at the correct manuscript location after introductory prose.
7. Add substantive Results and Benchmark Comparison discussion for every empirical figure/table.
8. Verify every figure/table/listing is explicitly referenced and discussed.
9. Verify captions identify dataset, model, split/protocol, metric, and relevant provenance.
10. Rebuild the chapter PDF and run figure-reference, missing-asset, sparse-section, and section-title relevance audits.

## Release decision
**NOT COMPLETE / BLOCKED.** Code remediation has progressed, but Chapter 1 cannot pass Master QA until the manuscript source and canonical figures are present and every case study/application is integrated with generated real-data results.
