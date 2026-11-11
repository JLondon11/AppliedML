# Scientific AI Chapter 11 — authentic scientific digital twin experiment and PDF integration (2026-11-11)

**Figure 11.54 now shows an actual executed laboratory result instead of an unsupported multiscale architecture.** The original source is Tariq Sawaqed and collaborators' [Heat-Conduction-Aerospace-Lab](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab), original eight-thermocouple heated aluminum-rod log `Part 1/Aluminum_26V_250mA.txt`, Git blob `968990e38e4fe0a1f1b981d117d8bf92a90f163c`. The original recorded values were independently checked against the local research copy.

### Real observations, execution and negative/positive evidence

- 168 recorded time points (0--1670 seconds, every 10 seconds), eight original sensors.
- Chronological split: 100 model-training timestamps, 30 validation timestamps, 38 untouched test timestamps.
- Sensor observation restriction in validation/testing: CH1, CH3, CH5, CH7 visible; CH2, CH4, CH6, CH8 withheld. The final time block contains **152 actually measured withheld target values**.
- Ridge-learned linear state transition, Kalman-style observation assimilation; process/measurement covariance parameters chosen using validation only.
- Held-out RMSE: assimilation **0.0688879493 °C**, static sensor regression **0.0721520553 °C**, unassimilated dynamic prior **0.0780471483 °C**. This gain is *small* and mixed across target channels. Predictions and measurements were saved, and RMSE/MAE recomputed directly.
- The complete experiment was run twice locally, producing byte-identical numeric result JSON. The four-panel premium-color scientific figure was visually inspected.

### Manuscript and figure QA

Updated the *full original LaTeX chapter* delivered in `Chapter11_MasterQA_Figure1101_20261110.zip`: replaced the `Application: Scientific Digital Twins` section with scientifically accurate background, sensor/temporal partition, numbered and referenced Kalman prediction/update equations, measured model comparisons and limitations. Replaced the old conceptual image asset at the existing filename `Figures/Figure_11_53_Bayesian_Multiscale_Digital_Twin_Assimilation.png`; preserved the original LaTeX label. **The image's historical filename number 53 is not the current figure number:** its compiled caption is **Figure 11.54**, located on PDF page 147 (printed page 142).

Two *complete clean* pdflatex passes generated a **159-page PDF**. Final log: no undefined citations or references; all 187 labels unique. The figure page was rendered and inspected; changing placement to `[H]` removed a preceding largely blank float-only page and placed figure with its discussion.

PDF SHA256: `34571e00d8b1e7df43eeea5631eabda047a227c17009c590dbab91fd79aedb87`.
New Figure 11.54 PNG SHA256: `16cd03b569bcfce6b74ba4069abd9d15f2f3567d27b16bd2de58c558d8151276`.

### GitHub source and status

- `chapters/scientific_ai/code/figure_11_54_real_rod_digital_twin/run.py`: full experiment, numeric records and PNG/PDF/SVG.
- `chapters/scientific_ai/code/figure_11_54_real_rod_digital_twin/test_outputs.py`: result and artifact arithmetic checks.
- `chapters/scientific_ai/code/figure_11_54_real_rod_digital_twin/measured_summary_20261111.json`: executed real-data results and limitations.
- `.github/workflows/scientific-ai-real-digital-twin.yml`: pull original source, verify original Git blob, execute and archive actual figures.
- `chapters/scientific_ai/qa/CH11_APPLICATION_FIGURE_EVIDENCE_MATRIX_20261110.json`: updated unit status.

The complete revised LaTeX, compiled PDF and binary real-data figures are distributed in `Chapter11_MasterQA_Real_Digital_Twin_20261111.zip`, but **the manuscript and binary graphics have not been independently confirmed committed to GitHub**. Workflow has been authored but no passing CI run was confirmed.

### Master QA decision

**Scientific Digital Twins: real-data result-figure coverage achieved for one authentic laboratory system.** Broad multi-domain clinical, climate and fleet claims are not experimentally validated, and there is no statistically independent external test. Other Chapter 11 named units still require output evidence. MoNuSeg / paired malignant nucleus classification remain on scientific HOLD. **Whole-chapter and whole-book certification remain HOLD.**
