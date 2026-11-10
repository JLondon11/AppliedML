# Scientific AI Figure 11 — real thermocouple remediation (2026-11-09)

## Scientific evidence replacement

The original `chapters/scientific_ai/code/figure_11_pinn_residual.py` trained a PINN using randomly chosen collocation points and an exact analytical heat-equation solution, and drew residual and field-error surfaces. These surfaces were computations on mathematical data, **not recorded experimental heat measurements**. They must not be called actual observed field error.

The script has now been **replaced** with executable experimental analysis based on a real laboratory heat-conduction test. Provenance: Sawaqed, Comer, Colgrove, Dobbie, [Heat-Conduction-Aerospace-Lab](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab); original `Part 1/Aluminum_26V_250mA.txt`, Git blob SHA `968990e38e4fe0a1f1b981d117d8bf92a90f163c`. This is a real heated-aluminum rod thermocouple acquisition (26 V, 250 mA), with 8 thermocouples at 12.7-mm spacing and 168 time samples from 0 to 1,670 seconds. The experimental data were retrieved and reconstructed numerically without sampling invented values. The authentic numerical measurements, not artificial heat-equation targets, drive all fitted models and measured figure panels.

## Partition and executed experiment

- Training: entire channels CH1, CH2, CH4, CH5, CH7, CH8 = **1,008 recorded temperature measurements**.
- Validation: complete independent channel CH3 = **168 recorded temperature measurements**.
- Held-out test: previously unused channel CH6 = **168 recorded temperature measurements**.
- 600 Adam steps at fixed seed 29 on a 2-32-32-1 tanh neural field. One no-physics variant and two variants with a simplified dimensionless squared diffusion residual `u_t - alpha*u_xx`. The differential residual is evaluated at actual measured *training* coordinates; no synthetic temperatures are introduced.
- Regularization weight `lambda` chosen using *only* the held-out validation sensor. The test sensor was not used for model selection.

| Physics weight | Validation CH3 RMSE (°C) | Untouched CH6 RMSE (°C) |
|---:|---:|---:|
| 0.0 | 0.13208838 | **0.11026219** |
| 0.002 | 0.14177403 | 0.18555533 |
| 0.02 | 0.34404158 | 0.49366333 |

**Scientific conclusion:** the simplified physics residual **hurt** held-out sensor interpolation; the data-only model was correctly selected by validation. The fitted nondimensional coefficient cannot be interpreted as a calibrated material diffusivity because boundary electrical heating and heat losses are not physically modeled.

## Verification

- Two separate full local model runs matched the entire JSON result file byte-for-byte (SHA-256 `54b63431f91a432db4f9ce28e3681c83a88f1f8227666772e59f5d414ded2248`).
- Numerical source temperature matrix SHA-256 as 64-bit floats `1e8972a871c8b45ef0a2a646611e75b9b151e1bb5fb6d4998deecfa272e0154f`. The original Git provider blob SHA above is the authoritative raw source-file identifier.
- Predictions were checked against raw observed temperatures to independently calculate RMSE at both held-out channels for all three variants.
- Generated PNG, SVG and PDF figures were visually inspected, and the PDF was independently rendered to PNG.
- The updated figure is a **four-panel authentic observational temperature and model-error comparison**, not an original analytical PDE field-error heatmap.

## Source and assets

- Executable GitHub script: `chapters/scientific_ai/code/figure_11_pinn_residual.py`.
- GitHub checks: `chapters/scientific_ai/code/test_figure_11_real_data.py`.
- Verified measured-value summary: `chapters/scientific_ai/code/figure_11_real_thermocouple_verified_summary.json`.
- Source and protocol documentation: `chapters/scientific_ai/code/figure_11_heat_real_data_README.md`.
- Manuscript insertion with corrected caption: `chapters/scientific_ai/code/figure_11_heat_real_data_manuscript.tex`.
- Independent CI for original source blob and 600-step measured-data model: `.github/workflows/scientific-ai-real-thermocouple-figure11.yml`.
- Complete locally executed figure PNG/PDF/SVG, numerical arrays and repeatability audit: `Scientific_AI_Figure11_Real_Lab_Thermocouples_20261109.zip`.

## Publication status and limitations

**Local authenticity and actual execution verified.** Original independent GitHub Actions execution is not yet confirmed, and the final chapter manuscript/PDF still needs replacement of the original analytical heat-equation prose and caption. There is only one laboratory run; the sensors are correlated within the rod. No model here establishes generalized industrial heat-transfer accuracy. Full global figure and manuscript certification remains **HOLD**.
