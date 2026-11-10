# Scientific AI Figure 11 — actual thermocouple measurement experiment

The prior Figure 11 implementation was a mathematical heat-equation simulation, not an experiment using authentic sensor data. This replacement trains a neural field with and without a simplified physical diffusion residual using **168 actual time samples from eight thermocouples** on a heated aluminum rod (1,344 measured temperatures). These are laboratory observations, not hypothetical data.

Original experimental measurements: Tariq Sawaqed, Brady Comer, Josh Colgrove and Graham Dobbie, [ASEN 3802 Heat Conduction in a Metal Rod](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab), `Part 1/Aluminum_26V_250mA.txt` (Git blob SHA `968990e38e4fe0a1f1b981d117d8bf92a90f163c`). Recorded every 10 seconds, from 0 to 1,670 seconds. Sensors are spaced 0.5 inches (12.7 mm). Do not substitute calculated data for these authentic measurements.

## Reproduce

Download the original source .txt from the cited repository into your local data folder (or use the checked source download step in the Figure 11 GitHub Actions workflow). The external dataset is **not vendored in this repository**.

```bash
python -m pip install numpy matplotlib torch
python chapters/scientific_ai/code/figure_11_pinn_residual.py \
  --data "Aluminum_26V_250mA.txt" \
  --output-dir "chapters/scientific_ai/code/figure_11_outputs" \
  --epochs 600
python chapters/scientific_ai/code/test_figure_11_real_data.py \
  --results "chapters/scientific_ai/code/figure_11_outputs/results.json"
```

**Data partition:** six complete channels CH1, CH2, CH4, CH5, CH7 and CH8 for fitting (1,008 observed temperatures); CH3 alone for selecting the physics-regularization weight (168 observed temperatures); and separate, untouched CH6 for final evaluation (168 observed temperatures). Although these sensors are from the same physical run and are therefore correlated, no measurement from CH6 or CH3 is used to train the network.

**Methods:** identical neural network architecture (2-32-32-1, tanh) at seed 29, 600 Adam steps, training-normalized inputs/targets; physical regularization weights 0, 0.002, 0.02. The residual `u_t - α u_xx` is calculated with automatic differentiation at **actual training observation coordinates**, not synthetic temperature observations. The learned parameter α is nondimensional and cannot be interpreted as independently measured aluminum thermal diffusivity; the simple PDE omits electrical heating boundary effects and convective losses.

**Measured results:** data-only model `λ=0` validation RMSE 0.1321 °C and held-out CH6 RMSE **0.1103 °C**; `λ=0.002` validation 0.1418 °C and CH6 0.1856 °C; `λ=0.02` validation 0.3440 °C and CH6 0.4937 °C. The best validation-selected model was the **data-only baseline**: a genuine negative result for the imposed simplified physics constraint.

**Outputs:** `results.json` contains the source metadata, actual measurement array, all fitted predictions, learning histories, and numerical results; `figure_11_pinn_residual_field_error.png/pdf/svg` are generated from those outputs. Figure caption must be updated: this is a thermocouple temperature and physics-regularization comparison, **not** a heat-field ground-truth error image. No material-property estimation, universal PINN advantage, or true diffusion-coefficient inference is claimed.

**Status:** real-data source and local CPU experiment verified; final chapter manuscript/source integration and GitHub Actions run confirmation are separate publication checks.
