# Chapter 11 — Scientific digital twins: measured heat-rod assimilation

This is an **executed single-laboratory real-data experiment**, not a schematic and not a claimed aerospace/clinical/climate benchmark. The original measured source is [Heat-Conduction-Aerospace-Lab](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab), `Part 1/Aluminum_26V_250mA.txt`, Git blob SHA `968990e38e4fe0a1f1b981d117d8bf92a90f163c`. The source contains 168 time points and eight measured thermocouple channels, sampled every 10 seconds. The original laboratory dataset is owned by its source authors; obtain it from their repository rather than treating it as author-created measurements.

## Execute
```sh
python -m pip install numpy scikit-learn matplotlib
curl -L --fail 'https://raw.githubusercontent.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab/main/Part%201/Aluminum_26V_250mA.txt' -o Aluminum_26V_250mA.txt
git hash-object Aluminum_26V_250mA.txt
# Original blob should be 968990e38e4fe0a1f1b981d117d8bf92a90f163c
python run.py --data Aluminum_26V_250mA.txt --output-dir outputs
python test_outputs.py --evidence outputs/figure_11_53_measured_evidence.json
```

The locally executed research copy was normalized to Unix line endings, so its raw file SHA differs from GitHub's original blob SHA even though the **numeric column sums were independently matched to the source repository**. Both files preserve the same 168x8 measured values. The script includes the actual input's SHA-256 in the result evidence.

## Experimental protocol
The original 168 timestamps are split chronologically: first 100 for fitting the 8-dimensional linear dynamical transition, next 30 for selection of process and sensor noise variances, final 38 for untouched evaluation. On each validation and test timestamp, only CH1, CH3, CH5 and CH7 are assimilated; CH2, CH4, CH6 and CH8 are never supplied as observations. The test contains 152 withheld-sensor measured target values. The training measurements contain all eight original channels; withholding applies to validation/test inference only.

An empirical Ridge transition supplies the dynamic prior. A Kalman-style observation update estimates latent states from available sensor readings. The model is compared with a static four-to-four sensor Ridge regression and the unassimilated one-step prior. All models learn solely from training data; noise settings are selected on validation data. No test target labels enter tuning.

Measured held-out aggregate RMSE (°C): dynamic assimilation **0.06888794934034266**; contemporaneous static regression **0.07215205526787877**; unassimilated dynamics prior **0.07804714830189091**. This is a *small* improvement over the static comparator, and it does **not** hold for every sensor: CH4 and CH6 have slightly worse assimilation RMSE. Do not claim statistically independent test cases (same apparatus and experiment), calibrated uncertainty, or true material diffusivity.

The rendered asset `Figure_11_53_Bayesian_Multiscale_Digital_Twin_Assimilation.png` is **Figure 11.54 in the compiled textbook**. It contains four science panels: true sensor histories, true CH6 held-out measurements and model outputs, empirical absolute-error CDFs for 152 actual targets, and per-sensor RMSE. The historical filename is preserved to maintain the existing `\includegraphics` path; the caption does not call this a multiscale or medical experiment.

**Master QA:** real-data source and executed local test/prediction arithmetic verified; figure integrated in current Chapter 11 draft. This closes one application-level measured-results-figure gap in the inspected manuscript. Comprehensive chapter certification still HOLD; no clean-checkout GitHub CI completion has yet been confirmed.
