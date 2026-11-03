# §5.6.8 — Executed generated-scene training/tracking/planning

The companion script `execute_train_track_plan.py` uses only generated sensor-like descriptors and generated 2D scene trajectories. It trains a logistic-regression classifier on 12,000 feature samples, tests on 4,000 independent samples, then runs 160 independent generated planning episodes. It uses bipartite-assignment Kalman tracking and compares static-detection lane selection with a velocity-predicting lane choice. The code does **not** use KITTI, nuScenes, Waymo, or Argoverse 2; it does not run a vehicle simulator, test a real neural detector, or establish physical safety.

Reproduce using Python 3, NumPy, SciPy, scikit-learn, and Matplotlib:

```bash
python execute_train_track_plan.py
```

The script creates `train_track_plan_metrics.csv`, `train_track_plan_episodes.csv`, `Figures/train_track_plan_results.{png,pdf,svg}` and `Figures/train_track_plan_scene.{png,pdf,svg}`.

Seeds: classifier training 56831 and independent tests 56832. The 160 episodes compare collision fractions under a simplified three-lane, constant-ego-speed (6 m/s) and 3-second horizon model. The tracker RMSE is per coordinate and uses geometric nearest-neighbor matching; it must not be called IDF1/HOTA. The planning fractions (0.20625 and 0.1875) are descriptive of this run only, not a statistically established safety benefit. Source numbers are the outputs committed in `train_track_plan_metrics.csv`; reevaluate on your environment if dependent-library versions change.

The matching Chapter 5 figure is Figure 5.37 in the particular PDF build; number may change after editing. Use its LaTeX label `fig:cv5:train-track-plan-results` rather than hardcoding this number in source.
