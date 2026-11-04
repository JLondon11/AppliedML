# Chapter 5 — Production autonomous-driving stack: recorded evidence

**Master QA: BLOCKED.** This application has a measurement/visualization implementation `evaluate_pipeline_trace.py`, but there was **no real recorded vehicle/simulator trace** in the available files. It generates a measured latency/coverage/intervention results table and scientific 3-panel figure **only when** a genuine recorded pipeline trace is supplied. Its test file contains *fixture-only* telemetry and has no experimental validity.

Input CSV columns (strict): `frame_id,timestamp_s,perception_ms,tracking_ms,planning_ms,obstacle_count,track_count,ego_speed_mps,planner_valid,emergency_brake`. Require >=20 unique, strictly increasing real recorded frames. Values must come from a documented, deployed perception–tracking–planning pipeline, with hardware, firmware, model checkpoints, timestamps, experiment scene and ground truth recorded independently. This is **not** a substitute for dataset-specific perception metrics, tracking IDF1/HOTA or measured closed-loop safety performance.

```bash
pip install numpy matplotlib
python test_local.py
python evaluate_pipeline_trace.py --trace /real/recorded_stack.csv \
 --deadline-ms 100 --output results/real_stack
```

Outputs: `metrics.csv`, `real_pipeline_trace.{pdf,png,svg}`, `provenance.json`. Intervention commands are not actual crashes or safety benefits; latency values are machine- and instrumentation-specific. The next scientific step is to instrument and execute a real simulated/physical autonomy stack with sourced perception inputs and explicit closed-loop benchmarks. No original vehicle trace exists in this run.
