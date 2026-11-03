# Chapter 5 — Figure 5.38 (camera–LiDAR projection)

This figure is generated from a deterministic geometric construction, **not** real KITTI data, LiDAR hardware, or trained-model predictions.

Run `python generate_calibration_geometry.py` in this directory (Python 3, NumPy, Matplotlib). It creates `calibration_geometry.png`, `calibration_geometry.pdf` and `calibration_geometry.svg`.

- Seed: `538`.
- LiDAR coordinates: x forward, y left, z up; fixed road plane, two cuboid surfaces and a pole.
- Camera extrinsics: `R=[[0,-1,0],[0,0,-1],[1,0,0]]`, `t=[0.10,0.25,-0.05]` meters.
- Intrinsics: `fx=fy=680` pixels; principal point `(640,360)` pixels; output plane `1280 x 720`.
- Validation: positive camera depth, independent pinhole-coordinate assertions, and `det(R)=1`.
- Panel (a): world-coordinate points colored by range; panel (b): the same points in the optical image plane with identical colors.
- Data usage: illustrative and synthetic geometry only; do not cite as measured model performance.

The matching chapter LaTeX caption must report **680**, not 600, pixels for focal length. Figure source, exact seed and parameter values are versioned here to enable review.
