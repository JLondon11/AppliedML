# Chapter 5 — real-photograph executable baselines

Run `python generate_real_photo_baselines.py` with NumPy, OpenCV, scikit-image and Matplotlib. The photographs come from the locally packaged `skimage.data.astronaut()` and `skimage.data.stereo_motorcycle()` examples. The latter is a Middlebury stereo-scene sample. Original dataset entries: https://scikit-image.org/docs/stable/api/skimage.data.html and https://vision.middlebury.edu/stereo/data/scenes2014/.

The three-panel figures show **actual code outputs** (mask and disparity), with black panel labels beneath the panels. These are algorithm-level baselines, **not** inference from Segment Anything Model (SAM), NeRF, or 3D Gaussian Splatting. The GrabCut mask has no supplied ground truth; do not infer segmentation accuracy. Stereo metrics are computed only at pixels with finite positive ground-truth disparity and positive predicted disparity below 96 pixels. The stereo result is not a novel-view rendering evaluation.

Recorded outputs on the chapter environment: GrabCut foreground pixels: 62,948. Stereo MAE: 1.2226076 pixels, RMSE: 5.3158202 pixels, valid fraction: 0.76848313. Re-run to reproduce the figures and CSV (depending on library versions tiny changes may arise). Do not cite these as a SAM/NeRF/3DGS benchmark.
