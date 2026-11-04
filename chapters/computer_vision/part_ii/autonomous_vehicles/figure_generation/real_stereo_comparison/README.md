# Chapter 5 — Real Middlebury stereo inference

**Scientific provenance.** Uses the one-quarter-resolution version of the real [Middlebury stereo motorcycle scene](https://vision.middlebury.edu/stereo/data/scenes2014/) distributed as `skimage.data.stereo_motorcycle()`. Both input images and the ground-truth disparity are from the same sample.

Run `python compare_middlebury_stereo.py` with NumPy, OpenCV, scikit-image and Matplotlib installed. Exact parameter settings and validity rules are in the code. The real-dataset metrics are in `stereo_method_metrics.csv`; outputs are generated locally in `Figures`. Runtime is measured locally and is not portable across hardware.

The Bad-3 rate counts invalid/missing predictions as bad relative to the finite positive reference disparities below 96px. MAE/RMSE apply to those reference pixels where the estimated disparity is also valid, making the coverage column essential to interpretation.

**Important limit:** These are results from classical OpenCV StereoBM and StereoSGBM implementations. They are not RAFT-Stereo, model-fusion, Waymo, KITTI, or NuScenes inferences. They do not validate camera–LiDAR fusion or vehicle-level autonomy. There is one evaluated real scene, so no cross-scene generalization estimate.

The scikit-image sample is downsampled by a factor of four from the full-resolution Middlebury 2014 scene, with matching reference disparities. The reported measurements use a locally defined Bad-3 metric and are **not** official Middlebury leaderboard results. Dataset documentation: https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.stereo_motorcycle
