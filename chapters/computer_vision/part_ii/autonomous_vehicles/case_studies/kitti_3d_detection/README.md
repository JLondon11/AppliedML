# Chapter 5 · KITTI 3D detection · PointPillars

**Status: OpenPCDet integration and focused unit checks completed; real inference BLOCKED.** This directory has a guarded checkpoint-backed evaluator and a renderer for *actual* model predictions, but no official KITTI LiDAR or checkpoint was accessible in the current runtime. No real PointPillars model predictions or benchmark measurements were generated or inserted in the book.

## Primary sources
- [KITTI 3D object-detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Obtain original LiDAR training data, calibration, labels and the desired validation split under the benchmark's access conditions.
- [PointPillars, Lang et al., CVPR 2019](https://arxiv.org/abs/1812.05784).
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), its [KITTI data preparation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md), and the checkpoint and configuration published in its model zoo.

OpenPCDet supplies the canonical KITTI evaluation. This project does **not** ship the official dataset, pretrained checkpoint, or benchmark results.

## Run
Install OpenPCDet and its CUDA dependencies as documented upstream. Arrange official KITTI data under OpenPCDet's `data/kitti` and generate `kitti_infos_val.pkl`. Obtain the published PointPillars checkpoint and verify its provenance. Run:

```bash
python test_local.py
python run_pointpillars_kitti.py --openpcdet /path/to/OpenPCDet \
  --kitti /path/to/OpenPCDet/data/kitti --checkpoint /path/to/model.pth \
  --output run_results --dry-run
python run_pointpillars_kitti.py --openpcdet /path/to/OpenPCDet \
  --kitti /path/to/OpenPCDet/data/kitti --checkpoint /path/to/model.pth \
  --output run_results
```

The evaluator invokes upstream `tools/test.py --save_to_file` and retains the command, file hashes and run log. After a successful run locate OpenPCDet's `result.pkl`, then make original output plots:

```bash
python render_verified_pointpillars.py \
  --result-pkl /path/to/OpenPCDet/output/.../result.pkl \
  --kitti /path/to/OpenPCDet/data/kitti \
  --output run_results/figures --score-threshold 0.35
```

Only the named-model's actual prediction boxes and actual KITTI `.bin` point-cloud samples are accepted. The renderer refuses missing or malformed prediction data. Qualitative plots are **not** official AP evaluation; report AP only from the verified official KITTI evaluator with the same split, checkpoint, and config.

## Verification done in this turn
- Local tests of oriented-box geometry, normalization of frame IDs and refusal of missing prediction fields: **PASS**.
- All three Python scripts passed compilation: **PASS**.
- Missing official inputs correctly triggered a preflight failure: **PASS**.
- GPU inference on real KITTI: **NOT PERFORMED**, because the data and model checkpoint are unavailable.
- Publication-ready model-result figures and AP tables: **NOT AVAILABLE**. Master QA remains **BLOCKED**.

Before approval supply the actual KITTI split, model/checkpoint hash, upstream framework commit, complete evaluation log, original predictions, real-model figures, official metrics, real comparison baseline and scientific analysis. Do not cite approximate AP or inferred metrics as results.
