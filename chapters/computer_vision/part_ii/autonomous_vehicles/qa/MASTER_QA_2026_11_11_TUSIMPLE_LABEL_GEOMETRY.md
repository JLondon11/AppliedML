# Chapter 5 — TuSimple ground-truth rasterization correction (2026-11-11)

**Master QA:** HOLD. **Scope:** validity of a local experimental evaluator, not a new trained-model inference result.

## Identified scientific error

The previous `case_studies/highway_lane_detection/evaluate_tusimple_onnx.py` implementation collected all valid lane sample coordinates, discarded `-2` and other invalid entries, then drew one polyline across the retained points. An invalid sample between two valid points could therefore be bridged by a drawn line even though **no intermediate annotation existed**. A benchmark proxy should not silently create ground-truth lane pixels across absent samples.

## Code correction

Commit `747bb47961f27692da373055c79abbffd77e77ea` changes `truth_mask` to rasterize only contiguous valid x/y sample runs and split at missing or out-of-frame coordinates. Isolated valid samples remain directly rasterized as dots. Input dimension and thickness validation were added.

Commit `beebba93a825c4ed1e398f54d24e91235c5b6633` extends `test_local.py` with continuity, missing-label, out-of-bounds, and annotation-length regression checks.

Commit `fbab1a0bdeff589544c2bad22268b9a3992f159f` documents the semantics and scientific limits in the case-study README.

## Local evidence

A targeted NumPy/OpenCV geometry regression was executed locally on a deliberately constructed annotation fixture with `h_samples=[2,10,18]`, `lane=[3,-2,17]`. The original implementation drew **17 pixels**, including an artificial pixel at the center of the gap; the corrected logic preserved only the two observed isolated points (**2 pixels**) and did not draw the gap-center pixel. A valid continuous-three-sample lane still rendered correctly. These tiny fixtures are **software tests, not synthetic scientific experiments, manuscript result plots or model inference**.

The exact GitHub test file was committed but no remote CI run for these newest commits is confirmed in this pass.

## Non-substitutable experimental gate

The local pixel IoU is not TuSimple official lane accuracy. Certification still requires access to the original TuSimple images and annotation splits, verified compatible pretrained ONNX lane checkpoint and preprocessing contract, actual prediction outputs and model SHA, official lane-point scoring where claiming official benchmark metrics, scientifically informative original-photo result overlays and failure cases, committed complete code and measured assets, manuscript text/caption integration and rendered PDF visual inspection.

**Decision:** Evaluator geometry correctness improved; CS10 and Chapter 5 remain NOT CERTIFIED. No new measured TuSimple score or figure produced.
