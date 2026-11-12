# Chapter 5 — Highway lane-detection application

**Master QA: BLOCKED** until actual lane-specific checkpoint inference on original TuSimple benchmark samples is executed. Script: `evaluate_tusimple_onnx.py`. Dataset documentation: https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection (the `raw_file`, `lanes` and `h_samples` fields are parsed).

```bash
pip install numpy opencv-python onnxruntime
python evaluate_tusimple_onnx.py --image-root /data/tusimple \
 --annotation-jsonl /data/tusimple/label_data_0313.json \
 --checkpoint-onnx /models/lane_segmentation.onnx --output-semantics logits --output results/
```

The required `--output-semantics` flag must match the checkpoint's training/export contract (`logits` or `probability`). Do not infer this from the current output tensor's numeric range: a logits tensor may temporarily lie within [0,1], causing an erroneous threshold if treated as probabilities.

Input model contract: one 1x3xHxW RGB float tensor, normalized to [0,1], and one 1x1xHxW lane-mask logits or probabilities output. Verify preprocessing and exact checkpoint identity manually. Architectures with row-anchor/class logits (e.g. UFLD) require an appropriate separate adapter; do not pass those tensors as binary mask probabilities.

The script writes actual predicted-mask overlays, per-image CSV and aggregate pixel metrics. These are **local rasterized-polyline scores**, not TuSimple's official lane-accuracy benchmark. Each dataset and trained checkpoint must be cited with checksum, implementation commit, split, metrics, and representative outputs. Do not claim any execution took place in this GitHub commit; algorithm-only unit tests do not meet model-inference QA.
