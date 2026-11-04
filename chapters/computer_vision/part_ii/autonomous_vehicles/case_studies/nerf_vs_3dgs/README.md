# Chapter 5 — Matched Nerfacto and Gaussian splatting evaluation

**Master QA: BLOCKED pending actual training/evaluation on a real posed dataset and GPU.** `run_nerfstudio_pair.py` runs two named **Nerfstudio implementations** on the *same* real scene and filename-defined evaluation split: `nerfacto` (NeRF family) and `splatfacto` (Gaussian splatting). It captures actual train logs, `ns-eval` outputs, rendered novel views, checkpoint configs, training wall-clock time, PSNR, SSIM, LPIPS, and dataset transforms hash.

References: [Nerfstudio CLI](https://docs.nerf.studio/reference/cli/ns_train.html), [ns-eval](https://docs.nerf.studio/reference/cli/ns_eval.html), [original Gaussian Splatting paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [Mip-NeRF360](https://jonbarron.info/mipnerf360/). **Nerfacto is not the original Mip-NeRF360 method**; do not re-label this as replicating the chapter's historical table or as its exact paper architectures.

Example after installing Nerfstudio and CUDA per official instructions:
```bash
python test_local.py
python run_nerfstudio_pair.py --scene /original/posed_real_scene --output results/comparison
```

The dataset must contain `transforms.json`, at least 12 real images, valid camera pose transforms, and a documented **held-out file-defined evaluation split**. The tool records real metrics only when both training and evaluation programs succeed. **It does not measure FPS**; model rendering throughput must be measured in a separate controlled hardware protocol before comparing real-time suitability. No dataset or GPU-based results were available in this environment, and none are invented.

The local unit test verifies dataset/provenance and JSON output contracts only. Named-model real-data figures, table values and manuscript integration remain to be completed.
