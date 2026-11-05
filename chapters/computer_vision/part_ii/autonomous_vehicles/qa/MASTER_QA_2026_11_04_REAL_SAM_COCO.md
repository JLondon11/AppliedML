# Chapter 5 — Genuine trained SAM ViT-B inference on original COCO instance masks

**Verified successful source execution**: https://github.com/JLondon11/AppliedML/actions/runs/41979092197

- Original Meta SAM ViT-B checkpoint, SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.
- Source images and segmentation masks: original COCO 2017 val2017 archives from hash-verified HTTPS source mirror. Original annotation archive SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`; original val2017 image ZIP SHA256 `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`.
- Selected original photo IDs: 139, 632, 776. Deterministic nonrandom selection: ascending ID, >=2 noncrowd instances >=3000 annotated pixels; first 8 eligible original COCO annotated instances; oracle GT box prompts, no SAM fine-tuning.
- Real trained-model inference and real segmentation ground-truth comparison: mean SAM IoU `0.8628225174975672`, mean Dice `0.9251755029211902`, N=8.
- On N=6 eligible identical-prompt paired objects, SAM IoU `0.8747302910070786`; original OpenCV GrabCut IoU `0.5260045809447624`. Other two GrabCut prompts invalid due to border constraints; no imputation.
- Visualized bed object (COCO image 632, instance 315724): SAM IoU `0.9040033749381746`, GrabCut IoU `0.9538623499076941`. Model not uniformly dominant on every object.
- Executed original model outputs available as PNG/PDF/SVG, CSV, JSON, and source hashes in the successful GitHub Actions artifact `chapter5-original-SAM-ViTB-real-COCO-inference`. JSON and per-object numeric evidence also committed under `case_studies/segment_anything/results/coco_vitb_3images/`.
- The 102-page revised **Chapter 5: Computer Vision Part II — Autonomous Systems** source/PDF, generated figures and numerical results are available together in the user-delivered ZIP. Scientific manuscript has Table 5.13 and Figures 5.46–5.47. Previously executed classical GrabCut demonstration is separately identified as Figure 5.48. No unexecuted model results were inserted.
- Two final LaTeX builds succeeded; no undefined citations/cross-references or overfull box warnings; 49 figure and 15 table environments.

**Scope caveat**: the result is a very small oracle-box local binary-mask IoU evaluation. Not the COCO AP metric, not a representative full-COCO mean, not prompt-free automatic instance segmentation, not a comparison to a supervised trained task model. SAM-specific larger-sample official protocol, confidence intervals, prompt robustness, and target-domain latency remain pending. Chapter-wide Master QA **NOT APPROVED** while other named case studies remain unverified.

**Source workflow**: `.github/workflows/chapter5_real_sam_coco_inference.yml`; `case_studies/segment_anything/prepare_real_coco_subset.py`, `evaluate_sam_coco.py`.
