# Executed pretrained Faster R-CNN on Penn–Fudan real imagery

- **Verification:** GitHub Actions workflow completed successfully: https://github.com/JLondon11/AppliedML/actions/runs/41939282154
- **Dataset:** Original Penn–Fudan Pedestrian Detection and Segmentation photographs and ID masks (https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip).
- **Sample:** The first 12 images in sorted filename order, original ground-truth person instances: 21. This is an intentionally small non-random sample.
- **Model:** torchvision Faster R-CNN MobileNetV3-Large 320 FPN, official pretrained COCO weights, *not* the ResNet-50 detector planned for CrowdHuman.
- **Weights URL:** https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth
- **Checkpoint SHA-256:** `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`.
- **Environment:** PyTorch 2.15.0+cpu; torchvision 0.29.2+cpu; no training or fine-tuning in this study.
- **Evaluation:** Retain class=person detections at confidence >=0.25. Greedy confidence-ordered one-to-one match to mask-derived GT boxes, IoU >=0.50; local all-point interpolated AP50.
- **Measured:** Local AP50=0.9935064935, GT=21, predictions=31, matched GT=21, precision at all retained predictions=0.6774193548, recall=1.0. Exact values in `results/pennfudan_12/metrics.csv`.
- **Source file:** `evaluate_pennfudan_pretrained.py`, section-specific executable with genuine model predictions, CSV, PNG/PDF/SVG outputs, checkpoint/source hashes.
- **Outputs:** GitHub Actions artifact `chapter5-pennfudan-real-pretrained-inference` on the verified run; downloaded original artifact and compiled Chapter 5 PDF in the corresponding user delivery.
- **Interpretation:** Neither official Penn–Fudan benchmark AP, COCO leaderboard AP, CrowdHuman/CityPersons AP/MR, nor heavy-occlusion evaluation; no independent large-sample generalization or safety claim.
- **Chapter status:** Real-image pretrained detection *supporting baseline verified*, but named occlusion model and original CrowdHuman/CityPersons study remain BLOCKED.
