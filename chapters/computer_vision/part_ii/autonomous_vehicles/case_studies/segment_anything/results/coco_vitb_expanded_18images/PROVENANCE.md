# Expanded executed SAM real COCO validation

**Successful original checkpoint inference:** https://github.com/JLondon11/AppliedML/actions/runs/41980537230

Official Meta SAM ViT-B SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`. Original COCO 2017 validation JPEG and instance annotation archives were verified against pinned checksums in the workflow. Pseudo-random **image-ID** shuffle seed 20261104 selected first 18 eligible distinct photos (each at least two noncrowd instance annotations with area>=3000 px). This is conditionally selected, not an official COCO test split. The original source image names, hashes, selection policy and original annotation hash are preserved in the workflow's `selection_provenance.json` artifact.

Real mask inference generated **59** oracle-box SAM masks: mean mask IoU **0.77732861175** and Dice **0.85788043966**. On **35** identical valid GrabCut box prompts, mean SAM IoU **0.81835742403**, mean GrabCut IoU **0.62150559368**, paired difference **0.19685183035**. The other 24 were excluded from paired baseline evaluation because GrabCut's rectangle initialization did not accept those prompts. They remain in the all-object SAM result.

Exploratory image-cluster bootstrap with **10,000** draws, random seed 20261104: SAM all-object IoU 95% interval **[0.7280471,0.8195956]**; paired SAM interval **[0.7717234,0.8625889]**; paired GrabCut interval **[0.5087083,0.7254713]**; paired difference interval **[0.1096129,0.2848496]**. Bootstrap source is `sam_bootstrap.py`; exact generated table/plot and all 59 object-level results remain in the full successful workflow artifact and chapter source delivery.

The earlier first-three-image pilot found mean SAM IoU **0.8628225** on eight objects; the expanded study shows lower average IoU on a different, broader sample. Neither sample supports official COCO AP, automatic instance segmentation without oracle prompts, heavy-occlusion robustness, a trained model comparator or arbitrary-population inference.

Manuscript: Chapter 5 Computer Vision Part II, **Table 5.14 and Figures 5.48–5.49**, in the 105-page revised PDF. Complete source and outputs also shipped in the local delivery archive.
