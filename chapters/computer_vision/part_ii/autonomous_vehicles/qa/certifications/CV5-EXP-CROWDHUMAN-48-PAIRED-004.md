# CV5-EXP-CROWDHUMAN-48-PAIRED-004 — CERTIFIED (bounded original-data experiment)

**Reaudited:** 2026-11-08. **Authoritative corrected model execution:** https://github.com/JLondon11/AppliedML/actions/runs/42525235516

**Original execution artifact ZIP SHA256:** `ef036deb489c9762b4e99fac34fd0d3d4b19f291c2dc21880ae2483ad2320688`. **Per-person CSV SHA256:** `41f92546f907a001baf46719e472640fb46cd140b3314fbb31cbda0a4306a643`.

**Bounded claim:** Frozen official COCO-pretrained torchvision Faster R-CNN MobileNetV3-Large 320-FPN and ResNet50-FPN were genuinely executed on the **same 48 original CrowdHuman validation images**, containing 1,589 eligible, original full-body person annotations. Both used confidence ≥0.30 and one-to-one full-body IoU ≥0.50. Results represent local person recall stratified by a visible/full **bounding-box area** proxy, not visible pixel fraction or official CrowdHuman scores. No model was trained locally.

**Precision-boundary correction:** The predecessor artifact assigned three persons whose stored occlusion proxy was exactly 0.20 to the low group due to floating-point subtraction at the decision boundary. This certificate supersedes earlier 004 certificate measurements. The authoritative code uses `round(proxy,12)` before assigning low <0.20, medium [0.20,0.50), and heavy ≥0.50. Its genuine model inference and person match flags were rerun and unchanged; original groups are now consistent with recorded numeric thresholds. The 96-image independent ResNet50 run was audited separately and has no such recorded boundary mismatch.

| Original reference people | MobileNetV3-FPN matches | Local recall | ResNet50-FPN matches | Local recall |
|---|---:|---:|---:|---:|
| Low, n=595 | 219 | 0.3681 | 462 | 0.7765 |
| Medium, n=415 | 74 | 0.1783 | 169 | 0.4072 |
| Heavy, n=579 | 30 | 0.0518 | 57 | 0.0984 |
| Overall, n=1589 | 323 | 0.2033 | 688 | 0.4330 |

**Absolute recall difference (ResNet50–MobileNetV3):** overall 0.2297, 95% image-clustered bootstrap [0.1538,0.3134]; low 0.4084 [0.2957,0.5045]; medium 0.2289 [0.1495,0.3081]; heavy 0.0466 [0.0089,0.0841]. All intervals were recomputed from **5,000 paired original-image cluster samples**, seed 20261108; fixed weights, person annotations and image selection.

**Ten scientific gates:** 1 PASS matching task/model/dataset; 2 PASS original CrowdHuman validation photos and annotations; 3 PASS both actual frozen pretrained models executed in successful CI; 4 PASS original 48/1589 and full model/score/IoU/seed configurations; 5 PASS task-specific original person recall and same-scene architecture comparison; 6 PASS code-generated corrected proxy-group and per-image model-output scientific figure integrated into §5.8; 7 PASS organized executable source, source data and model references, artifact/CSV SHA256 and verifier; 8 PASS local results explicitly separated from official AP/MR/JI and literature; 9 PASS 5000 image-cluster paired bootstrap plus scale/density and severe-occlusion failure analyses; 10 PASS independent 3178-record validation, bin boundary check, recomputed intervals, table/figure consistency, successfully rebuilt and visually inspected 122-page chapter.

**Scope limitations:** This is a fixed 48-image original validation subset, not full 4,370-image CrowdHuman validation. The two models differ in backbone, input preprocessing and other training properties; measured differences are not causal architecture ablations. The proxy is based on rectangular annotation areas, not visible body pixels. This experiment does not establish official CrowdHuman mAP, miss-rate or Jaccard; specialized part-based detector performance, tracking, or deployment safety are untested.

**Approval classification:** internal scientific audit of bounded executed experiment, not independent external accreditation; no automatic approval of broader Chapter 5 claims.
