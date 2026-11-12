# Chapter 5 — VisDrone real-result integrity and CI continuation (2026-11-11)

**Chapter identity:** Computer Vision Part II, Chapter 5. **Overall Master QA:** HOLD. This pass re-executed measured-CSV figure regeneration, not YOLO inference.

## Tests actually executed locally

The existing real VisDrone experimental CSV records (24 photographed scenes, 972 original GT objects, 2,589 retained predictions) were used as **unaltered inputs** to `generate_real_visdrone_transfer.py` in the local deliverable package. Results reproduced: macro local AP50 = 0.1406456330251602 and cutoff recall = 0.33127572016460904. Bus has zero annotated instances and remains excluded from AP averaging. The original recorded files have SHA256: per-class `8782178b4878ea86e8e705d5b6b8319440856f5ad971a77617ee296f781adfdb`, summary `76bcb875e09db91fdfd4b9b577a27851d7d0fcae973f9f86d63392873a795589`.

The one-page figure PDF was rendered to PNG and visually inspected. Its values, titles, units, class counts, undefined AP annotation and cross-domain limitations are readable. No original-photo overlays are substituted.

A new regression test `figure_generation/real_visdrone_transfer/test_real_visdrone_transfer.py` completed on the local package script:
- Genuine original CSV reconciliation: **PASS**
- Local metric precision reproduction: **PASS**
- Expected undefined bus AP: **PASS**
- Rejection of corrupt macro AP, recall, original GT count, detections count, and number of evaluated classes (in separate temporary copied files): **5/5 PASS**

Corrupt copies are negative tests only; not model datasets or results. The locally executed plotting script has the same CLI and source-data validation contract as the separately committed repository plotting implementation, but this report does **not** claim byte-for-byte code identity.

## GitHub artifacts committed

- Evidence regression: `b85ace93421b425c01aaebfab1d481d91b014af0`
- CI workflow: `b45096b15dff77932873110b96984950e85e9a39`

`.github/workflows/chapter5_visdrone_real_result_quality.yml` checks out the repository, runs original-record integrity and corruption-rejection tests against the committed plotting source, regenerates figure PNG/PDF/SVG, and uploads its generated evidence as a workflow artifact. **No claim of completed CI execution:** the connected API did not confirm an actual successful run, and remote workflow status must be checked independently.

## Experimental blockers unchanged

This does not close original TuSimple inference, CrowdHuman occlusion-stratified model benchmark, KITTI PointPillars, original camera–LiDAR fusion, video tracking, agricultural-field detection, fastMRI or NeRF/3DGS. The latest matching Chapter 5 LaTeX/PDF manuscript was not in this local workspace, therefore its insertion/figure placement is not verified. New generalization metrics or fabricated per-scene intervals have not been introduced.

**Scientific decision:** existing measured-figure numerical evidence QA PASS; complete Chapter 5 unit and manuscript certification HOLD.
