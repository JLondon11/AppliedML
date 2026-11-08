# Chapter 5: graduate textbook two-level experimental certification register

**Adopted 2026-11-08.** There are 11 case-study/application units, all of which retain their complete-unit Master QA gate.

| Unit | Real execution in record | Complete-unit status | Principal claim-matched blocker |
|---|---|---|---|
| KITTI 3D detection | **CERTIFIED original-scene PointPillars 3D GT localization experiment** CV5-EXP-KITTI-ORIGINAL-3D-GT-006 (6/6 car matches at 3D IoU 0.7, one training scene) | PENDING | Complete original held-out split with KITTI official AP40 across class and difficulty, repeated hardware-matched full-pipeline latency |
| Agricultural UAV obstacles | VisDrone aerial urban detection | PENDING | Agricultural field images and agricultural obstacle model |
| YOLO drone | Original aerial-image model predictions | PENDING | Video tracking with real identity labels and control evidence |
| Cross-dataset driving | Matched Penn-Fudan/CrowdHuman 2D pedestrian detection | PENDING | Original driving-domain paired benchmark |
| Production driving stack | Real subsystem evidence only | PENDING | Instrumented end-to-end stack trial |
| Multimodal driving | **CERTIFIED bounded genuine two-model calibrated KITTI association** CV5-EXP-KITTI-CAMERA-LIDAR-LATE-FUSION-007 (18 projected LiDAR predictions, 15 RGB detections, 9 verified same-class matches) | PENDING | Original multi-scene jointly trained camera–LiDAR feature fusion and target-task accuracy or robustness where claimed |
| Accelerated MRI | Fourier phantom only | PENDING | Authorized original fastMRI k-space and trained VarNet |
| NeRF versus 3DGS | Stereo geometry only | PENDING | Paired real-scene NeRF and 3DGS |
| Segment Anything | **CERTIFIED original SAM/COCO experiment** CV5-EXP-SAM-COCO-ORACLEBOX-001 | **APPROVED — scoped application** CV5-UNIT-09-SAM-PROMPTED-001 | Complete-unit claims explicitly limited to oracle-box mask IoU/Dice, paired GrabCut comparison, uncertainty and failures; NOT COCO AP or autonomous deployment |
| TuSimple lane detection | **CERTIFIED bounded original-test U-Net vs HSV classical comparator** CV5-EXP-TUSIMPLE-32-PAIRED-005 | **APPROVED — scoped application** CV5-UNIT-10-TUSIMPLE-LANE-003 | Certified original-photo mask-overlap teaching objective; checkpoint training history, independent generalization and official TuSimple lane-point ranking NOT claimed |
| Pedestrian occlusion | **CERTIFIED** CrowdHuman 96-image experiment (003), corrected paired CrowdHuman 48-image experiment (004; authoritative 5000-bootstrap run 42525235516), and Penn-Fudan 170-image comparison (002) | **APPROVED — scoped application** CV5-UNIT-11-PEDESTRIAN-OCCLUSION-002 | Original-data box-area proxy recall and paired frozen detector comparison certified; no official CrowdHuman MR/AP/JI or deployment claims |

**Experiment certificates:** 7. **Complete unit certificates:** 3 of 11. These counts are not interchangeable.

Sections can only be scoped down if the resulting scientific objective still teaches the concept announced in the title and covers the chapter's pedagogical requirements. Evidence cannot be obtained by renaming an unrelated experiment.

See `../GRADUATE_EXPERIMENT_CERTIFICATION_POLICY.md` and the individually numbered experiment certificates in this directory.
