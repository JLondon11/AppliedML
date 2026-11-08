# Chapter 5: graduate textbook two-level experimental certification register

**Adopted 2026-11-08.** There are 11 case-study/application units, all of which retain their complete-unit Master QA gate.

| Unit | Real execution in record | Complete-unit status | Principal claim-matched blocker |
|---|---|---|---|
| KITTI 3D detection | PointPillars, 1 genuine KITTI LiDAR scene | PENDING | Multi-scene 3D detection metric where broad accuracy is claimed |
| Agricultural UAV obstacles | VisDrone aerial urban detection | PENDING | Agricultural field images and agricultural obstacle model |
| YOLO drone | Original aerial-image model predictions | PENDING | Video tracking with real identity labels and control evidence |
| Cross-dataset driving | Matched Penn-Fudan/CrowdHuman 2D pedestrian detection | PENDING | Original driving-domain paired benchmark |
| Production driving stack | Real subsystem evidence only | PENDING | Instrumented end-to-end stack trial |
| Multimodal driving | Genuine calibrated PointPillars plus Faster R-CNN late fusion on KITTI | PENDING | Jointly trained sensor fusion where claimed |
| Accelerated MRI | Fourier phantom only | PENDING | Authorized original fastMRI k-space and trained VarNet |
| NeRF versus 3DGS | Stereo geometry only | PENDING | Paired real-scene NeRF and 3DGS |
| Segment Anything | **CERTIFIED original SAM/COCO experiment** CV5-EXP-SAM-COCO-ORACLEBOX-001 | **APPROVED — scoped application** CV5-UNIT-09-SAM-PROMPTED-001 | Complete-unit claims explicitly limited to oracle-box mask IoU/Dice, paired GrabCut comparison, uncertainty and failures; NOT COCO AP or autonomous deployment |
| TuSimple lane detection | Pretrained U-Net on 32 original test-partition frames | PENDING | Checkpoint train membership and suitable task reference/comparator |
| Pedestrian occlusion | **CERTIFIED** CrowdHuman 96-image experiment (003), corrected paired CrowdHuman 48-image experiment (004), and Penn-Fudan 170-image comparison (002) | **APPROVED — scoped application** CV5-UNIT-11-PEDESTRIAN-OCCLUSION-002 | Original-data box-area proxy recall and paired frozen detector comparison certified; no official CrowdHuman MR/AP/JI or deployment claims |

**Experiment certificates:** 4. **Complete unit certificates:** 2 of 11. These counts are not interchangeable.

Sections can only be scoped down if the resulting scientific objective still teaches the concept announced in the title and covers the chapter's pedagogical requirements. Evidence cannot be obtained by renaming an unrelated experiment.

See `../GRADUATE_EXPERIMENT_CERTIFICATION_POLICY.md` and the individually numbered experiment certificates in this directory.
