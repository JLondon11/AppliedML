# CV5-UNIT-11-PEDESTRIAN-OCCLUSION-002 — APPROVED (claim-scoped graduate-level application)

**Date:** 2026-11-08. **Revision:** corrected original paired CrowdHuman rerun 42525235516 and revised Chapter 5 Pass 20, 122 pages. **Unit:** Application: Pedestrian Detection Under Occlusion (§5.8).

## Scope of the approved scientific and educational claim

The application demonstrates original-data measurement of **frozen pretrained full-body pedestrian detector recall under an explicit bounding-box annotation occlusion proxy**, a true two-model matched-image comparison, cluster-based uncertainty, and meaningful limitations. It does *not* claim evaluation of specialized part-based networks, CityPersons official miss rate, CrowdHuman official AP/MR/JI, complete validation-set generalization, causal isolation of architecture, or operational safety.

The previously overbroad objective and model descriptions were corrected so the text names the Faster R-CNN ResNet50-FPN and MobileNetV3-Large 320-FPN detectors actually executed. Descriptions of part-based models, CityPersons and official miss rates are contextual research material, clearly distinguished from executed experiments.

## Authentic executed experiments supporting the unit

1. **Original CrowdHuman single detector** `CV5-EXP-CROWDHUMAN-96-OCCLUSION-003`: 96 original validation images, 2,296 annotated people, real COCO-pretrained ResNet50-FPN inference; 1,070 matched eligible full-body persons; low 653/861=0.7584, medium 325/638=0.5094, heavy 92/797=0.1154; 5,000 original-photo-cluster bootstrap replicates. No boundary mismatch in its recorded results.
2. **Original CrowdHuman two-detector matched comparison** `CV5-EXP-CROWDHUMAN-48-PAIRED-004`: 48 distinct original sample photos, 1,589 eligible annotated people, 3,178 actual per-person/model match records, COCO-pretrained MobileNetV3-FPN vs ResNet50-FPN. After stable numerical boundary rule, overall recall 323/1589=0.2033 versus 688/1589=0.4330. Low 219/595=0.3681 vs 462/595=0.7765; medium 74/415=0.1783 vs 169/415=0.4072; heavy 30/579=0.0518 vs 57/579=0.0984. Paired overall difference 0.2297 with 5,000-draw 95% image-cluster bootstrap CI [0.1538,0.3134]. **Corrected actual model rerun:** https://github.com/JLondon11/AppliedML/actions/runs/42525235516; artifact SHA256 `ef036deb489c9762b4e99fac34fd0d3d4b19f291c2dc21880ae2483ad2320688`.
3. **Original Penn–Fudan contrast** `CV5-EXP-PENNFUDAN-170-PAIRED-FRCNN-002`: 170 of 170 original images, 423 annotated people, two real model architectures and genuine locally computed AP50/uncertainty; complementary cross-domain detection evidence, **not** an occlusion result.

The separate 12-photo Penn–Fudan pilot is explicitly introductory and not used as full-set benchmark evidence. The 96- and 48-image CrowdHuman experiments are **different seeded samples and are never pooled**. Training and checkpoint configuration are independently documented; no local fine-tuning occurred.

## Ten graduate textbook scientific criteria

| # | Criterion | Finding |
|---|---|---|
| 1 | Topic/dataset/model alignment | PASS — original CrowdHuman full/visible annotated pedestrians and two named pretrained detectors directly match proxy-occlusion application |
| 2 | Original-data held-out protocol | PASS within claimed scope — authenticated validation photographs and original ODGT; explicitly nonrepresentative subset; no local training |
| 3 | Actual named-model execution | PASS — successful original pretrained inference on both 96-photo single model and 48-photo matched pair |
| 4 | Counts/configuration | PASS — 96/2296, 48/1589, 170/423; fixed confidence and IoU, image-selection seed, checkpoint, no local training |
| 5 | Meaningful metrics/comparator | PASS — one-to-one original full-body IoU50 recall by occlusion proxy with paired architecture comparator; complementary full-set Penn AP50 |
| 6 | Scientific figures from real outcomes | PASS — calibrated measurements visualized with code-generated group recall and per-image plots in correct section; original prediction outputs are separately documented |
| 7 | Complete code/provenance | PASS — original pretrained implementation, dependencies, workflow, source data/checkpoints, per-object CSV, SHA256 and independently committed verifier |
| 8 | Separation of measured/published | PASS — original local results not mistaken for official CrowdHuman/CityPersons leaderboard scores |
| 9 | Proper uncertainty and limitations | PASS — 5,000 photo-cluster replicates in both genuine CrowdHuman experiments; low heavy-group recall, confounding, proxy and sample-size limitations |
| 10 | Numerical/figure/table integrity | PASS — all 3,178 paired records audited; corrected three exact-0.20 boundary persons; no model matches altered; original group numerators verified, confidence intervals recomputed and PDF rebuilt with zero missing refs or overfull boxes |

## Exclusions and certification extent

No full CrowdHuman validation (4,370 images), official CrowdHuman miss rate/AP/Jaccard or CityPersons benchmark, specialized occlusion-aware head detector, causal backbone ablation, tracking or operational autonomous-vehicle assessment is claimed. The occlusion variable is the ratio of annotation rectangle areas rather than the visible-pixel fraction. Source sample sizes are expressly reported, and statistical intervals are conditional on those samples.

**Decision: APPROVED, the complete claim-scoped graduate textbook *application* within the above real-data objective.** This is an internally reviewed scientific unit certification and does not approve the other nine units or Chapter 5 overall.

**Evidence:** certificates 002/003/004, independent source-record audits, corrected run 42525235516, original 96-photo and complete Penn–Fudan runs, and rendered 122-page manuscript Pass 20.
