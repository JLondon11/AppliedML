# Graduate-level experimental certification — adopted 8 November 2026

## Two separate certification levels

**Experiment certificate:** certifies the precise scientific claim supported by an executed, original-data (or expressly justified physics-simulation) experiment. Approval applies only to the declared task, sample, model, configuration and metrics.

**Case-study/application unit certificate:** certifies the complete section only if all experimentally testable claims and necessary comparisons in its title, text and figures are supported by appropriate experiment certificates; all required text, figure and editorial QA also passes.

Chapter publication certification adds the full scientific, narrative and production Master QA. **A bounded certified experiment never automatically certifies its parent unit.**

## Ten retained scientific criteria

1. Task/model/dataset match the stated, bounded scientific question.
2. Authentic original data and independent evaluation, or a physically validated simulation only when its scientific question truly requires one. Check and disclose any training leakage.
3. Genuine execution of the named model, not a restatement of published predictions.
4. Applicable training/validation/test counts and settings, checkpoints, transformations, environment and seeds. For frozen pretrained inference, document the actual evaluated count and state no local training occurred.
5. Appropriate calculated task metrics compared to a scientifically meaningful reference/baseline. Official competition metrics are required only for official/overall challenge claims.
6. Scientific code-generated model outputs, errors or measurements integrated and discussed in the manuscript.
7. Complete executable code, dataset provenance/license, checkpoint and data hashes where possible, results archive, code committed under chapter and section.
8. Explicit separation of measured results from literature and published benchmarks.
9. Appropriate uncertainty/failure/limitations analysis. Repeated training seeds are expected when comparing variable trained models, not universally for frozen deterministic evaluation.
10. Independent numerical consistency verification and editorial/figure/table/caption coherence, including rendered-manuscript visual QA at unit level.

Each criterion is recorded as PASS, PENDING or FAIL. A detailed scientific reason may establish that a subcomponent such as train-set count or repeated training runs is **not applicable**, but missing numerical evidence, inputs, execution or figures cannot be waived. All ten criteria must pass before issuing an experiment certificate.

## Different expected depth

**Applications** demonstrate a substantive technique on a fitting engineering problem with a good measured comparison, a real output figure and scientifically explicit limits. They do not always need a complete official benchmark.

**Case studies** demand a deeper and broader engineering investigation with additional meaningful comparisons, tradeoffs, failure analysis and stronger uncertainty analysis, proportionate to the claims.

## Unacceptable scope substitutions

* Original Penn-Fudan detection evaluation cannot certify a named CrowdHuman occlusion application.
* Oracle-box SAM segmentation does not prove unprompted segmentation or COCO AP.
* One KITTI LiDAR scene establishes PointPillars execution, not full KITTI AP.
* A TuSimple mask IoU experiment is not official TuSimple lane-point accuracy and must disclose unknown checkpoint train/test overlap.
* A physical phantom is not original fastMRI multicoil reconstruction, and stereo disparity is not NeRF or 3DGS.

## Approval process

Record immutable result artifact hash, Actions run, input data, checkpoint identity/hash, exact evaluated count, local metric definition, sensitivity/failures, and corresponding manuscript figure. Independently reconcile per-object/per-frame measurements. Sign bounded experiment certificate only when all ten gates pass. Unit certificate requires an additional explicit mapping of every unit claim to certified experiments or an otherwise verified scientific source.

**This revises the interpretation of the ten criteria, not their rigor.**