# CV5-EXP-SAM-COCO-ORACLEBOX-001 — CERTIFIED (bounded experiment)

**Date:** 2026-11-08. **Experiment certification only; parent SAM application and Chapter 5 are NOT certified by this certificate.**

**Exact scientific claim:** Original pretrained Meta SAM ViT-B was actually executed on 120 annotated objects from 37 original COCO 2017 validation photographs, with oracle ground-truth box prompts. Local object mask IoU and Dice were evaluated against the corresponding original COCO instance annotations. A paired OpenCV GrabCut comparison used only the 78 objects for which a valid rectangle prompt could be constructed.

**Execution evidence:** https://github.com/JLondon11/AppliedML/actions/runs/42300131086

**Archived artifact SHA256:** `af2b64d93b5fe2ff3682c2fef4e72b42d02bd88d948d1fa9aec3457cbc80178a`

**Actual model checkpoint SHA256:** `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.

**COCO instance annotation source SHA256:** `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

**Independently verified results:** SAM mean object IoU `0.7822379222`, mean Dice `0.8605632305`; on 78 GrabCut-valid pairs, SAM IoU `0.8076488277`, GrabCut IoU `0.5622356850`, difference `0.2454131427`. Exactly 42 invalid baseline prompts were excluded transparently. Image-clustered 5,000-draw bootstrap 95% intervals: SAM mean IoU `[0.74302649, 0.81991773]`; paired delta `[0.16030957, 0.33059317]`. The input sample was seeded and capped at 120 eligible objects, not a representative sample of all COCO.

**Ten bounded-experiment criteria:** 1 PASS (prompted segmentation / COCO correspondence); 2 PASS (original annotated COCO val images, no local training, declared scope and sampling); 3 PASS (original checkpoint executed); 4 PASS (37 photographs/120 objects/78 valid pairs, prompt configuration and seed); 5 PASS (IoU and Dice with matching reference masks and paired GrabCut reference); 6 PASS (actual evaluated per-object distribution and comparison figures); 7 PASS (checkpoint/annotation/image hashes, per-object records and committed code); 8 PASS (measured results explicitly separated from literature and COCO AP); 9 PASS (5,000 image-cluster bootstrap, 42 prompt failures, substantial sample-selection qualifications); 10 PASS (independent arithmetic recomputation from all 120 rows, source TeX figures and discussion checked).

**Not certified:** official COCO segmentation AP; prompt-free localization or classification; generalization to the COCO population; superior SAM performance on failed GrabCut prompts; all models/prompts; the broader application/unit. Any change in dataset, checkpoint, prompt policy, metric or larger claim requires re-certification.

**Review method:** Independent numerical audit of archived per-object outputs and printed benchmark metadata, with scope review against latest Chapter 5 manuscript. This is an evidence-level scientific certificate, not third-party accreditation.
