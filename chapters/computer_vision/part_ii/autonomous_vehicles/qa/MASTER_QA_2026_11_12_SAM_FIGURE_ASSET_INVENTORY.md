# Chapter 5 — original SAM/Mask R-CNN complete figure asset audit

**Date:** 2026-11-12. **Status:** figure archive inventory verified; final manuscript/R9 remains HOLD.

## Executed original experiment asset audit

The three original GitHub Actions ZIPs from the recorded SAM and Mask R-CNN genuine COCO experiments were downloaded and their independently recorded per-instance metrics rechecked. The original 42 COCO scenes, 147 SAM masks, 99 valid paired SAM–GrabCut objects and same-scene 59 annotation IDs/56 matched Mask R-CNN predictions reconcile with previously published local experimental summaries.

A complete original image asset audit script was run on the three ZIPs. It identified **24 authentic original-output PNG figures**: **9** in the SAM 18-scene run, **7** in the SAM 24-scene run and **8** in the matched Mask R-CNN 18-scene run. All 24 original PNG figures have corresponding original SVG and PDF files with the same basename. All PNG members were read, verified and individually hashed (SHA256). The minimum dimensions across their axes were 1,200 pixels wide and 400 pixels high. The complete per-asset CSV and summary JSON are supplied in the user artifact package; the independently committed verifier is `qa/audit_original_sam_figure_assets.py`.

All 24 source PNGs were visually surveyed in a combined contact sheet. The model output panels show original COCO photos and relevant predicted/GT mask overlays. The two SAM–GrabCut scatter figures show paired instance IoU and the equality reference. Several panel annotation labels are small at reduced viewing size; final print-size legibility and explicit panel semantics need inspection at actual manuscript width. This contact sheet is **not** independent pixelwise ground-truth mask verification, nor is it a full printed-page review.

Original model archive SHA256 values: 18-scene SAM `b603229a59fee731eb0e4c794e34b0d776c23492a64e7cf8d6b69db16e3b3f6d`; 24-scene SAM `3a4304d56d8d7ad2eebf880069f2fc35afc79f5c78679eb94a2c36a192ed5579`; Mask R-CNN `85ceb81911ff0f69975d1177f928d98be0e82c67f17d3df361fbde2bb92beacb`.

## Missing actual manuscript

An exact recursive `main` GitHub tree listing (983 objects; not truncated) contained no Chapter 5 `.tex` or `.pdf` manuscript. The current local workspace also lacks a Chapter 5 PDF and TeX. A historical QA note reports a delivered 108-page release; its existence as a **past reported release** cannot certify the current manuscript version or printed figure page layout.

Therefore current figure numbers/captions, in-text discussion, actual final page placement and publication-scale legibility cannot be independently checked. Original workflow ZIPs have a reported retention expiry of **2027-02-03 UTC**, so permanent archive retrieval still needs assurance.

## Certification decision

The original scientific figure-asset **inventory and per-file integrity verification are PASS**. However, full R9 includes original model code/config/checkpoint correspondence, durable storage and final manuscript figure links. The current R9 and Chapter 5 decisions remain **PARTIAL / HOLD**; SAM remains 9 of 10 criteria provisionally passed under the restricted Level B offline oracle-box-prompt experimental definition. No new model inference or generalization claim was introduced.

## Required final closure

Locate the latest Chapter 5 TeX, source figure assets and compiled PDF. Cross-index each final SAM figure number to its original zip/member SHA256, per-object output source, plotting code, original run and caption. Review every resulting PDF page at physical print scale; fix unreadable multi-panel annotation labels and any displaced legends without altering measured results. Preserve authorized original outputs durably before the run archives expire. Reinspect the evidence prior to any R9 PASS or certification.
