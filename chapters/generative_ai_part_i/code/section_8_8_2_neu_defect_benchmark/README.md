# Section 8.8.2 — NEU surface defect benchmark (Pass 10)

**Status: dataset-ready local implementation; full NEU experiment NOT executed or approved.**

## Actual dataset
Northeastern University NEU Surface Defect Database, Song and Yan (2013), *Applied Surface Science*, DOI 10.1016/j.apsusc.2013.05.009. Six labeled classes: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches. The expected raw dataset contains 1,800 distinct grayscale photographs (200 × 200), 300/class. No normal/clean class is supplied, so the task is **six-class defect classification**, not binary anomaly detection.

## Planned evaluation — implemented and smoke-tested locally
- Reject incorrect image dimensions, missing classes, unexpected class counts, and exact duplicates.
- Class-wise numbered blocks of 30 adjacent images: six training blocks, two validation blocks, two test blocks per class, yielding 1080 / 360 / 360 real images.
- Train a class-conditional DCGAN on training data only. Save the generated samples and checkpoint.
- Compare identical CNN initialization trained on real images only versus real images plus GAN generated examples, selecting checkpoints using *validation* balanced accuracy.
- Evaluate only once on the held-out original test images; save per-image probabilities, macro-F1, balanced accuracy, confusion matrices, training traces, and figures.
- Maintain SHA-256 input manifest, code-version metadata, and restrictions on using test labels.

The complete `run.py`, `test_pipeline.py`, and `requirements.txt` have now been committed within this folder. The local execution and QA files are additionally packaged in **Chapter6_NEU_Benchmark_Pass10_Code_And_QA.zip**. A real-data benchmark run and independent CI verification are still required.

## QA
A deterministic **fabricated-noise fixture** was used solely for software testing; end-to-end smoke test PASSED. Its classifier accuracy is not evidence about NEU and MUST NOT appear in the chapter as benchmark data. Actual NEU images could not be fetched by the execution runtime. No real NEU model checkpoints, predictions, or scientific result figures have been produced.

## Dataset access
https://huggingface.co/datasets/dronefreak/NEU-DET/blob/main/README.md

## Publication gate
**HOLD** until the entire experiment is committed and rerun from GitHub using the genuine NEU images, its image and label provenance is verified, and the measured results are integrated into the case study. Also audit sampling quality with repeated seeds and independent fidelity/inspection metrics. The prior unlabeled brick/gravel/grass GAN experiment cannot be substituted for this real defect benchmark.
