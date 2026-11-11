# Chapter 6 BDD100K multi-scene generative augmentation — Pass 42

STATUS: HOLD. No BDD100K multi-scene images or matching official 2018 detection JSON labels were present in the execution environment. No BDD training, generated outputs, or held-out classification metrics were produced.

Sources: https://github.com/bdd100k/bdd100k and https://dl.cv.ethz.ch/bdd100k/data/ . The official 100K image archive is approximately 5.3 GB, with labels a separate download.

The dataset-ready Python generator, genuine-source validator, downstream road-object crop classifier comparison and README are included in the Pass 42 validation ZIP. Python syntax and refusal of missing genuine data were tested. It compares real-only, resampled real and conditional VAE-augmented road-object crop classification, not whole-scene detection mAP. This repository README is a status pointer; the full runner has not yet been committed or independently executed with a clean checkout.

Next evidence gates: obtain licensed frames/labels, verify splits and provenance, train the models, inspect actual generated crops, save real held-out frame predictions and recalculate downstream scores. Whole-scene synthetic driving requires added scene synthesis and genuine held-out detection mAP.
