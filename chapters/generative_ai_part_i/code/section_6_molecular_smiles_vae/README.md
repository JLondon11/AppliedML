# Chapter 6 — NCI SMILES sequence VAE experiment (Pass 20)

**Status:** Completed small real-data molecular-string generation diagnostic; scientific case study remains HOLD.

A character-level GRU variational autoencoder was trained on 600 **unique RDKit-sanitized NCI** SMILES from RDKit's bundled `Data/NCI/first_5K.smi`. This is a non-commercialized scientific demonstration; it does not establish molecular target activity, synthesizability or drug candidacy. The RDKit data source SHA-256 is `91e71c015f14939837f2943dcc904f7c87e5a3a0124d82b05c28ad2f23004def`.

Partition (seed 20261109): 360 train, 120 validation, 120 test. Six epochs, 32-dimensional Gaussian latent, bidirectional GRU encoder / autoregressive GRU decoder, validation-token-loss checkpoint. Fixed first 600 canonical unique structures after filtering SMILES length >75 (NOT a representative random sample of the full NCI database).

Measured results:
- Test decoder structural validity 57.5%; exact held-out reconstruction 0%; mean Morgan Tanimoto 0.05407.
- 1,024 unconditional prior samples, 36 RDKit-valid (3.5156%).
- 23 distinct valid canonical SMILES absent from the **training** subset; 63.89% uniqueness among the 36 valid strings; no pharmacological validation.
- Training still improved at epoch 6, so **not converged**.

**Essential limitations:** Tiny subset, one seed, no scaffold-separated test, low string validity, no property conditioning, no biological endpoint or wet lab. Novelty relative to 360 training structures does not imply novelty in global chemistry corpora.

The runnable code and full output evidence (source manifest, per-test reconstructions, all 1,024 prior samples, checkpoint, training trace and two generated figures) are contained in `Chapter6_Generative_AI_PartI_Pass20.zip`. This repository directory is the intended canonical section-specific location. The exact script committed to GitHub must be independently executed on a fresh checkout before the code itself can be marked fully verified.

This experiment is relevant to the manuscript's **Case Study: Generative AI for Drug Discovery and Molecular Design**, which otherwise remains uncertified.
