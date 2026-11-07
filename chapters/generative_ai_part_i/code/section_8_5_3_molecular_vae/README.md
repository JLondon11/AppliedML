# Section 8.5.3 — Molecular VAE experiment (NCI)

## Verified scope
Implemented and executed a beta-weighted variational autoencoder (VAE) trained to reconstruct 256-bit Morgan fingerprints from 200 genuine NCI molecules distributed with RDKit (`RDKit/Data/NCI/first_200.props.sdf`). The reconstruction is a binary molecular descriptor, **not** a valid molecule or SMILES generator. Do not describe these results as molecular discovery, synthetic molecular validity, drug activity, or de novo molecular synthesis.

## Run
Install `rdkit`, `torch`, `numpy`, `scikit-learn`, `matplotlib` and execute:
```bash
python chapters/generative_ai_part_i/code/section_8_5_3_molecular_vae/run.py
```
The script uses a deterministic seed, 120/40/40 training/validation/test molecule partition, 12-dimensional VAE latent representation, beta=0.03, Adam learning rate 0.001, and validation-objective checkpoint selection. It produces `results/dataset_manifest.csv`, `results/metrics.json`, `results/heldout_predictions.csv`, `results/training_trace.csv`, `results/nci_fingerprint_vae.pt` and three quantitative figure PNGs. The NCI SDF is sourced from the installed RDKit package; it is not redistributed.

## Executed results (40 held-out molecules)
| Method | Mean fingerprint Tanimoto |
| --- | ---: |
| VAE (latent 12; binary reconstruction) | 0.24138 |
| PCA (12 components, train-only fit) | 0.40973 |
| Nearest training molecule (retrieval reference) | 0.44631 |

VAE underperforms the PCA baseline. This negative finding must be retained. Random molecule splits can overstate generalization relative to scaffold splits. A 200-molecule convenience dataset is insufficient for broad claims. This experiment **does not close** all requirements for the chapter's molecular drug-design case study.

## Artifacts
The separately packaged local execution evidence contains the full test-level predictions, model checkpoint, trace, and generated figures. The repository commit includes executable source and a metric summary; outputs can be regenerated in `results/`.

## Provenance
RDKit packaged NCI molecular structures, SHA-256 `c3eef33eec2c9676a54bbcec6dd1b91a099df9b0d0c8a1b60f5178767e4a3e13`. Seed: 20261106. RDKit 2025.09.4, PyTorch 2.10.0+cpu. This dataset is NCI molecular structures, **not** experimentally measured biological activity.
