# Chapter 6 — scientifically accurate captions for five staged figures

**Status:** Editorial candidates only; **not inserted** into current manuscript, whose PDF and TeX are absent. Do not assign guessed Figure 6.xx numbers or page positions. Use exact trained-model results and source identity below.

## Authentic microscopy input and spatial split
*Original immunohistochemistry source photograph with spatial train, validation and test source-row regions and representative observed 32×32-pixel crops. Training occupies image rows 0–319, validation rows 320–383 and testing rows 384–511. The 589/93/217 derived patches all originate from one 512×512 specimen photograph; spatial separation prevents identical pixels crossing these partitions but does not establish external tissue, patient or clinical generalization. The displayed crops are original observed inputs, not VAE reconstructions.*

Source `figure_generation/original_source_image_plates/make_authentic_source_image_plates.py`; stage file `original_tissue_source_and_patches.pdf`.

## Authentic handwritten digit source observations
*Twenty original scanned handwritten-digit observations from `sklearn.datasets.load_digits()`, with two observations per class 0–9 and each native scan represented at its true 8×8-pixel resolution. These are original source observations used to contextualize the class-conditioned latent diffusion experiment; they must not be mistaken for samples produced by the trained diffusion generator.*

Source `figure_generation/original_source_image_plates/make_authentic_source_image_plates.py`; stage file `original_scanned_digits_gallery.pdf`.

## Actual tissue VAE reconstruction and model-generated prior samples
*Eight authentic spatially held-out microscopy patches, the corresponding outputs of a 12-epoch executed 24-dimensional convolutional variational autoencoder, absolute RGB-pixel error maps and separately sampled latent-prior outputs. The measured test-set averages across all 217 spatially held-out patches of this same source photograph are PSNR 20.155 dB, SSIM 0.2927 and reconstruction pixel MSE 0.010481. The blurred reconstructions and feature-poor generated samples indicate important modeling limitations. Neither this within-photo test nor the generated patches constitute independent clinical or patient-level evidence.*

Source `code/section_6_medical_imaging_histology_vae/plot_tissue_original_model_outputs.py`; stage file `actual_tissue_cvae_reconstruction_gallery.pdf`.

## Authentic original NCI source chemistry
*Twelve chemical graph structures rendered with RDKit from the original NCI `first_5K.smi` source, with four source compounds drawn from each of the 360/120/120 original training, validation and test partitions. Chemical bond diagrams and molecular formulas are computed from the real original input SMILES strings, not newly generated or experimentally assayed chemicals. The figure establishes input-source chemical diversity only, not the validity, utility or biological activity of the sequence VAE's outputs.*

Source `code/section_6_molecular_smiles_vae/make_original_nci_structure_plate.py`; stage file `original_nci_chemical_structures.pdf`.

## Scientific architecture of the actually implemented molecular VAE
*Source-code-aligned character-level SMILES variational autoencoder used for the NCI study. A 32-feature character embedding feeds a bidirectional GRU with 128 hidden units per direction; separate linear heads predict a 32-dimensional latent Gaussian mean and log variance. Reparameterized latent vectors condition each step of a 128-hidden-unit autoregressive GRU decoder. Training combines non-padding next-token cross-entropy and scheduled KL regularization. Distinct evaluation modes are greedy reconstruction of 120 original held-out molecular strings and stochastic decoding of 1,024 latent-prior strings, assessed by RDKit structural validity and related diagnostic metrics. The schematic shows model design, not observed chemical efficacy or newly performed laboratory experiments.*

Source `code/section_6_molecular_smiles_vae/plot_nci_smiles_vae_code_architecture.py`; stage file `nci_molecular_vae_code_architecture.pdf`.

## Production integration controls

For each prospective manuscript insertion, verify the exact current section/title, place the figure near the technical explanation and before end-of-chapter problems, ensure captions cite authentic data identity and distinguish source/architecture/model outputs, use vector PDF when legible, and validate final rendered PDF page for no clipping. Do not add an unassigned figure labeled Figure 6.20, which is reserved for the unresolved industrial-defect figure evidence mapping. These production tasks **require the current full manuscript** and cannot be marked complete using staged graphics alone.
