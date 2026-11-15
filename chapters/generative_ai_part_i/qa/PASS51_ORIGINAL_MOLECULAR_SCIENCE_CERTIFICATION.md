# Chapter 6 — Pass 51: original NCI SMILES and independently recalculated measured ESOL certification

**Scientific review date:** 2026-11-14. **Governing policy:** original-data requirement R1–R10 applied proportionally to explicit, honestly limited experiments. **Pass 51 result: two more narrowly scoped Level A scientific certifications (D09 and D10), bringing the current explicit experimental register to 10 certificates (9 Level A and 1 Level B). This does not mean 10 of the original 14 named parent applications are certified.** All final Chapter 6 manuscript production approvals remain HOLD until actual PDF/TeX integration is checked.

## CH6-D09 — original NCI molecular SMILES sequence-VAE validity diagnostic

**Scientific certification: Level A SCIENTIFIC_CERTIFIED, only for the named 600-original-record string-model diagnostic.** The underlying original-data generative training and source-output analysis were previously performed in Pass20 and independently examined in Pass47. This pass did **not** claim a new NCI model training run.

- **R1:** RDKit bundled NCI original `first_5K.smi` with recorded source SHA256 `91e71c015f14939837f2943dcc904f7c87e5a3a0124d82b05c28ad2f23004def`; 600 genuine uniquely canonicalized RDKit-valid original structures.
- **R2:** disjoint original 360 training, 120 validation, 120 test molecular-string records, source IDs and split manifest. Not scaffold-independent or population-representative.
- **R3–R4:** original trained character-level bidirectional-encoder/GRU-decoder VAE, six epochs; original 120 measured reference held-out reconstructions and 1,024 actually generated prior sequences, with original model checkpoint/trace. The output strings are **model-generated mathematical sequences**, not laboratory measured products.
- **R5–R6:** actual annotated known-chemistry reference strings, exact-match, RDKit syntax/structure validity, Morgan/Tanimoto similarity and prior diversity. Test reconstruction 69/120 syntactically valid (57.5%), exact held-out reconstruction 0/120, mean Morgan Tanimoto about 0.05407. Of 1,024 unconditional prior-decoded strings, **36 RDKit-valid (3.5156%)**; only **23 distinct valid canonical molecules**. Model had not clearly converged by last epoch.
- **R7:** valid for offline *molecular string generation diagnostic* only; not new target activity, potency, chemical safety, synthetic accessibility or lab discovery.
- **R8–R9:** original experiment retained two scientific output figures and original per-observation records in its earlier archive, verified in Pass47; complete model runner, checker and dataset source traceability are committed in `code/section_6_molecular_smiles_vae/`. Pass51 adds `plot_original_smiles_diagnostic.py`, a source-record-gated publication-figure generator using original per-test Tanimoto error ECDF and original prior validity to replace the prior trivial multiple-bar composition. **The new replacement script is committed but its new plot has not been verified as executed in this pass**; the original figure evidence supports the limited scientific diagnostic.
- **R10:** a negative-result demonstration of poor model validity and reconstruction, **not** a conclusion that any new small molecule was experimentally discovered, effective or safe.

**Remediation:** explicit README scope and reproducible new plot source. New CI source reruns original VAE on its genuine bundled NCI dataset and checks actual 120/1024 output records, checkpoint and figures; **the new remote workflow's outcome must be checked separately**.

## CH6-D10 — independently verified retrospective ESOL measured-assay ranking

**Scientific certification: Level A SCIENTIFIC_CERTIFIED, retrospective ranking of known assayed compounds only.** This pass directly retrieved the *original publicly accessible source* `delaney-processed.csv` from immutable DeepChem Git commit `108af2176a5f876c2df1df922baadb02d647c826`, and independently recalculated the previous `reproduce_esol.py` protocol from the source, using the identical deterministic xorshift32 Fisher–Yates split, training-set feature standardization and 5-descriptor ridge regression with ridge penalty 10.

- **R1:** original 1,128 distinct measured Delaney/ESOL solubility records, real experimental labels. The separate published original ESOL score column is **precomputed by the dataset provider**, not a new model trained in the book.
- **R2:** deterministic 902 measured-observation training / **226 original held-out observation test** records. Ridge coefficients and high-solubility threshold computed on training only; threshold `-1.6` log mol/L equals training 75th percentile.
- **R3–R4:** actual locally re-executed ridge evaluation from source records, 226 retained real-compound test predictions and measured original test label/SMILES pairs committed in `retrospective_esol/verified_original_ESOL_226_test_predictions.csv`. The original source code is `reproduce_esol.py`.
- **R5:** two quantitative meaningful references: published provider ESOL predictions and random-selection top-20 expected hits. The provider column is correctly described as **published, not newly executed**.
- **R6:** observed ridge test **RMSE = 1.1795769332543176** log solubility units, provider published ESOL **RMSE = 0.8755685030537721**; ridge **14** high-solubility measured hits among its top 20; published score **17** of top 20; random selection expected **5.221238938053097** hits out of 20. These values were independently recalculated from the source and matched the original report.
- **R7:** appropriate retrospective validation against genuinely already-measured aqueous solubility; **no prospective chemistry/synthesis or robot experimentation** is implied.
- **R8–R9:** original Pass25 figure and code evidence is supplemented by a source-locked generator `plot_verified_original_assays.py` that plots all **226 measured solubilities versus both prediction sources**, with real original top-20 highlighted, equality reference and explicit limitations. Source-original 226-row CSV and numerical summary committed. New remote CI can regenerate PDF/PNG/SVG from this exact, previously independently verified data. **Do not mark the new figure visually reviewed until it is rendered and checked.**
- **R10:** classification is an offline retrospective molecular ranking *exercise*, not the parent chapter's broad autonomous-scientific-discovery or new-compound wet-lab claim.

## Verification and execution boundaries

The newer source files were committed to GitHub and original source/experiment matching was performed on the 226 source compounds. No 2026-11-14 original NCI model rerun, new generated chemical validation, or independent chemical assay was performed or claimed. The original NCI prior validity (3.52%) and lack of molecular reconstruction are **reported scientific failures**, not reasons to block an honest Level A *diagnostic* certificate.

The current production manuscript remains the previously reported Pass45 109-page edition; the user-supplied `Proceed Chapter 6.txt` is a **Pass37 figure-placement handoff** describing the earlier Figure 6.20 on printed page 67, not a present PDF asset we can inspect. The old placement note does not establish current printed-figure approval. The chapter's historical 14 named case-study/application registry remains separately subject to unresolved physical concrete, BDD100K actual downstream driving, domain generalization and other title-scope evidence. **No parent application is silently elevated by these two nested certifications**.

## Current status

- **10 specific experimental units scientifically certified:** D01–D10 (9 Level A; 1 Level B).
- **14 broader named parent applications:** original Master QA HOLD status remains unless individually reassessed and promoted based on truly matching evidence.
- **Production approval:** HOLD for all current experiments pending final chapter source/PDF figure integration.
- **New GitHub Actions workflow:** `.github/workflows/chapter6_real_molecular_evidence.yml` committed; remote execution has not yet been independently confirmed.
