# Chapter 6 — Pass 55: exact-head evidence manifest checker correction

## Actual failure mode found
The previously committed `figure_generation/verify_source_provenance.py` read only `manifest["files"]`, whereas the existing, committed exact-head source replay manifest `figure_generation/vae_pca_results/exact_head_2026_11_13/artifact_manifest.json` records artifact checksums under `files_sha256`, plus `source_git_blob_sha1`. Passing that genuine manifest to the previous checker always returned an empty-manifest failure rather than checking the evidence.

## Committed changes
- `figure_generation/verify_source_provenance.py`: now accepts both the legacy `files` schema and the exact-head `files_sha256` schema, but rejects a manifest that contains both. For the exact-head package, `--outputs` points to the package root containing both `vae_pca_real_digits.py` and `output/`. The source file is checked separately against the supplied Git blob ID **and** its manifest SHA-256. Checks all listed generated artifacts against SHA-256. Rejects absolute and parent-traversal artifact paths.
- `figure_generation/test_verify_source_provenance.py`: eight deterministic regression tests for each schema; source change; output change; traversal; missing source SHA-256; conflicting schemas; and manifest/source Git blob mismatch.

## Exact-head evidence values retained
- Git blob SHA-1 for the committed source: `666cac09a9ec5b127f2ba7d89f109c6b10065df4`.
- Recorded held-out data: authentic scikit-learn handwritten digits, 1,257 train / 270 validation / 270 test.
- Test VAE reconstruction MSE: 0.05387162044644356, PCA baseline 0.05312391743063927.
- The saved exact-head run is **numerically** consistent with earlier archived results, but its output files are not all byte-identical to an earlier archive. The changed checker does not claim otherwise.

## Verification status
The revised checker and regression tests are committed to GitHub. They were source-inspected, **not independently executed within this pass**. They do not certify the current Chapter 6 manuscript, Figure 6.20, a synthetic-driving benchmark, or prospective physical testing of generated concrete. Those production and scientific claim gates remain HOLD.

## Invocation
```bash
python -m unittest discover -s chapters/generative_ai_part_i/figure_generation -p 'test_verify_source_provenance.py'
python chapters/generative_ai_part_i/figure_generation/verify_source_provenance.py \
  --source EVIDENCE_PACKAGE_ROOT/vae_pca_real_digits.py \
  --expected-git-blob 666cac09a9ec5b127f2ba7d89f109c6b10065df4 \
  --outputs EVIDENCE_PACKAGE_ROOT \
  --manifest chapters/generative_ai_part_i/figure_generation/vae_pca_results/exact_head_2026_11_13/artifact_manifest.json \
  --report provenance_check.json
```
