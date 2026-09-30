# AppliedML Final Publication Recovery Rules

Recovery baseline: `AppliedML_Complete_Book_Proof_Pass20.pdf`.
Superseding editorial requirements are taken from the Book Editing project history and later accepted figure packages.

## Manuscript hard gates
- Reconstruct editable LaTeX chapter sources; the final deliverable may not be a PDF-inclusion wrapper.
- Audit every chapter line by line for incomplete sentences, fragments, duplicated prose, broken cross-references, malformed headings, and extraction artifacts.
- Remove all editorial/production narrative, including discussion of original code/chapter drafts, repair history, placeholders, production passes, and instructions to future editors.
- Remove irrelevant Tips/Notes/tooltips about original code, missing code, prior chapter versions, or editorial process.
- Application and Case Study sections begin with multiple substantive introductory/background paragraphs. Only afterward use bold standalone labels such as **Problem**, **Dataset**, **Model**, **Method**, **Evaluation**, **Results**, **Error Analysis**, **Limitations**, and **Deployment/Research Implications**, each followed by full prose.
- Every retained Application and every retained Case Study must include at least one scientifically appropriate, nonredundant figure and at least one evidence-based benchmark/comparison table tied directly to that section.
- The figure/table requirement must never be satisfied with decorative artwork, generic diagrams, duplicated chapter-level assets, invented numbers, or irrelevant leaderboards.
- Figures and tables may be based on cited published evidence or reproducible executable analysis using the stated dataset/protocol.
- If an Application or Case Study cannot support a scientifically meaningful figure and comparison table without repetition or unsupported evidence, consolidate it with a stronger neighboring section or remove it.
- Quantitative claims/tables require real provenance or reproduced experiments. Remove unsupported illustrative numerical claims.
- Ensure current/SOTA topics required by the later project history are present and integrated without duplicating specialized later chapters.

## Code hard gates
- 15-22 substantive code listings per chapter (never exceed 25 without explicit approval).
- Every retained listing must be titled, numbered, referenced, and discussed in the surrounding prose.
- Remove repetitive, trivial, repair-oriented, or non-informative code dumps.
- Retained code must be internally complete enough to support the stated pedagogical point; full implementations may live in GitHub.
- Remove references/tooltips to code that does not exist or does not run.

## Figure hard gates
- Audit every canonical figure individually for scientific correctness, provenance, caption fidelity, and aesthetics.
- Scientifically correct but visually weak figures remain REVISE, not ACCEPT.
- Use real data/executable computation for empirical, benchmark, application, and case-study claims.
- No invented benchmark values or infographic substitutes for scientific renderings.
- White scientific background where appropriate; premium restrained palette; compact whitespace; consistent typography/line weights; clean legends/colorbars.
- Every heatmap/matrix/saliency map that encodes magnitude must include a quantitative legend/colorbar.
- Every panel `(a)`, `(b)`, `(c)`, etc. must be explicitly described in the unified caption and referenced where appropriate in the prose.
- Detect and remove/replace exact or semantic duplicate figures. If a figure is removed, remove or rewrite every corresponding in-text reference.
- Final surviving figure binaries must be archived in PNG, SVG, and PDF and committed to GitHub.

## Known authoritative later corrections
- NLP II Figures 4, 6, 7, 8, 9, 12 and Scientific AI Figure 14 use the later Pass 16 scientific remediations as the scientific baseline; aesthetic refinements must not change validated computations/results.
- Deep Learning I duplicate/overlap groups: 2.16/2.17, 2.41/2.42, 2.44/2.45 require scientifically distinct roles.
- Computer Vision II 5.37/5.38 require distinct scientific roles rather than duplicate architecture diagrams.
- NLP II 7.13/7.15 are distinct and should not be removed merely as visual duplicates.
- Chapter 1 page-33 tip about trying L-BFGS is removed as nonessential editorial/tip material.

## Persistence
- All accepted text and binary changes are committed to branch `book-publication-recovery-final`.
- Local deliverables carry SHA-256 checksums and a chapter/figure manifest.
