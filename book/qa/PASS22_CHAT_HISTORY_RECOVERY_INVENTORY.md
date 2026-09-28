# Book Editing Chat-History Manuscript Recovery Inventory

Date: 2026-09-28

## Purpose

Record manuscript/source artifacts that are explicitly evidenced in the Book Editing project conversation history even though they are not present in the current GitHub repository.

This inventory corrects the narrower Git-only recovery conclusion: absence from GitHub does **not** mean the chapter source never existed in the Book Editing project.

## Confirmed chapter-manuscript artifacts in project history

### Foundations of Machine Learning

**Confirmed uploaded manuscript/source files:**

- `Foundations_of_Machine_Learning_InProgress.tex`
  - Appears in the “Systems Engineering and MLOps” Book Editing conversation during the table-provenance reconstruction work.
  - The conversation explicitly instructs a table-by-table provenance reconstruction and removal of noncompliant tables.

- `Foundations_of_Machine_Learning_Main_ProductionPass34_CIFAR100CaseStudy.tex`
  - Appears in the “Rewrite Continuation Plan” conversation.
  - The conversation explicitly says to continue rewrites, update the main `.tex` file, continue the production pass, and enforce quantitative-table rules.

**Recovery status:** CHAT-HISTORY SOURCE CONFIRMED. Latest-version selection still requires comparing the surviving project artifacts/conversation outputs.

### Deep Learning Part I

**Confirmed uploaded manuscript/source file:**

- `Deep_Learning_Part_I.tex`
  - Appears in the “Editorial Reconstruction Strategy” conversation.
  - The author explicitly states that all chapters must be 80+ pages and that a prior generated file had been reduced to scaffolding even though the complete chapter had already been written.
  - The author subsequently asks to review all Book Editing chats and recover the complete chapters.

**Recovery status:** CHAT-HISTORY SOURCE CONFIRMED. The uploaded file may represent a scaffold or intermediate state; the history also establishes that a fuller prior chapter existed in conversation.

## Confirmed frozen inventories / chapter-production source artifacts

These are not substitutes for the manuscript, but they are authoritative recovery inputs for figures/captions and chapter integration.

### NLP Part II
- `NLP_Part_II_Frozen_Figure_Inventory(1)(2).tex`

### Scientific AI
- `Scientific_AI_Frozen_Figure_Inventory_UPDATED.tex`
- `Scientific_AI_Frozen_Figure_Inventory_BALANCED_v3(1).tex`

### Generative AI Part I
- `GenAI_Part_I_Frozen_Figure_Inventory_v2.tex`
- Additional earlier uploaded inventory files are evidenced in the project history.

### Systems Engineering and MLOps
- A Systems Engineering and MLOps frozen figure inventory was explicitly generated in `.tex` form and requested for download.

## Other recovery packages evidenced in project history

### Reinforcement Learning
- `rl_flagship_package.zip`
  - Explicitly uploaded in the Reinforcement Learning chapter conversation.
  - Must be treated as a potential chapter-content/code/asset recovery source.

### Computer Vision
- A ZIP package `4450b990-c893-449c-a5eb-3980edfa2945.zip` appears in the Computer Vision production conversation after figure integration work.
  - Contents require inspection if the artifact can be recovered from project storage.

## Important project-history findings

The Book Editing conversations establish that:

1. Complete chapter prose was developed in chat before some later generated `.tex` files were reduced to scaffolding.
2. The author explicitly objected to scaffold-only outputs and requested reconstruction from prior Book Editing chats.
3. Multiple later production passes modified chapter content, tables, case studies, figures, citations, and code-listing policy after the initially uploaded/generated chapter files.
4. Therefore the correct recovery target is **not automatically the last named `.tex` attachment**. The target is the latest complete chapter state assembled from:
   - the strongest surviving manuscript artifact;
   - subsequent conversation edits;
   - frozen figure inventory;
   - Master QA changes;
   - later citation/reference additions;
   - later code-listing normalization;
   - later case-study/application rewrites.

## Current recovery disposition by chapter

| Chapter | Chat-history manuscript evidence | Recovery disposition |
|---|---|---|
| Foundations of Machine Learning | YES — at least two named `.tex` manuscripts | RECOVER / RECONCILE |
| Deep Learning Part I | YES — named `.tex` manuscript plus explicit evidence of fuller prior chapter | RECOVER / RECONSTRUCT |
| Deep Learning Part II | Conversation development confirmed; exact complete manuscript filename not visible in current project summary | SEARCH PROJECT HISTORY |
| Computer Vision Part I | Extensive chapter/figure integration history; exact complete manuscript filename not visible in current project summary | SEARCH PROJECT HISTORY |
| Computer Vision Part II | Extensive chapter/figure integration history; exact complete manuscript filename not visible in current project summary | SEARCH PROJECT HISTORY |
| NLP Part I | Extensive production history and citation-expansion instruction; exact complete manuscript filename not visible in current project summary | SEARCH PROJECT HISTORY |
| NLP Part II | Frozen inventory confirmed; manuscript development history confirmed | SEARCH PROJECT HISTORY |
| Generative AI Part I | Frozen inventory confirmed; extensive chapter production history | SEARCH PROJECT HISTORY |
| Generative AI Part II | Extensive chapter/figure production history | SEARCH PROJECT HISTORY |
| Reinforcement Learning | `rl_flagship_package.zip` confirmed; chapter development history confirmed | RECOVER PACKAGE / SEARCH PROJECT HISTORY |
| Scientific AI | Multiple frozen inventories and extensive production history | SEARCH PROJECT HISTORY |
| Systems Engineering and MLOps | Frozen inventory production confirmed; chapter/table reconstruction history confirmed | SEARCH PROJECT HISTORY |

## Corrected conclusion

The canonical manuscripts are **missing from GitHub**, but Book Editing project history contains direct evidence that manuscript source files and substantially complete chapter content existed.

The release-recovery process must therefore use the Book Editing project artifacts and conversation history as a source, then commit recovered canonical manuscripts to GitHub.

**Do not classify these chapters as lost or requiring from-scratch reconstruction solely because GitHub lacks the files.**
