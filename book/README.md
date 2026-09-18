# Applied Machine Learning — Canonical Book Repository

This repository is the canonical source for the complete Springer-bound book.

## Required deliverables for every chapter

- manuscript/ — current LaTeX source, bibliography, and build metadata
- figures/ — all final accepted figures organized by manuscript section, plus provenance
- code/ — complete executable code for chapter examples, experiments, and figure generation
- instructor_solutions/ — complete instructor solutions for every end-of-chapter problem
- qa/ — compile, code, reproducibility, figure, and release audits

Historical code paths are retained for provenance. Each chapter RELEASE_MANIFEST.md is the canonical map from historical paths to the release layout.

## Book order

1. **Foundations of Machine Learning** — 32 production figures — chapters/foundations_machine_learning
2. **Deep Learning Part I** — 70 production figures — chapters/deep_learning/part_i
3. **Deep Learning Part II** — 38 production figures — chapters/deep_learning/part_ii
4. **Computer Vision Part I** — 14 production figures — chapters/computer_vision/part_i/foundations
5. **Computer Vision Part II** — 56 production figures — chapters/computer_vision/part_ii/autonomous_vehicles
6. **Natural Language Processing Part I** — 23 production figures — chapters/nlp_part_i
7. **Natural Language Processing Part II** — 26 production figures — chapters/nlp_part_ii
8. **Generative AI Part I** — 35 production figures — chapters/generative_ai_part_i
9. **Generative AI Part II** — 11 production figures — chapters/generative_ai_part_ii
10. **Reinforcement Learning** — 33 production figures — chapters/reinforcement_learning
11. **Scientific AI** — 54 production figures — chapters/scientific_ai
12. **Systems Engineering and MLOps** — 23 production figures — chapters/systems_engineering_and_mlops

A chapter is not COMPLETE merely because a figure or script exists. COMPLETE requires current manuscript source, reproducible PDF build, complete figures/provenance, complete code, complete instructor solutions, and passing QA.