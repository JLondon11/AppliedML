# Reproducibility Repository Structure

All book artifacts are organized by chapter and section.

```text
<chapter>/
  <section>/
    figures/
      <figure_id>/
        README.md
        src/
        data/
        outputs/
    applications/
      <application_id>/
        README.md
        src/
        configs/
        data/
        outputs/
    case_studies/
      <case_study_id>/
        README.md
        src/
        configs/
        data/
        outputs/
```

## Hard rules

1. Every generated empirical figure must be committed with its complete executable generation code.
2. Quantitative figures must include the exact numerical source data used to render them.
3. Application and Case Study figures must come from authentic public data, actual executable model outputs, recorded experiment logs, or provenance-matched published results.
4. Each experiment must record dataset/version, preprocessing, model/checkpoint, configuration, random seed where applicable, software dependencies, and hardware when performance or timing is reported.
5. Publication figure assets are committed under `outputs/`; source code and data live beside them.
6. No synthetic illustration may be labeled as a real dataset sample, model output, or measured benchmark result.
7. Every README must contain exact reproduction commands and provenance.
