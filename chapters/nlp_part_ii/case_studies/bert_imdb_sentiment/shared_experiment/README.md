# Shared IMDb experiment for Figures 8 and 9

Figures 8 and 9 are case-study figures and therefore share one controlled,
real-data experiment. The design uses the actual BERT architecture
`google/bert_uncased_L-2_H-128_A-2` to make CPU reproduction practical while
retaining genuine transformer fine-tuning.

Protocol:
- 2,000 reviews: training, official IMDb train split
- 500 disjoint reviews: validation, official IMDb train split
- 1,000 reviews: untouched evaluation, official IMDb test split
- deterministic seed 1729
- three fine-tuning epochs
- best checkpoint chosen exclusively by validation loss

Figure 9 is generated from recorded training/validation losses. Figure 8 is
generated from predictions on the untouched test sample using that selected
checkpoint. Thus the two figures have one internally consistent provenance chain.
