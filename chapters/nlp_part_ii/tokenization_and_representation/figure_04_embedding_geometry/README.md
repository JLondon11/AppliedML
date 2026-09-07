# Figure 4 — Embedding Geometry in Contextual Representation Spaces

This directory contains the native-vector reconstruction for NLP Part II Figure 4.

## Scientific basis
The reconstruction uses real pretrained Stanford GloVe 6B 50-dimensional vectors trained on Wikipedia 2014 + Gigaword 5 (6B tokens). Panel (a) is a PCA projection of the pretrained word vectors. Panel (b) uses two context-conditioned composite representations of `bank`, formed by averaging the pretrained `bank` vector with financial-context or geographic-context vectors before projection through the same PCA model. Panel (c) shows centroid and displacement geometry in that same representation space.

No plotted coordinate is manually positioned. The panel-b composites are derived context-conditioned representations and are not claimed to be token-level BERT contextual embeddings.

## Files
- `figure_04_embedding_geometry.svg` — fully native SVG; no embedded raster image.
- `figure_04_embedding_geometry.py` — reproducible generation code.
- `figure_04_glove_subset_and_pca.csv` — exact pretrained vector subset and PCA coordinates.
- `figure_04_contextualized_bank.csv` — derived context-conditioned coordinates.
- `figure_04_embedding_geometry.png` — raster companion already stored in this directory.

Source model: Stanford GloVe, GloVe 6B pretrained vectors.
