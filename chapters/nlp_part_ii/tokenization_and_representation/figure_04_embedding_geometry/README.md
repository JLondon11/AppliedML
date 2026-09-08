# Figure 4 — Embedding Geometry in Contextual Representation Spaces

**Chapter:** NLP Part II  
**Section:** Tokenization and Representation Learning  
**Frozen label:** `fig:embedding_geometry`

This directory contains the certified, reproducible Figure 4 production bundle.

The figure is generated from **actual last-layer contextual token vectors** from
Google BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`; 2 layers, hidden size 128).
The generation script downloads the pretrained model, extracts the contextual
hidden state for each target token occurrence, averages WordPiece fragments when
necessary, fits a two-component PCA over the complete extracted vector set, and
renders the three scientific panels.

- **(a)** contextual semantic neighborhoods for animal, vehicle, financial, and
  geographic target words;
- **(b)** the same lexical token `bank` in five financial and five river/geographic
  contexts, with empirical context centroids and their displacement;
- **(c)** relational geometry among actual contextual centroids in the same PCA
  representation space.

No point is manually positioned. No synthetic embedding vectors are used.

## Provenance outputs

- `figure_04_bert_tiny_contextual_vectors.csv`: every extracted 128-dimensional
  contextual vector plus its exact sentence, WordPiece token(s), and PCA coordinates.
- `figure_04_pca_components.csv`: the two PCA component vectors and explained variance.
- `figure_04_model_provenance.json`: model ID, resolved Hugging Face revision,
  hidden size, context count, PCA variance, and PyTorch version.
- `figure_04_embedding_geometry.py`: complete executable generation code.
- `figure_04_embedding_geometry.svg`: native vector artwork.
- `figure_04_embedding_geometry.png`: 300-DPI raster companion.

Model source: Google BERT Miniatures, described by Turc et al. (2019),
*Well-Read Students Learn Better: On the Importance of Pre-training Compact Models*.
