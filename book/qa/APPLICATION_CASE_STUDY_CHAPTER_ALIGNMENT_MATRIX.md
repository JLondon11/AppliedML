# Book-Wide Application and Case Study Concept-Alignment Matrix

This is a publication gate for every retained Application and Case Study. Each section must explicitly apply concepts taught in the chapter where it appears. Domain context alone is insufficient.

| Chapter | Required concept spine for Applications / Case Studies |
|---|---|
| 1 Foundations of Machine Learning | problem formulation, features/representation, optimization, loss/objective design, data quality, leakage, imbalance, validation, regularization, generalization, HPO, reproducibility, baseline model selection, deployment/governance fundamentals |
| 2 Deep Learning Part I | multilayer perceptrons, backpropagation, initialization, normalization, optimizers, CNN fundamentals, representation learning, regularization, training dynamics, transfer learning foundations |
| 3 Deep Learning Part II | attention, transformers, sequence modeling, state-space/selective-state-space models, Mamba-style architectures, attention/SSM hybrids, scaling/efficiency, long-context modeling |
| 4 Computer Vision Part I | image formation/representation, convolution, features, classification, detection/segmentation foundations, augmentation, transfer learning, vision evaluation |
| 5 Computer Vision Part II | advanced detection/segmentation/tracking, multimodal/perception systems, 3D/geometry, autonomous perception, deployment and robustness in vision systems |
| 6 NLP Part I | text preprocessing, tokenization, embeddings, sequence models, language modeling foundations, classification/tagging, evaluation and data issues |
| 7 NLP Part II | transformers/LLMs for language, retrieval, RAG, fine-tuning, interpretability, scaling, long-context behavior, factuality/hallucination, advanced NLP evaluation |
| 8 Generative AI Part I | latent-variable models, GANs, autoregressive models, diffusion, flow matching/rectified flow, likelihood/generative objectives, evaluation and sampling |
| 9 Generative AI Part II | instruction tuning, preference optimization/DPO, reasoning-model post-training, multimodal generation, alignment/evaluation, tool use and advanced generative systems |
| 10 Reinforcement Learning | MDPs, value functions, policy gradients, actor-critic, exploration, model-based/model-free RL, offline RL, multi-agent/safe RL, evaluation under sequential decision making |
| 11 Scientific AI | physics-informed ML, neural operators, symmetry/equivariance, scientific inverse problems, surrogate modeling, uncertainty, domain-specific scientific validation and real-data provenance |
| 12 Systems Engineering and MLOps | data/versioning, pipelines, feature stores, serving, CI/CD, monitoring, drift, observability, distributed training, reliability, governance, cost/latency/resource tradeoffs |

## Hard gate
A section fails alignment if its chapter-specific concepts could be deleted while leaving essentially the same Application or Case Study. Misplaced sections must be moved, rewritten, merged, or removed.

## Cross-chapter duplication rule
A domain may recur across chapters only when each occurrence answers a different chapter-specific question. For example, fraud detection may appear in Foundations to teach imbalance and metric choice, in Deep Learning to teach learned representations, and in MLOps to teach streaming monitoring; the prose, figure, table, and evaluation question must differ accordingly.
