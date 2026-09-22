# Figure 14 — Application: Grad-CAM Explainability Pipeline for Retinal Classification

**Section:** Artificial Intelligence for Ophthalmology and Retinal Imaging  
**Subsection:** Application: Data Augmentation and Gradient-Based Explainability for Retinal Classifiers  
**Classification:** Application  
**Figure type:** Evidence-grounded scientific rendering  
**LaTeX label:** `fig:scientific_ai_10_application_grad_cam_explainability_pipeline_for_retinal_cla`

**Unified production caption:** Grad-CAM explainability for retinal classification using an executable CNN trained on the official RetinaMNIST training split and evaluated on its held-out test split. **(a)** Real held-out RetinaMNIST fundus image selected deterministically as the first correctly classified test example. **(b)** Grad-CAM activation map computed by backpropagating the predicted-class score to the classifier's final convolutional layer and combining its feature maps using channel-wise mean gradient weights; the quantitative colorbar reports normalized Grad-CAM activation from 0 to 1. **(c)** The computed Grad-CAM map registered and overlaid on the same source fundus image, providing the spatial attribution view used for model audit. The plate demonstrates the pipeline directly from saved model activations and gradients rather than using a schematic or synthetic retinal image.

RetinaMNIST is a low-resolution benchmark, so this figure should be interpreted as a reproducible methodological demonstration of gradient-based attribution rather than as a clinical-resolution retinal audit. Panel identifiers are below each panel.
