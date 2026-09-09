# Figure 14 — Application: Grad-CAM Explainability Pipeline for Retinal Classification

**Section:** Artificial Intelligence for Ophthalmology and Retinal Imaging  
**Subsection:** Application: Data Augmentation and Gradient-Based Explainability for Retinal Classifiers  
**Classification:** Application  
**Figure type:** Evidence-grounded scientific workflow / architecture rendering  
**LaTeX label:** `fig:scientific_ai_10_application_grad_cam_explainability_pipeline_for_retinal_cla`

**Frozen caption:** Grad-CAM explainability pipeline for a retinal classifier, showing gradient backpropagation to a late convolutional layer, channel-wise weighting, and heatmap overlay onto the source fundus image for clinical audit.

**Unified production caption:** Grad-CAM explainability for retinal classification using an executable CNN trained on the official RetinaMNIST training split and evaluated on its held-out test split. **(a)** Real held-out RetinaMNIST fundus image selected deterministically as the first correctly classified test example. **(b)** Grad-CAM activation map computed by backpropagating the predicted-class score to the classifier's final convolutional layer and combining its feature maps using channel-wise mean gradient weights; no heatmap values are manually positioned. **(c)** The computed Grad-CAM map registered and overlaid on the same source fundus image, providing the spatial attribution view used for model audit. The plate demonstrates the frozen pipeline directly from saved model activations and gradients rather than using a schematic or synthetic retinal image.

The inventory does not prescribe fixed panels; these three panels are the minimal scientific decomposition of source image, computed attribution, and registered audit overlay. Panel identifiers are below each panel.
