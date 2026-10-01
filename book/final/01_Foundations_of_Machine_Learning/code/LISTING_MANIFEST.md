# Chapter 1 Code Listing Repository Manifest

Authoritative manuscript baseline: Pass20 Chapter 1.

Every numbered listing below has a full runnable repository companion. The manuscript may show a pedagogically focused excerpt, but these files are the complete implementations that must remain semantically synchronized with the published text.

| Listing | Title | Repository path |
|---|---|---|
| 1.1 | Minimal AdamW optimizer update | `code/optimization/listing_01_01_adamw_update.py` |
| 1.2 | Comparing L-BFGS and SGD-family solvers for convex logistic regression | `code/optimization/credit_risk/listing_01_02_lbfgs_vs_sgd.py` |
| 1.3 | PyTorch CNN training loop instrumented for optimizer memory and accuracy comparison | `code/optimization/cnn/listing_01_03_cnn_optimizer_memory_accuracy.py` |
| 1.4 | Hybrid AdamW-then-L-BFGS training strategy for a PINN | `code/optimization/pinn/listing_01_04_hybrid_adamw_lbfgs_pinn.py` |
| 1.5 | Core PPO clipped surrogate objective computation | `code/optimization/reinforcement_learning/listing_01_05_ppo_clipped_surrogate.py` |
| 1.6 | Windowing and spectral feature extraction for bearing vibration signals | `code/applications/predictive_maintenance/listing_01_06_bearing_windowing_spectral_features.py` |
| 1.7 | Expected calibration error (ECE) for a binary classifier | `code/applications/medical_image_classification/listing_01_07_expected_calibration_error.py` |
| 1.8 | Cost-sensitive threshold selection for a fraud classifier | `code/applications/financial_fraud/listing_01_08_cost_sensitive_threshold.py` |
| 1.9 | Bird's-eye-view bounding-box IoU for 3D detection evaluation | `code/applications/intelligent_transportation/listing_01_09_bev_bbox_iou.py` |
| 1.10 | Latitude-weighted RMSE for global weather forecast evaluation | `code/applications/climate_weather/listing_01_10_latitude_weighted_rmse.py` |
| 1.11 | INT8 post-training quantization of a weight tensor | `code/applications/edge_ai/listing_01_11_int8_post_training_quantization.py` |
| 1.12 | Pinball (quantile) loss for probabilistic load forecasting | `code/applications/smart_grid/listing_01_12_pinball_loss.py` |
| 1.13 | Asymmetric PHM08 prognostic scoring function | `code/applications/aerospace_prognostics/listing_01_13_phm08_score.py` |
| 1.14 | NDVI computation from multispectral imagery | `code/applications/precision_agriculture/listing_01_14_ndvi.py` |
| 1.15 | Randomized hyperparameter search with cross-validation | `code/hyperparameter_optimization/random_search/listing_01_15_randomized_search_cv.py` |
| 1.16 | Define-by-run hyperparameter search using Optuna | `code/hyperparameter_optimization/optuna/listing_01_16_define_by_run_optuna.py` |
| 1.17 | Multi-objective hyperparameter search with competing directions | `code/hyperparameter_optimization/multi_objective/listing_01_17_multi_objective_optuna.py` |
| 1.18 | Hyperparameter search with median-based trial pruning | `code/hyperparameter_optimization/pruning/listing_01_18_median_pruner.py` |
| 1.19 | Tracked hyperparameter search for a reproducible medical-imaging benchmark | `code/hyperparameter_optimization/tracked_medical_imaging/listing_01_19_tracked_medical_hpo.py` |

## Dependencies

Core:
- Python 3.11+
- NumPy
- SciPy
- scikit-learn

Listing-specific:
- PyTorch: Listings 1.3-1.5
- Optuna: Listings 1.16-1.19

## Repository policy

1. Manuscript listing number and repository filename must remain one-to-one.
2. A code behavior change requires editing both the manuscript excerpt and the full companion.
3. If a listing moves to another chapter during consolidation, preserve git history and move the complete implementation with it.
4. Every retained listing must use the book's line-numbered LaTeX style and be substantively referenced/discussed in prose.
