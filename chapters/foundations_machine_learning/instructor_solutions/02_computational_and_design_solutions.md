# Sections 1.27-1.28 Instructor Solutions

## 1.27 Computational Exercises

All reference code is in `code/`.

### 1. German Credit logistic regression
**Code:** `ce01_german_credit_optimizers.py`.
Use one preprocessing pipeline and one stratified split for SGD, Momentum, AdamW, and L-BFGS. Report ROC-AUC, log loss, runtime, and final gradient norm. L-BFGS should converge efficiently on this smooth convex problem, while first-order methods require learning-rate tuning. Conclusions must use identical data and preprocessing.

### 2. CIFAR-100 CNN optimizer comparison
**Code:** `ce02_cifar100_optimizers.py`.
Train the same CNN with Momentum, AdamW, and a true Lion update. Report top-1 accuracy, calibration error, peak accelerator memory, runtime, and convergence curves. AdamW often reaches good accuracy quickly; Momentum can be competitive with tuning; Lion reduces optimizer-state memory relative to AdamW.

### 3. WikiText-103 transformer
**Code:** `ce03_wikitext103_transformer.py`.
Fine-tune the same decoder-only transformer with AdamW and Adafactor. Report validation perplexity, optimizer-state bytes, and runtime. Adafactor should reduce state memory substantially on matrix parameters; final perplexity depends on its learning-rate configuration.

### 4. Burgers PINN
**Code:** `ce04_burgers_pinn.py`.
Compare AdamW, L-BFGS, and hybrid optimization under the same network and collocation design. Report relative field error and PDE residual on a fixed evaluation grid. AdamW is usually robust early; L-BFGS is often effective for late-stage refinement.

### 5. PPO on HalfCheetah-v4
**Code:** `ce05_ppo_halfcheetah.py`.
Vary learning rate, clipping threshold, entropy coefficient, and gradient clipping. Report episodic return, approximate KL, clip fraction, entropy, and gradient norm across seeds. Large rates or clip ranges can destabilize policy updates.

### 6. Proximal gradient for LASSO
**Code:** `ce06_lasso_proximal.py`.
Compare proximal gradient with coordinate descent at the same regularization coefficient. Report objective value, coefficient sparsity, KKT residual or coefficient agreement, and runtime. The reference implementation converges to the coordinate-descent solution to numerical tolerance.

### 7. Distributed ridge with ADMM
**Code:** `ce07_admm_distributed_ridge.py`.
Use consensus ADMM on partitioned data and compare the consensus parameter vector with the centralized ridge solution. Report primal/dual residuals and communication rounds. The default reference configuration converges closely to the centralized solution.

### 8. Learning-rate schedules
**Code:** `ce08_lr_schedules.py`.
Compare constant, step, cosine, and warmup+cosine schedules using identical initialization and data. Report validation performance and learning curves. Warmup is most useful when the target rate is large enough to destabilize early training.

## 1.28 Engineering Design Projects

### 1. Optimizer benchmarking framework
**Code:** `project01_optimizer_benchmark.py`.
The reference design separates configuration from execution, uses a common result schema, records runtime and output, and isolates failures by task. Acceptance requires reproducible seeds, common metric naming, immutable machine-readable results, and plots generated from logged results rather than hand-entered values.

### 2. Distributed transformer training
**Code:** `project02_distributed_transformer.py`.
The reference supports mixed precision, gradient accumulation, optional FSDP, checkpoint recovery, and throughput reporting. A complete submission must checkpoint optimizer and RNG state, synchronize correctly at accumulation boundaries, and report tokens/s with world size and hardware.

### 3. Scientific-ML optimization package
**Code:** `project03_sciml_optimization.py`.
The reference provides adaptive loss weighting, residual-based collocation refinement, and hybrid AdamW-to-L-BFGS training. Acceptance requires separate reporting of field error and physics residual, explicit boundary/initial-condition losses, and deterministic collocation seeds.

### 4. Optimization diagnostics dashboard
**Code:** `project04_diagnostics_dashboard.py`.
The dashboard reads tidy logged metrics and visualizes loss, validation metrics, gradient norm, memory, communication, and utilization. Instructor acceptance requires provenance and timestamp consistency; fabricated arrays are not acceptable.
