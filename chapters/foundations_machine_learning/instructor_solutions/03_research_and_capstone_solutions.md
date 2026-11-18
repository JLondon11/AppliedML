# Sections 1.29-1.30 Instructor Solutions

## 1.29 Research Problems

These are open research problems. The instructor solution is a reference experimental design and acceptance rubric, not a single fixed numerical answer.

### 1. Optimizer-state compression
Compare quantization, factored moments, low-rank state, blockwise statistics, and offloading under a fixed model/data budget. Report memory per parameter, tokens/s, convergence loss, and downstream quality. Hold effective batch size, schedule, and training tokens fixed.

### 2. Implicit regularization of AdamW, Lion, and Sophia
Use identical architecture, initialization distribution, token order, schedule budget, and explicit weight decay. Compare training/validation loss, parameter norms, curvature approximations, and robustness. Include matched-decay and matched-effective-step ablations.

### 3. Communication-aware adaptive optimizers
Model communication and heterogeneous compute explicitly. Compare synchronous baselines, compression, local/adaptive update variants, and overlap strategies. Report wall-clock convergence, bytes communicated, straggler sensitivity, and time-to-target.

### 4. Hessian spectrum, flat minima, and robustness
Estimate leading Hessian eigenvalues or spectral density at checkpoints and evaluate corruption robustness. Control for model size and training loss. Use interventions such as sharpness-aware training or regularization rather than relying only on correlation.

### 5. Hybrid first-/second-order methods
Define a switching rule based on gradient norm, curvature, or plateau detection. Compare first-order, quasi-Newton/Krylov, and hybrid schedules. Report field/task error, residual, wall time, memory, and sensitivity to the switching criterion.

### 6. BF16 and FP8 stability
Hold model, data, optimizer, and training budget fixed while changing numerical format. Track overflow/underflow, loss scaling, gradient statistics, convergence, and final metrics. Report numerical failures rather than silently dropping runs.

### 7. Policy optimization with reduced KL spikes
Compare clipping, explicit KL penalties, adaptive trust regions, and proposed mechanisms at matched environment interactions. Report return, sample efficiency, KL distribution, catastrophic-update frequency, and seed variance.

### 8. Distribution shift and continual learning
Create controlled task sequences or shifts and measure forward/backward transfer, forgetting, calibration, and optimizer-state adaptation. Include reset-versus-retain optimizer-state ablations.

### 9. Hardware-aware optimizers
Co-design state representation and update kernels around memory bandwidth, low-precision formats, and communication topology. Report achieved bandwidth, kernel occupancy, optimizer memory, step time, energy if available, and convergence.

### 10. Convergence theory for adaptive optimizers
State smoothness/noise/adaptivity assumptions and prove a stationarity, regret, or convergence result. Identify which assumptions differ from practical LLM training. Test a prediction of the theory empirically, such as sensitivity to beta values or gradient-noise scale.

## 1.30 Capstone Project

### Unified optimization benchmark suite
**Code:** `code/capstone_unified_benchmark.py`.

A complete capstone contains convex statistical learning, image classification, transformer language modeling, PINN optimization, and reinforcement-learning benchmarks. Each benchmark defines provenance, split or rollout protocol, task metric, runtime, memory, and convergence diagnostics. At least five optimizers should be evaluated where technically meaningful; incompatible optimizers should not be forced onto inappropriate tasks.

Minimum acceptance criteria: reproducible configuration, deterministic seeds where possible, immutable raw results, generated plots, hardware/software metadata, checkpoint recovery for long runs, uncertainty across seeds/folds when stochasticity matters, and conclusions bounded to the evidence.

Recommended grading: 20% scientific design/provenance, 20% implementation correctness, 20% quantitative evaluation, 15% reproducibility, 15% engineering/system analysis, 10% conclusions and limitations.
