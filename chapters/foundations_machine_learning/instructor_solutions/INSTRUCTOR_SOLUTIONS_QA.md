# Chapter 1 Instructor Solutions QA

Date: 2026-11-18
Repository target: `JLondon11/AppliedML`
Canonical chapter: `chapters/foundations_machine_learning`

## Coverage audit

The final Chapter 1 PDF contains 50 end-of-chapter items:

- 12 Review Questions
- 15 Mathematical Problems
- 8 Computational Exercises
- 4 Engineering Design Projects
- 10 Research Problems
- 1 Capstone Project

The instructor-solutions package covers **50/50** items.

## Code audit

Thirteen executable solution scripts are provided:

- 8 computational-exercise implementations
- 4 engineering-design reference implementations
- 1 capstone orchestrator

All 13 scripts pass Python bytecode compilation through `verify_solutions_code.py`.

Additional executed numerical checks:

- `ce06_lasso_proximal.py`: proximal-gradient and coordinate-descent objectives agree to floating-point precision; coefficient L2 difference approximately `2.5e-7` under the default iteration budget.
- `ce07_admm_distributed_ridge.py`: consensus ADMM converges to within approximately `3.4e-3` L2 distance of the centralized ridge solution with small primal and dual residuals under the default configuration.
- `ce08_lr_schedules.py`: executed successfully on the complete scikit-learn Digits dataset in a reduced-epoch smoke test.

The CIFAR-100, WikiText-103, HalfCheetah-v4, German Credit, and long PINN/reference-project runs require their documented public datasets/environments and optional dependencies. Their scripts were syntax-validated locally and contain no placeholder functions or unimplemented algorithm labels.

## Scientific correctness checks

- German Credit uses true torch SGD, Momentum, AdamW, and L-BFGS implementations on the same logistic-regression objective.
- CIFAR-100 includes an explicit Lion update rather than substituting AdamW.
- WikiText-103 uses actual AdamW and Adafactor optimizer classes.
- PPO uses the HalfCheetah-v4 environment and exposes learning-rate, clipping, entropy, and gradient-clipping controls.
- LASSO uses the proximal soft-thresholding operator and verifies against coordinate descent.
- Distributed ridge uses consensus ADMM and verifies against the centralized ridge solution.
- The distributed-transformer reference supports optional FSDP, AMP/BF16, gradient accumulation, checkpoint recovery, and throughput reporting.

## Release conclusion

**Instructor-solutions completion: PASS.**
**Solution-code repository requirement: PASS after the commit containing this package is present on the canonical Chapter 1 tree.**
