# Figure 25 — Case Study III: Compute–Quality–Sampling Tradeoff

Assigned section: Case Studies — Case Study III: Efficiency and Deployment  
Classification: Case Study figure

The code benchmarks the public pretrained model `google/ddpm-cifar10-32` on authentic CIFAR-10 observations using DDPM, DDIM, and DPM-Solver++ at matched inference-step budgets.

Measured outputs include FID, wall-clock sampling latency, throughput, exact UNet forward-call counts/NFE, peak CUDA memory where applicable, repeat variation, and a full hardware/software environment record.

The plot command refuses to render without real benchmark CSVs. Pareto-dominated points are identified by minimizing both latency and FID.

Production example:

```bash
python src/figure25_compute_quality_sampling.py benchmark --device cuda --num-images 5000 --batch-size 64 --repeats 3 --steps 10 20 50 100
python src/figure25_compute_quality_sampling.py plot
```

Outputs: `figure25.png`, `figure25.svg`, `figure25.pdf`, run CSV, summary CSV, and environment JSON under `outputs/`.
