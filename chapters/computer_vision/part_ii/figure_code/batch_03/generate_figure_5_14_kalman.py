"""Scientific AI, Chapter 5: exact scalar Kalman numerical experiment.
Outputs a 25-step deterministic simulation and uncertainty plot.
Simulated observations only; no real sensor or KITTI data.
Requires numpy, matplotlib. Example: python generate_kalman.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent / "outputs"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(5140)
Q, R, n = 0.035, 0.35, 25
truth = np.cumsum(rng.normal(.33, .15, n))
observations = truth + rng.normal(0, np.sqrt(R), n)
estimate, variance = 0., 1.
prior, posterior, gains, post_var = [], [], [], []
for z in observations:
    prediction = estimate
    predictive_variance = variance + Q
    gain = predictive_variance / (predictive_variance + R)
    estimate = prediction + gain * (z - prediction)
    variance = (1 - gain) * predictive_variance
    prior.append(prediction)
    posterior.append(estimate)
    gains.append(gain)
    post_var.append(variance)

t = np.arange(n)
posterior, post_var = np.array(posterior), np.array(post_var)
np.savetxt(out / "Figure_5_14_Kalman_series.csv",
    np.column_stack([t, truth, observations, prior, posterior, gains, post_var]),
    delimiter=",", comments="",
    header="k,true,observation,prior,posterior,gain,posterior_variance")
plt.rcParams.update({"pdf.fonttype":42, "svg.fonttype":"none"})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
    gridspec_kw={"width_ratios":[2.2, 1]})
navy, teal, gold, plum, red = "#233D59","#187B78","#AF813A","#79547F","#A95552"
axes[0].plot(t, truth, color=navy, label="simulated state", lw=2)
axes[0].scatter(t, observations, color=gold, label="noisy observations", s=15)
axes[0].plot(t, posterior, color=teal, label="posterior", lw=1.7)
axes[0].fill_between(t, posterior - 1.96*np.sqrt(post_var),
    posterior + 1.96*np.sqrt(post_var), color=teal, alpha=.15)
axes[0].set(xlabel="time step k", ylabel="state position (m)")
axes[0].legend(loc="upper left", frameon=False, fontsize=8)
axes[0].set_title("Scalar Kalman recursion (simulated)", color=navy)
axes[1].plot(t, gains, color=plum, lw=2, label="gain")
axes[1].plot(t, post_var, color=red, lw=2, label="posterior variance")
axes[1].set(xlabel="time step k", ylabel="gain / variance")
axes[1].legend(frameon=False)
axes[1].set_title("Uncertainty update", color=plum)
fig.tight_layout()
for extension in ("png", "svg", "pdf"):
    fig.savefig(out / f"Figure_5_14.{extension}", dpi=300, bbox_inches="tight")
plt.close(fig)
