"""Replot published agricultural obstacle-detection measurements, NOT model inference.

Primary source: Christiansen et al., Sensors 16(11):1904, 2016.
DOI: 10.3390/s16111904.
Inputs: 45–90m and 25ms (DeepAnomaly); 0–30m and 182ms (Faster R-CNN).
Run: python replot_published_agriculture.py
Outputs: agriculture_published_benchmark_plot.png/.pdf/.svg.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
reported = {
    "DeepAnomaly": {"range_m": (45.,90.), "runtime_ms": 25.},
    "Faster R-CNN": {"range_m": (0.,30.), "runtime_ms": 182.},
}
assert reported["Faster R-CNN"]["runtime_ms"] / reported["DeepAnomaly"]["runtime_ms"] == 7.28
names = ["Faster R-CNN", "DeepAnomaly"]
colors = ["#183b60", "#12847d"]
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.3), facecolor="white")
a = axs[0]
for i, name in enumerate(names):
    lo, hi = reported[name]["range_m"]
    a.plot([lo,hi],[i,i], color=colors[i], linewidth=10, solid_capstyle="butt")
    a.plot([lo,hi],[i,i], color="white", linewidth=.55, alpha=.6)
    a.text(hi+2,i,f"{lo:g}–{hi:g} m",va="center",fontsize=9,color=colors[i])
a.set_xlim(0,110); a.set_ylim(-.65,1.65); a.set_yticks([0,1],names)
a.set_xlabel("Reported human-detection distance (m)")
a.set_xticks([0,30,60,90]); a.grid(axis="x",alpha=.23); a.set_axisbelow(True)
a.text(.5,-.37,"(a)",transform=a.transAxes,ha="center",va="top",fontsize=11,color="black")
b = axs[1]
times = [reported[n]["runtime_ms"] for n in names]
b.barh(names, times, color=colors, height=.45)
for i,t in enumerate(times):
    b.text(t+4,i,f"{t:g} ms",va="center",fontsize=9)
b.set_xlim(0,210); b.set_xlabel("Reported processing time per image (ms)")
b.grid(axis="x",alpha=.23); b.set_axisbelow(True)
b.text(.5,-.37,"(b)",transform=b.transAxes,ha="center",va="top",fontsize=11,color="black")
for ax in axs:
    ax.invert_yaxis()
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=9)
fig.subplots_adjust(left=.17,right=.97,top=.94,bottom=.31,wspace=.52)
for ext in ("png","pdf","svg"):
    fig.savefig(ROOT/("agriculture_published_benchmark_plot."+ext),
                dpi=230,facecolor="white")
plt.close(fig)
