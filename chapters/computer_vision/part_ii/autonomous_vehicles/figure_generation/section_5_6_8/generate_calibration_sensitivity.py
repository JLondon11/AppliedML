"""Controlled, reproducible calibration-sensitivity demonstration for §5.6.8.

NOT a run of KITTI, nuScenes, Waymo, or Argoverse 2; not model training.
Requires NumPy, Matplotlib; seed 568. Generates PDF, PNG, SVG, and CSV.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Figures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(568)
n = 360
z = rng.uniform(12, 50, n)
x = rng.uniform(-3.0, 3.0, n)
y = rng.uniform(-1.2, 1.2, n)
p = np.column_stack((x, y, z))
vx = rng.uniform(-3.0, 3.0, n)
fy = fx = 680.0

def project(q):
    assert np.all(q[:, 2] > 0)
    return np.column_stack((fx*q[:, 0]/q[:, 2]+640., fy*q[:, 1]/q[:, 2]+360.))

truth = project(p)
delays = [0, 0.05, 0.1, 0.2]
yaws = [0, .1, .25, .5]
rows = []

def rms(a):
    return float(np.sqrt(np.mean(np.sum(a*a, axis=1))))

for dt in delays:
    pert = p.copy()
    pert[:, 0] += vx*dt
    rows.append(("time_offset_ms", dt*1000, rms(project(pert)-truth)))

for deg in yaws:
    theta = np.deg2rad(deg)
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    pert = p @ R.T
    rows.append(("yaw_error_deg", deg, rms(project(pert)-truth)))

assert rows[0][2] == 0 and rows[len(delays)][2] == 0
assert all(rows[i][2] < rows[i+1][2] for i in (0, 1, 2, 4, 5, 6))
with (ROOT/"calibration_sensitivity_results.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["perturbation","magnitude","rms_image_reprojection_error_px"])
    writer.writerows(rows)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                     "axes.labelsize":9,"pdf.fonttype":42,"svg.fonttype":"none"})
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.0), facecolor="white")
for i, (ax, series, xlabel, color) in enumerate(zip(
    axes, [rows[:4], rows[4:]],
    ["Camera–LiDAR time offset (ms)","Residual extrinsic yaw error (degrees)"],
    ["#137e7b","#bd622a"])):
    a = np.array([(r[1],r[2]) for r in series])
    ax.plot(a[:,0],a[:,1],marker="o",lw=1.8,markersize=5,color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMS reprojection error (pixels)")
    ax.grid(alpha=.22)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(.02,.96,f'({"ab"[i]})',transform=ax.transAxes,va="top",
            color="black",fontsize=10)
    for t,e in a[1:]:
        ax.annotate(f"{e:.1f}",(t,e),xytext=(0,6),
                    textcoords="offset points",ha="center",fontsize=7)
fig.subplots_adjust(left=.085,right=.985,bottom=.26,top=.92,wspace=.33)
for ext in ("png","pdf","svg"):
    fig.savefig(OUT/f"calibration_timing_executed.{ext}",
                dpi=240,facecolor="white")
plt.close(fig)
for row in rows:
    print(f"{row[0]:18s} {row[1]:8.3f} {row[2]:9.3f} px")
