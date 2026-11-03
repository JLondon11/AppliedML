"""Figure 5.38: deterministic camera–LiDAR projection.

This is a geometric construction, not a learned-model or KITTI result.
Usage: python generate_calibration_geometry.py
Dependencies: numpy, matplotlib.
Outputs: calibration_geometry.png, .pdf, .svg in the script directory.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "calibration_geometry"
rng = np.random.default_rng(538)

# LiDAR coordinate frame: x forward, y left, z up.
x = rng.uniform(4.0, 29, 1650)
y = rng.uniform(-5.5, 5.5, 1650)
z = np.full_like(x, -1.55)
ground = np.column_stack((x, y, z))

def box_surface(cx, cy, length, width, height, n):
    lo = np.array([cx-length/2, cy-width/2, -1.55])
    hi = np.array([cx+length/2, cy+width/2, -1.55+height])
    p = rng.uniform(lo, hi, (n, 3))
    side = rng.integers(0, 6, size=n)
    for k in range(3):
        p[side == 2*k, k] = lo[k]
        p[side == 2*k+1, k] = hi[k]
    return p

car1 = box_surface(11.5, -1.9, 3.4, 1.7, 1.5, 570)
car2 = box_surface(22.0, 2.3, 4.2, 1.9, 1.6, 410)
phi = rng.uniform(0, 2*np.pi, 150)
pole = np.column_stack((
    np.full(150, 17.0) + 0.055*np.cos(phi),
    np.full(150, -4.4) + 0.055*np.sin(phi),
    rng.uniform(-1.55, 3.2, 150)
))
pts = np.concatenate([ground, car1, car2, pole])

# R_CL transforms LiDAR to optical camera frame (x right, y down, z forward).
R = np.array([[0,-1,0], [0,0,-1], [1,0,0]], float)
t = np.array([0.10, 0.25, -0.05])
K = np.array([[680.,0.,640.], [0.,680.,360.], [0.,0.,1.]])
cam = pts @ R.T + t
assert np.all(cam[:,2] > 0)
uv_h = cam @ K.T
uv = uv_h[:,:2] / uv_h[:,2,None]
assert np.allclose(uv[:,0], 680*cam[:,0]/cam[:,2] + 640)
assert np.allclose(uv[:,1], 680*cam[:,1]/cam[:,2] + 360)
keep = (uv[:,0]>=0) & (uv[:,0]<=1280) & (uv[:,1]>=0) & (uv[:,1]<=720)

plt.rcParams.update({
    "font.size":9, "font.family":"DejaVu Sans", "axes.labelsize":9,
    "xtick.labelsize":8, "ytick.labelsize":8,
    "pdf.fonttype":42, "svg.fonttype":"none"
})
fig = plt.figure(figsize=(12.3,4.35), facecolor="white")
ax = fig.add_subplot(121, projection="3d")
norm = Normalize(vmin=4, vmax=29)
cmap = plt.get_cmap("viridis")
colors = cmap(norm(pts[:,0]))
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1.55, c=colors,
           alpha=.65, rasterized=True, depthshade=False)
ax.set_xlabel("Forward $x_L$ (m)", labelpad=4)
ax.set_ylabel("Lateral $y_L$ (m)", labelpad=3)
ax.set_zlabel("Height $z_L$ (m)", labelpad=3)
ax.set_xlim(3,30)
ax.set_ylim(-6,6)
ax.set_zlim(-2,4)
ax.view_init(elev=25, azim=-66)
ax.set_box_aspect((2.2,1.05,.7), zoom=.94)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis._axinfo["grid"]["color"] = (.86,.89,.92,.8)
ax.text2D(.03,.96,"(a)",transform=ax.transAxes,
          fontweight="bold",fontsize=11,color="black")

bx = fig.add_subplot(122)
bx.scatter(uv[keep,0], uv[keep,1], s=2.3, c=colors[keep],
           alpha=.72, linewidths=0, rasterized=True)
bx.set_xlim(0,1280)
bx.set_ylim(720,0)
bx.set_xlabel("Image coordinate $u$ (pixels)")
bx.set_ylabel("Image coordinate $v$ (pixels)")
bx.set_aspect("equal", adjustable="box")
bx.grid(alpha=.17, linewidth=.6)
bx.text(.02,.96,"(b)",transform=bx.transAxes,
        fontweight="bold",fontsize=11,color="black")
bx.plot(640,360,"+",markersize=8,markeredgewidth=1,color="#b65424",zorder=6)
bx.annotate("principal point",xy=(640,360),xytext=(670,295),fontsize=7,
            color="#394754",
            arrowprops={"arrowstyle":"-","lw":.6,"color":"#394754"})
fig.subplots_adjust(left=.025,right=.992,top=.93,bottom=.18,wspace=.18)
cbar_ax = fig.add_axes([.28,.075,.44,.022])
cb = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cbar_ax,
                  orientation="horizontal")
cb.set_label("Forward range $x_L$ (m)",fontsize=8,labelpad=2)
cb.ax.tick_params(labelsize=7)
for ext in ("png","pdf","svg"):
    fig.savefig(OUT.with_suffix("."+ext),dpi=230,facecolor="white",
                bbox_inches="tight",pad_inches=.10)
plt.close(fig)
print("points",len(pts),"projected",int(keep.sum()),
      "R determinant",np.linalg.det(R))
