"""Deterministic scientific illustrations for Chapter 5 figures 5.1 and 5.3.

Computed synthetic shape geometry; these are scientific METHOD illustrations,
not measured experimental data, not KITTI imagery and not benchmarks.
SVG/PDF are native vector artwork; PNG is a raster derivative.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT=Path(__file__).resolve().parents[1]/'figures'
OUT.mkdir(exist_ok=True,parents=True)
NAVY='#18364E'; SLATE='#64748B'; TEAL='#15766E'; GOLD='#A67B32'; PLUM='#6C517D'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
rng=np.random.default_rng(5303)

def box_surface(bounds, n):
    lo=np.array([bounds[0],bounds[2],bounds[4]])
    hi=np.array([bounds[1],bounds[3],bounds[5]])
    arr=rng.uniform(lo,hi,size=(n,3))
    side=rng.integers(0,6,size=n)
    for j in range(3):
        arr[side==2*j,j]=lo[j]
        arr[side==2*j+1,j]=hi[j]
    return arr

parts=[box_surface((-1.9,1.9,-.80,.80,.3,1.05),5500),
       box_surface((-.85,1.05,-.70,.70,1.05,1.75),1800)]
for x in (-1.2,1.15):
    for y in (-.79,.79):
        theta=rng.uniform(0,2*np.pi,850)
        rr=np.sqrt(rng.uniform(.08,.25,850))
        yy=y+rng.uniform(-.12,.12,850)
        parts.append(np.c_[x+rr*np.cos(theta),yy,.34+rr*np.sin(theta)])
P=np.vstack(parts)
np.save(OUT/'Figure_5_03_synthetic_point_cloud.npy',P)

fig=plt.figure(figsize=(11.8,5.8),facecolor='white')
# Main title resides in the LaTeX caption, not in the artwork.
# Data provenance is stated in the LaTeX caption.
axes=[fig.add_axes([.045,.27,.275,.55],projection='3d'),fig.add_axes([.373,.27,.275,.55],projection='3d'),fig.add_axes([.710,.31,.235,.45])]
q=P[::7]
axes[0].scatter(q[:,0],q[:,1],q[:,2],s=.35,c=PLUM,alpha=.55,rasterized=False)
axes[0].set_title('Direct / set-based processing',pad=8,color=NAVY,fontsize=11)
axes[0].set_axis_off(); axes[0].view_init(elev=22,azim=-62);axes[0].set_box_aspect((3.8,1.9,1.8))
bins=[np.linspace(-2,2,22),np.linspace(-1.2,1.2,15),np.linspace(0,2,13)]
H,_=np.histogramdd(P,bins=bins)
occ=H>2
axes[1].voxels(occ,facecolors=TEAL,edgecolors='#F5F6F8',linewidth=.15,alpha=.85)
axes[1].set_title('Voxelized grid (3D / sparse CNN)',pad=8,color=NAVY,fontsize=11)
axes[1].set_axis_off();axes[1].view_init(elev=22,azim=-62);axes[1].set_box_aspect((22,14,12))
axes[2].hist2d(P[:,0],P[:,1],bins=(85,42),cmap='cividis')
axes[2].set_aspect('equal');axes[2].set_xlabel('x (m)');axes[2].set_ylabel('y (m)');axes[2].set_title('Projected 2D BEV (2D CNN)',pad=8,color=NAVY,fontsize=11)
fig.text(.175,.19,'unordered points  →  shared per-point MLP  →  symmetric pool',ha='center',fontsize=8,color=PLUM)
fig.text(.51,.19,'metric voxel grid  →  3D / sparse convolution',ha='center',fontsize=8,color=TEAL)
fig.text(.82,.19,'projection  →  2D convolution',ha='center',fontsize=8,color=GOLD)
# No explanatory caption printed in artwork.
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'Figure_5_01_Point_Cloud_Representations.{ext}',dpi=250,bbox_inches='tight',pad_inches=.15)
plt.close(fig)

fig=plt.figure(figsize=(11.7,6),facecolor='white')
ax3=fig.add_axes([.01,.20,.32,.63],projection='3d')
q=P[::5]
ax3.scatter(q[:,0],q[:,1],q[:,2],s=.45,alpha=.7,color=PLUM,rasterized=False)
ax3.view_init(elev=20,azim=-58);ax3.set_box_aspect((3.8,1.6,1.7));ax3.set_axis_off()
ax3.set_title('3D point set (synthetic geometry)',color=NAVY,fontsize=11,pad=12)
views=[('Side view','x','z',NAVY),('Front view','y','z',TEAL),('Top view','x','y',GOLD),('Rear view','y','z',PLUM)]
cols=[(0,2),(1,2),(0,1),(1,2)]
for i,((name,xname,yname,color),(cx,cy)) in enumerate(zip(views,cols)):
    ax=fig.add_axes([.415+(i%2)*.310,.59-(i//2)*.39,.225,.225])
    values=P[:,[cx,cy]].copy()
    if i==3:values[:,0]*=-1
    ax.hist2d(values[:,0],values[:,1],bins=(75,52),cmap=('cividis' if i%2==0 else 'viridis'))
    ax.set_title(name,fontsize=10,color=color,pad=4)
    ax.set_xlabel(xname+' (m)',fontsize=8)
    ax.set_ylabel(yname+' (m)',fontsize=8)
    ax.tick_params(labelsize=7)
# Main title resides in LaTeX caption.
# Projection limitations described in LaTeX caption.
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'Figure_5_03_Computed_Multiview.{ext}',dpi=240,bbox_inches='tight',pad_inches=.16)
plt.close(fig)
print('Computed 5.1 and 5.3 from',len(P),'simulated 3D points')
