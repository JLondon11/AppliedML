"""Reconcile old Computer Vision Part II source figures into revised figures 22, 24 and 26.
This script expects the prior accepted source-panel PNGs under source_panels/.
It does not alter empirical scientific pixels; it only composes panels and adds panel IDs.
"""
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

SRC=Path("source_panels")
OUT=Path("../figures")

def compose(a_name,b_name,out_stem,vertical=False):
    a=Image.open(SRC/a_name); b=Image.open(SRC/b_name)
    if vertical:
        fig,axs=plt.subplots(2,1,figsize=(12.5,9.0),constrained_layout=True)
    else:
        fig,axs=plt.subplots(1,2,figsize=(13.5,5.4),constrained_layout=True)
    for ax,im,label in zip(axs,[a,b],["(a)","(b)"]):
        ax.imshow(im); ax.axis("off")
        ax.text(.5,-.025,label,transform=ax.transAxes,ha="center",va="top",fontsize=10)
    for ext in ("png","svg","pdf"):
        fig.savefig(OUT/f"{out_stem}.{ext}",dpi=350 if ext=="png" else None,
                    bbox_inches="tight",facecolor="white")
    plt.close(fig)

compose("old_24_BEV_LiDAR_with_HD_Map_Information.png",
        "old_25_HDNET_Network_Structures.png",
        "Figure_22_HDNet_HDPMap_BEV_and_Detection_Architecture")

compose("old_27_PointPillars_Qualitative_KITTI_Analysis.png",
        "old_28_PointPillars_Failure_Cases_on_KITTI.png",
        "Figure_24_PointPillars_KITTI_Qualitative_and_Failure_Cases",
        vertical=True)

compose("old_30_BEV_versus_Range_View.png",
        "old_31_LiDAR_Views_for_3D_Object_Detection.png",
        "Figure_26_LiDAR_Representation_Geometry_3D_BEV_Range_View",
        vertical=True)
