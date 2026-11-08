"""Integrate original KITTI annotated 3D GT results into Chapter 5 LaTeX.

Requires a verified original-data model-vs-GT artifact from Actions 42531178943.
Uses existing PointPillars section and refuses double-insertion. Compiles
LaTeX separately; the tool never invents benchmark results or official AP.
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

FIGURE="kitti_real_3d_gt_vs_pointpillars"
FIGLABEL="fig:cv5_pointpillars_actual_3d_gt"
TABLELABEL="tab:cv5_pointpillars_actual_3d_gt"

def integrate(tex_path, original_artifact):
    tex_path=Path(tex_path)
    original_artifact=Path(original_artifact)
    if not tex_path.is_file():raise FileNotFoundError(tex_path)
    if not original_artifact.is_file():raise FileNotFoundError(original_artifact)
    s=tex_path.read_text(encoding="utf-8")
    if FIGLABEL in s or TABLELABEL in s:
        raise RuntimeError("Already inserted original GT evidence into this manuscript")
    if "\\subsection{Case Study: KITTI 3D Object Detection Benchmarking and Real-Time Inference Latency}" not in s:
        raise RuntimeError("Wrong manuscript or section")
    with zipfile.ZipFile(original_artifact) as z:
        record=json.loads(z.read("measured_3d_iou.json"))
        assert record["sample"]=="000008"
        assert record["original_annotated_cars"]==6
        assert record["original_nonnegative_difficulty_cars"]==4
        assert record["actual_pointpillars_car_proposals"]==10
        assert all(any(r["scope"]==scope and r["threshold_3d_iou"]==t
                       and r["matched_reference_cars"]==num for r in record["results"])
                   for scope,num in [("all_six_annotated_cars",6),
                                     ("four_nonnegative_difficulty_cars",4)]
                   for t in (.25,.50,.70))
        figs=tex_path.parent/"Figures"
        figs.mkdir(exist_ok=True)
        for ext in ("pdf","png","svg"):
            target=f"{FIGURE}.{ext}"
            (figs/target).write_bytes(z.read(target))
    snippet=r"""
\noindent\textbf{Original ground-truth 3D localization check.} Original KITTI scene 000008 contains camera-coordinate, bottom-centered oriented 3D car annotations and the calibrated camera-to-LiDAR transformation. Subtract half the original camera-box height from the bottom-center camera y coordinate, apply the inverse lidar2cam homogeneous transformation, and rotate the camera box's longitudinal axis into LiDAR yaw. Compute exact oriented 3D overlap as the BEV polygon-intersection area multiplied by vertical intersection height, divided by union volume. One-to-one matches are assigned in confidence-score order.

Of ten actual model Car proposals, six independently match all six original Car references at a genuine 3D IoU threshold of 0.70. Four of the six original references have nonnegative KITTI difficulty codes, and all four are matched. Figure~\ref{fig:cv5_pointpillars_actual_3d_gt} shows the actual reference and network geometry, while Table~\ref{tab:cv5_pointpillars_actual_3d_gt} compares match counts under three localization thresholds.

\begin{figure}[!htbp]\centering
\includegraphics[width=.82\linewidth,height=.28\textheight,keepaspectratio]{Figures/kitti_real_3d_gt_vs_pointpillars.pdf}
\caption{Original KITTI frame 000008 LiDAR bird's-eye view. Solid teal boxes denote original calibrated 3D Car reference annotations; dashed orange boxes denote genuine pretrained PointPillars class-Car predictions. Original 3D overlap is evaluated geometrically; no full KITTI AP performance is inferred.}
\label{fig:cv5_pointpillars_actual_3d_gt}
\end{figure}

\begin{table}[!htbp]\centering\small
\caption{Local one-to-one original-annotation 3D Car overlap on the single original KITTI training scene 000008, not the official KITTI AP40 benchmark. The four-car subset includes only objects with nonnegative difficulty codes.}
\label{tab:cv5_pointpillars_actual_3d_gt}
\begin{tabularx}{.95\linewidth}{@{}Xrrrr@{}}\toprule
Original annotations & 3D IoU & Matched & References & Local recall\\\midrule
All six 3D cars & 0.25 & 6 & 6 & 1.000\\
All six 3D cars & 0.50 & 6 & 6 & 1.000\\
All six 3D cars & 0.70 & 6 & 6 & 1.000\\
Nonnegative-difficulty cars & 0.70 & 4 & 4 & 1.000\\\bottomrule
\end{tabularx}
\end{table}

\noindent\textbf{Measured-versus-official interpretation.} A perfect match fraction on one training scene is not a KITTI benchmark-wide recall estimate. These measurements establish that original calibrated reference boxes and real predicted boxes can be compared correctly in 3D; they do not measure model generalization, class-specific average precision on a held-out split, or deployment latency. The source paper's GPU stage-total and this one CPU model-only timing are not hardware-matched experiments.
"""
    old="\\noindent\\textbf{Comparison and limitations.} The model's output demonstrates functioning pretrained LiDAR inference"
    idx=s.find(old)
    if idx==-1:raise RuntimeError("Cannot locate old single-frame limitations paragraph")
    # Insert measured original GT comparison immediately BEFORE the existing
    # limitations, which are retained to preserve the chapter's safety context.
    s=s[:idx]+snippet+"\n"+s[idx:]
    s=s.replace("\noindent\\textbf{Benchmark interpretation.}", "\n\\noindent\\textbf{Benchmark interpretation.}")
    s=s.replace("No accuracy is inferred from the absence of a full ground-truth comparison.",
                "Original 3D annotation accuracy is evaluated separately in Figure~\\ref{fig:cv5_pointpillars_actual_3d_gt}.")
    tex_path.write_text(s,encoding="utf-8")
    print("Inserted original-data KITTI ground-truth box comparison into",tex_path)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("chapter_tex",type=Path)
    p.add_argument("executed_original_gt_artifact",type=Path)
    a=p.parse_args()
    integrate(a.chapter_tex,a.executed_original_gt_artifact)
