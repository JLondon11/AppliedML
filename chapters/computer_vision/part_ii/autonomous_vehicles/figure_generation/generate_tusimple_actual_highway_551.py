"""Figure 5.51: true TuSimple highway photograph + matching original lane annotations.

The image and annotations are both from the same original TuSimple training frame.
This is a ground-truth annotation visualization, NOT predicted lane detection.
The downloaded archive and source photographs are not committed to GitHub.
"""
from pathlib import Path
import argparse
import json
import zipfile
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

DATASET="AhmedNasr5804/Lane-and-eye-detection-dataset"
RAW_FRAME="clips/0313-1/9540/20.jpg"

def annotations(archive):
    for name in archive.namelist():
        if not name.lower().endswith((".json",".jsonl")):
            continue
        try:
            text=archive.read(name).decode("utf-8")
            records=json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
            for rec in records:
                if isinstance(rec,dict) and rec.get("raw_file")==RAW_FRAME and "lanes" in rec:
                    return rec,name
        except (ValueError,UnicodeError,TypeError):
            pass
    raise FileNotFoundError("Missing exact matching TuSimple lane annotation "+RAW_FRAME)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("tusimple_real_highway"))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    zfile=Path(hf_hub_download(repo_id=DATASET,repo_type="dataset",filename="tusimple.zip"))
    with zipfile.ZipFile(zfile) as z:
        record,source=annotations(z)
        segments=RAW_FRAME.split("/")
        flattened=segments[-3]+"_"+segments[-2]+".jpg"
        matches=[n for n in z.namelist() if n.endswith("/training/frames/"+flattened)]
        if len(matches)!=1:
            raise RuntimeError("Expected a unique genuine corresponding frame; found "+str(matches))
        image=cv2.imdecode(np.frombuffer(z.read(matches[0]),np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Original TuSimple photograph failed decoding")
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    if (w,h)!=(1280,720):
        raise ValueError(f"Unexpected original TuSimple resolution: {(w,h)}")
    annotated=rgb.copy()
    binary=np.zeros((h,w),np.uint8)
    colors=[(255,204,64),(70,206,215),(209,115,218),(80,229,139),(240,130,73)]
    counts=[]
    for i,lane in enumerate(record["lanes"]):
        if len(lane)!=len(record["h_samples"]):
            raise ValueError("Official annotation x-y row count mismatch")
        points=np.asarray([(int(x),int(y)) for x,y in zip(lane,record["h_samples"])
                           if 0<=int(x)<w and 0<=int(y)<h],dtype=np.int32)
        counts.append(int(len(points)))
        if len(points)<2:
            continue
        cv2.polylines(annotated,[points],False,colors[i][::-1],5,lineType=cv2.LINE_AA)
        # exact colors apply in RGB via OpenCV direct-array drawing, reversed BGR specified
        cv2.polylines(binary,[points],False,255,4,lineType=cv2.LINE_8)
        for x,y in points[::5]:
            cv2.circle(annotated,(int(x),int(y)),3,colors[i][::-1],-1,lineType=cv2.LINE_AA)
    if sum(counts)<20 or np.count_nonzero(binary)<1000:
        raise RuntimeError("No meaningful real lane annotation found")

    fig,axes=plt.subplots(1,2,figsize=(12.7,4.5),
                          gridspec_kw={"width_ratios":[1.65,1]},
                          layout="constrained")
    axes[0].imshow(annotated)
    axes[0].set_title("(a) Original TuSimple highway frame with annotated lanes",fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(binary,cmap="gray",vmin=0,vmax=255)
    axes[1].set_title("(b) Matching lane reference mask",fontsize=10)
    axes[1].axis("off")
    for ext in ("png","pdf","svg"):
        fig.savefig(a.output/f"tusimple_actual_highway_551.{ext}",dpi=210,
                    facecolor="white",bbox_inches="tight",pad_inches=.05)
    plt.close(fig)
    results={
        "source_dataset":"TuSimple (original frame and ground-truth lane data)",
        "mirror":DATASET,"annotated_frame":RAW_FRAME,
        "source_annotation_entry":source,
        "frame_resolution":[w,h],
        "annotated_lane_point_counts":counts,
        "rasterized_reference_mask_pixels":int(np.count_nonzero(binary)),
        "plot_type":"Ground-truth annotation, not trained-model prediction"
    }
    (a.output/"figure_551_source.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2),flush=True)

if __name__=="__main__":
    main()
