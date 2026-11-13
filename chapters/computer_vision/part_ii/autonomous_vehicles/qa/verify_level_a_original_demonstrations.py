"""Evidence-contract audit for original YOLO/VisDrone and Faster R-CNN/Penn–Fudan.

This independently checks recorded outputs and numerical consistency; it does
not rerun neural networks or recompute AP from omitted original GT label files.
"""
import argparse,csv,hashlib,io,json,math,zipfile
from collections import Counter
from pathlib import Path

def table(archive,member):
    return list(csv.DictReader(io.StringIO(archive.read(member).decode("utf-8"))))

def metric_dict(items):
    return {r["metric"]:float(r["value"]) for r in items}

def inspect(yolo,penn):
    results={}
    for key,path in (("yolo",yolo),("pennfudan",penn)):
        with zipfile.ZipFile(path) as archive:
            members=archive.namelist()
            if len(set(members))!=len(members):raise ValueError("Duplicate ZIP members")
            provenance=json.loads(archive.read("provenance.json"))
            images=provenance["source_files" if key=="yolo" else "sample_hashes"]
            if any(len(item["sha256"])!=64 for item in images):
                raise ValueError("Malformed original image SHA256")
            figure_names=[f for f in members if f.startswith("Figures/") and f.endswith(".png")]
            if len(figure_names)!=(5 if key=="yolo" else 6):
                raise ValueError("Unexpected original output figure count")
            for f in figure_names:
                if f[:-4]+".pdf" not in members or f[:-4]+".svg" not in members:
                    raise ValueError("Original result figure missing PDF/SVG")
            result={"archive_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "source_image_files":len(images),
              "unique_source_image_sha256":len({im["sha256"] for im in images}),
              "original_png_figure_count":len(figure_names),
              "checkpoint_sha256":provenance["checkpoint_sha256"]}
            if key=="yolo":
                predictions=table(archive,"predictions.csv")
                byclass=table(archive,"per_class_metrics.csv")
                metrics=metric_dict(table(archive,"summary_metrics.csv"))
                if len(images)!=24 or len(predictions)!=2589 or metrics["n_original_gt"]!=972:
                    raise ValueError("Incorrect original VisDrone counts")
                if sum(int(r["true_positives"]) for r in byclass)!=322:
                    raise ValueError("Original class true positives do not reconcile")
                if sum(int(r["original_gt_count"]) for r in byclass)!=972:
                    raise ValueError("Original class ground-truth count mismatch")
                nonempty=[r for r in byclass if int(r["original_gt_count"])>0]
                average=sum(float(r["local_ap50"]) for r in nonempty)/len(nonempty)
                if len(nonempty)!=5 or not math.isclose(average,metrics["local_macro_AP50"],abs_tol=1e-12):
                    raise ValueError("Original macro AP source mismatch")
                names={r["file"] for r in images}
                if any(r["image"] not in names for r in predictions):
                    raise ValueError("Prediction with nonexistent original image")
                if any(not (.01<=float(r["confidence"])<=1 and float(r["x2"])>float(r["x1"]) and float(r["y2"])>float(r["y1"])) for r in predictions):
                    raise ValueError("Invalid measured detection geometry/confidence")
                grouped=Counter(im["sha256"] for im in images)
                repeated=[[im["file"] for im in images if im["sha256"]==digest] for digest,count in grouped.items() if count>1]
                if len(repeated)!=1 or len(repeated[0])!=2:
                    raise ValueError("Expected documented one duplicated VisDrone image content")
                result.update(n_pred=len(predictions),n_gt=972,n_tp=322,
                  local_macro_ap50=average,identical_image_file_pairs=repeated)
            else:
                predictions=table(archive,"detections.csv")
                metrics=metric_dict(table(archive,"metrics.csv"))
                if len(images)!=170 or len({i["sha256"] for i in images})!=170:
                    raise ValueError("Penn-Fudan image source count/uniqueness mismatch")
                if len(predictions)!=1199 or metrics["ground_truth_persons"]!=423 or metrics["true_positives"]!=417:
                    raise ValueError("Penn-Fudan original observed count mismatch")
                if any(r["image"] not in {im["image"] for im in images} for r in predictions):
                    raise ValueError("Prediction from unknown original Penn-Fudan photograph")
                if any(not (.01<=float(r["score"])<=1 and float(r["x2"])>float(r["x1"]) and float(r["y2"])>float(r["y1"])) for r in predictions):
                    raise ValueError("Invalid original pedestrian detections")
                if not math.isclose(417/423,metrics["recall_at_last_prediction"],abs_tol=1e-12):
                    raise ValueError("Recorded recall inconsistent with individual matched counts")
                if not math.isclose(417/1199,metrics["precision_at_last_prediction"],abs_tol=1e-12):
                    raise ValueError("Recorded precision inconsistent with observations")
                result.update(n_pred=len(predictions),n_gt=423,n_tp=417,
                  local_ap50=metrics["local_ap50"],
                  limitation="Original GT labels are not included in the artifact ZIP; independent local AP50 recomputation requires authentic source masks")
            results[key]=result
    return results

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--yolo",required=True,type=Path)
    parser.add_argument("--penn",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    report=inspect(args.yolo,args.penn)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
