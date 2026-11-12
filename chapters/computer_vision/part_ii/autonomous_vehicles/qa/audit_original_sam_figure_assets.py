"""Source-archive inventory for all real SAM and Mask R-CNN figure assets.

Validates ZIP figure-file membership, PNG contents, dimensions, SHA-256 and
matching PDF/SVG companion existence. Never substitutes for model inference
or actual manuscript PDF review.
"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from PIL import Image


def audit(original_archives, output):
    rows = []
    source_hashes = {}
    for label, archive in original_archives.items():
        archive = Path(archive)
        source_hashes[label] = hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as zip_file:
            filenames = zip_file.namelist()
            if len(set(filenames)) != len(filenames):
                raise ValueError(f"Duplicated archive members: {label}")
            listed = set(filenames)
            pngs = sorted(f for f in filenames if f.startswith("Figures/") and f.endswith(".png"))
            if not pngs:
                raise ValueError(f"No original real-result figures in {label}")
            for name in pngs:
                actual = zip_file.read(name)
                im = Image.open(io.BytesIO(actual))
                im.verify()
                width, height = Image.open(io.BytesIO(actual)).size
                prefix = name[:-4]
                if prefix+".pdf" not in listed or prefix+".svg" not in listed:
                    raise ValueError(f"Original figure companion missing: {label}: {name}")
                rows.append(dict(archive=label,filename=name,
                    sha256=hashlib.sha256(actual).hexdigest(),width_px=width,
                    height_px=height,pdf_present=True,svg_present=True))
    if len(rows) != 24:
        raise ValueError(f"Expected 24 recorded original PNG figures, received {len(rows)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"original_figure_asset_manifest.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report=dict(original_png_figure_count=len(rows),
        by_experiment={label:sum(row["archive"]==label for row in rows) for label in original_archives},
        original_archive_sha256=source_hashes,
        original_min_png_dimensions=[min(r["width_px"] for r in rows),
                                     min(r["height_px"] for r in rows)],
        note="Original image output audit only. No new inference. Manuscript captions/print-size and pixelwise mask correctness require independent inspection.")
    (output/"original_figure_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("sam-18","sam-24","maskrcnn-18"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit({"sam_18":args.sam_18,"sam_24":args.sam_24,
           "maskrcnn_18":args.maskrcnn_18},args.output)
