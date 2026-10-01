#!/usr/bin/env python3
"""Build a publication-ready chapter ZIP from a final chapter directory."""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, sys, zipfile
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("chapter_dir")
    ap.add_argument("--tex",required=True,help="final chapter tex filename relative to chapter_dir")
    ap.add_argument("--pdf",required=True,help="compiled chapter pdf filename relative to chapter_dir")
    ap.add_argument("--out-dir",default="book/releases/chapters")
    args=ap.parse_args()

    root=Path(args.chapter_dir).resolve()
    tex=(root/args.tex).resolve()
    pdf=(root/args.pdf).resolve()
    if not root.exists(): raise SystemExit(f"missing chapter dir: {root}")
    if not tex.exists(): raise SystemExit(f"missing final tex: {tex}")
    if not pdf.exists(): raise SystemExit(f"missing compiled pdf: {pdf}")

    figs=root/"figures"
    if not figs.exists(): raise SystemExit("missing figures directory")
    ext_counts={e:len(list(figs.rglob(f"*.{e}"))) for e in ("png","svg","pdf")}
    if min(ext_counts.values())<1:
        raise SystemExit(f"missing required figure format(s): {ext_counts}")

    # Require matching figure stems across all three required formats.
    stems={e:{p.relative_to(figs).with_suffix("") for p in figs.rglob(f"*.{e}")} for e in ("png","svg","pdf")}
    common=stems["png"] & stems["svg"] & stems["pdf"]
    union=stems["png"] | stems["svg"] | stems["pdf"]
    if common != union:
        missing={}
        for stem in sorted(union):
            have=[e for e in ("png","svg","pdf") if stem in stems[e]]
            if len(have)!=3: missing[str(stem)]=have
        raise SystemExit(f"figure tri-format mismatch: {missing}")

    code=root/"code"
    if not code.exists(): raise SystemExit("missing code directory")

    manifest=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel=p.relative_to(root)
        if rel.parts and rel.parts[0] in {".git","__pycache__"}: continue
        manifest.append({"path":str(rel),"bytes":p.stat().st_size,"sha256":sha256(p)})

    (root/"PACKAGE_MANIFEST.json").write_text(json.dumps({
        "chapter_directory":root.name,
        "final_tex":str(tex.relative_to(root)),
        "compiled_pdf":str(pdf.relative_to(root)),
        "figure_counts":ext_counts,
        "files":manifest,
    },indent=2),encoding="utf-8")

    safe=re.sub(r"[^A-Za-z0-9._-]+","_",root.name)
    outdir=Path(args.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    zpath=outdir/f"{safe}_PUBLICATION_READY.zip"
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            rel=Path(root.name)/p.relative_to(root)
            z.write(p,rel.as_posix())

    checksum=sha256(zpath)
    zpath.with_suffix(zpath.suffix+".sha256").write_text(f"{checksum}  {zpath.name}\n")
    print(json.dumps({"zip":str(zpath),"sha256":checksum,"figure_counts":ext_counts},indent=2))

if __name__=="__main__":
    main()
