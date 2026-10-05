#!/usr/bin/env python3
"""Detect exact and near-duplicate figures within a chapter figure directory.

Exact duplicates are detected by SHA-256. Near duplicates are detected from
rendered PNGs using perceptual hash (pHash) and normalized image correlation.
"""
from __future__ import annotations
import argparse, hashlib, itertools, sys
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def phash(path: Path, size=32, low=8):
    img=Image.open(path).convert("L").resize((size,size),Image.Resampling.LANCZOS)
    a=np.asarray(img,dtype=float)
    # 2D DCT using separable cosine basis, avoiding scipy dependency.
    n=size
    x=np.arange(n)
    k=np.arange(n)[:,None]
    C=np.cos(np.pi*(2*x+1)*k/(2*n))
    C[0,:] *= 1/np.sqrt(2)
    C *= np.sqrt(2/n)
    d=C@a@C.T
    block=d[:low,:low]
    med=np.median(block[1:,:])
    return (block>med).ravel()

def corr(path: Path, size=128):
    img=Image.open(path).convert("L")
    img=ImageOps.autocontrast(img)
    img=img.resize((size,size),Image.Resampling.LANCZOS)
    a=np.asarray(img,dtype=float).ravel()
    a=(a-a.mean())/(a.std()+1e-12)
    return a

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("figure_dir")
    ap.add_argument("--phash-distance",type=int,default=8)
    ap.add_argument("--corr-threshold",type=float,default=0.985)
    args=ap.parse_args()
    root=Path(args.figure_dir)
    files=sorted(root.glob("*.png"))
    if not files:
        print("no PNG figures found")
        return 1

    exact=[]
    sha_map={}
    for p in files:
        s=sha256(p)
        sha_map.setdefault(s,[]).append(p)
    for group in sha_map.values():
        if len(group)>1:
            exact.append(group)

    hashes={p:phash(p) for p in files}
    arrays={p:corr(p) for p in files}
    near=[]
    for a,b in itertools.combinations(files,2):
        hd=int(np.count_nonzero(hashes[a]!=hashes[b]))
        cc=float(np.dot(arrays[a],arrays[b])/len(arrays[a]))
        if hd<=args.phash_distance or cc>=args.corr_threshold:
            near.append((a,b,hd,cc))

    print(f"png_files={len(files)}")
    print(f"exact_duplicate_groups={len(exact)}")
    for g in exact:
        print("EXACT:", " | ".join(str(x) for x in g))
    print(f"near_duplicate_pairs={len(near)}")
    for a,b,hd,cc in near:
        print(f"NEAR: {a.name} | {b.name} | phash_distance={hd} | corr={cc:.6f}")
    return 1 if exact or near else 0

if __name__=="__main__":
    raise SystemExit(main())
