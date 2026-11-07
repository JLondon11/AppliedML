"""Execute official pretrained fastMRI multicoil knee VarNet, CPU only.

Requires original authorized fastMRI multicoil HDF5 data supplied separately.
This script never substitutes synthetic kspace, phantoms, or paper-reported scores.
Official source: facebookresearch/fastMRI, MIT-licensed implementation.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

OFFICIAL_SCRIPT = "fastmri_examples/varnet/run_pretrained_varnet_inference.py"

def preflight(path):
    files=sorted(path.glob("*.h5"))
    if not files:
        raise FileNotFoundError("Original licensed fastMRI multicoil .h5 data are required")
    info=[]
    for file in files:
        with h5py.File(file,"r") as inp:
            if "kspace" not in inp:
                raise ValueError("Missing original k-space in "+str(file))
            ds=inp["kspace"]
            if ds.ndim != 4 or ds.shape[1]<2 or not np.issubdtype(ds.dtype,np.complexfloating):
                raise ValueError("Expected complex [slice, coil, H, W] kspace in "+str(file))
            info.append({"file":file.name,"shape":list(ds.shape)})
    return info

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--official-fastmri-repository",required=True,type=Path)
    p.add_argument("--original-data-directory",required=True,type=Path)
    p.add_argument("--output-directory",required=True,type=Path)
    p.add_argument("--checkpoint",type=Path,default=None)
    a=p.parse_args()
    original=a.original_data_directory.resolve()
    if not original.is_dir():
        p.error("Authorized original fastMRI multicoil HDF5 directory is missing")
    manifest=preflight(original)
    script=a.official_fastmri_repository.resolve()/OFFICIAL_SCRIPT
    if not script.is_file():
        raise FileNotFoundError("Official fastMRI pretrained VarNet implementation is missing")
    a.output_directory.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(script),"--device","cpu","--challenge","varnet_knee_mc",
         "--data_path",str(original),"--output_path",str(a.output_directory)]
    if a.checkpoint is not None:
        if not a.checkpoint.is_file() or a.checkpoint.stat().st_size<1_000_000:
            raise FileNotFoundError("Official pretrained checkpoint file is unavailable")
        cmd += ["--state_dict_file",str(a.checkpoint.resolve())]
    with (a.output_directory/"varnet_cpu.log").open("w") as logfile:
        subprocess.run(cmd,check=True,cwd=str(a.official_fastmri_repository),stdout=logfile,stderr=subprocess.STDOUT)
    recon=list((a.output_directory/"reconstructions").glob("*.h5"))
    if not recon:
        raise RuntimeError("No real model reconstruction output produced")
    (a.output_directory/"input_manifest.json").write_text(json.dumps({
        "model":"official pretrained fastMRI knee multicoil VarNet",
        "device":"cpu","input_type":"original licensed multicoil HDF5",
        "status":"model inference executed (not official score)",
        "inputs":manifest,"reconstruction_files":[x.name for x in recon]},indent=2)+"\n")
    print("Executed official CPU VarNet on",len(manifest),"original MRI acquisitions")

if __name__=="__main__":
    main()
