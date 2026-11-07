"""Inspect public KITTI 000008 calibration metadata and original image safely.

Debugging only; this does not claim any fusion-model predictions.
"""
import io
import json
import pickle
import urllib.request
import numpy as np
import cv2

base="https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/demo/data/kitti/"
for name in ("000008.png","000008.pkl","000008.txt"):
    url=base+name
    try:
        with urllib.request.urlopen(url,timeout=30) as reply:
            raw=reply.read()
        print("FILE",name,"BYTES",len(raw),flush=True)
        if name.endswith(".png"):
            arr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            print("IMAGE_SHAPE",arr.shape if arr is not None else None,flush=True)
        elif name.endswith(".pkl"):
            # Input is from exact official source repository, not arbitrary user pickle.
            obj=pickle.loads(raw)
            print("PICKLE_TYPE",type(obj).__name__,flush=True)
            if isinstance(obj,dict):
                print("TOP_KEYS",list(obj),flush=True)
                for key,val in obj.items():
                    if isinstance(val,dict):
                        print("DICT",key,list(val),flush=True)
                        for k,v in val.items():
                            if isinstance(v,(list,tuple,np.ndarray)):
                                vv=np.asarray(v)
                                print("ARRAY",key,k,"shape",vv.shape,"dtype",str(vv.dtype),flush=True)
                    elif isinstance(val,(list,tuple,np.ndarray)):
                        vv=np.asarray(val)
                        print("ARRAY",key,"shape",vv.shape,"dtype",str(vv.dtype),flush=True)
        else:
            print("TEXT",raw[:750].decode("utf-8",errors="replace"),flush=True)
    except Exception as exc:
        print("ERROR",name,type(exc).__name__,str(exc)[:500],flush=True)
