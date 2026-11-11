"""BDD100K runner invariant tests using FAKE FIXTURES ONLY.

These are input-integrity unit tests, NOT scientific driving experiments.
No trained model, accuracy, or generated image may be reported from fixtures.
"""
import json,tempfile
from pathlib import Path
import numpy as np
from run import CLASSES,split_data,read_crops

def fixtures(n,prefix,rng):
    return [(f"{prefix}{i}.jpg",cls,rng.integers(0,256,(64,64,3),dtype=np.uint8))
            for i in range(n) for cls in CLASSES]

def main():
    rng=np.random.default_rng(20261111)
    tr,val,te=split_data(fixtures(180,"train_",rng),fixtures(45,"test_",rng))
    assert len(tr)+len(val)==540 and len(te)==135
    assert not {x[0] for x in tr}&{x[0] for x in val}
    print("PASS: same-frame crops grouped and genuine-frame splits disjoint")
    try:
        split_data(fixtures(180,"shared_",rng),fixtures(45,"shared_",rng))
    except (ValueError,AssertionError):print("PASS: overlapping frame IDs rejected")
    else:raise AssertionError("overlapping frame IDs wrongly accepted")
    training=fixtures(180,"train_",rng);test=fixtures(45,"test_",rng)
    test[0]=(test[0][0],test[0][1],training[0][2].copy())
    try:split_data(training,test)
    except ValueError as err:
        assert "Identical" in str(err) or "pixel-identical" in str(err)
        print("PASS: cross-split exact duplicate images rejected")
    else:raise AssertionError("duplicate images wrongly accepted")
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp);file=base/"annotations.json"
        file.write_text(json.dumps([{"name":"not_a_real_frame.jpg","labels":[
          {"category":"car","box2d":{"x1":0,"y1":0,"x2":80,"y2":80}}]}]))
        try:read_crops(file,base,CLASSES,500,11)
        except FileNotFoundError:print("PASS: missing labeled real photos rejected")
        else:raise AssertionError("missing genuine photos wrongly accepted")
    print("PASS: integrity gates ONLY; no real driving benchmark performed")

if __name__=="__main__":main()
