"""Check section 8.8.3 generated evidence."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parent/"results"
manifest=json.loads((root/"manifest.json").read_text())
assert manifest["seed"]==20261106
assert manifest["steps_per_model"]==120
assert set(manifest["source_hashes"])=={"brick","gravel","grass"}
with (root/"metrics.csv").open() as fh: rows=list(csv.DictReader(fh))
assert len(rows)==6
for name in ("brick","gravel","grass"):
    assert sorted(int(r["augment_horizontal_flip"]) for r in rows if r["texture"]==name)==[0,1]
    for flag,suffix in ((0,"none"),(1,"flip")):
        x=np.load(root/f"generated_{name}_{suffix}.npy")
        assert x.shape==(100,32,32) and np.isfinite(x).all()
        ckpt=torch.load(root/f"generator_{name}_{suffix}.pt",weights_only=True,map_location="cpu")
        assert ckpt["steps"]==120 and ckpt["augmentation"]==bool(flag)
with (root/"losses.csv").open() as fh:
    assert len(list(csv.DictReader(fh)))==720
assert all((root/"figures"/name).stat().st_size>20000 for name in
           ("augmentation_ablation_metrics.png","augmentation_ablation_samples.png"))
print("PASS: six checkpoints, six sample arrays, results, logs and figures")
