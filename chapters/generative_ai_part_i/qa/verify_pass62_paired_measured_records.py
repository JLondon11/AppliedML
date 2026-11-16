"""Verify authentic source 217-patch paired tissue errors and disclose oracle information.

Operates on numerical rows from the actually executed VAE checkpoint, not fixtures.
This is a source-integrity and numerical check, not publication approval.
"""
import argparse,csv,json,math
from pathlib import Path
def verify(records,summary):
    with Path(records).open(newline="",encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    report=json.loads(Path(summary).read_text())
    if len(rows)!=217:raise ValueError("Missing original test patch records")
    if {int(r["test_index"]) for r in rows}!=set(range(217)):
        raise ValueError("Incomplete original test sample identifiers")
    if len({(r["source_row"],r["source_col"]) for r in rows})!=217:
        raise ValueError("Repeated held-out source-pixel locations")
    if not all(384<=int(r["source_row"])<=480 and 0<=int(r["source_col"])<=480 for r in rows):
        raise ValueError("Original held-out source region violated")
    mean=lambda k:sum(float(r[k]) for r in rows)/len(rows)
    vae=mean("vae_mse")
    oracle=mean("nearest_real_train_patch_mse")
    delta=mean("nearest_minus_vae_mse")
    if not math.isclose(vae,.010481160134077072,abs_tol=1e-8):
        raise ValueError("Source original model error differs")
    if not math.isclose(oracle,.010597637985416684,abs_tol=1e-8):
        raise ValueError("True real-image oracle baseline error differs")
    if not math.isclose(delta,oracle-vae,abs_tol=1e-12):
        raise ValueError("Original paired delta does not reconcile")
    vwins=sum(float(r["nearest_minus_vae_mse"])>0 for r in rows)
    owins=sum(float(r["nearest_minus_vae_mse"])<0 for r in rows)
    if (vwins,owins)!=(92,125):
        raise ValueError("Original 217 held-out patch counts changed")
    if report["n_original_source_images"]!=1 or report["test_patches"]!=217:
        raise ValueError("Correlated single-photo study must not claim patients")
    if "oracle" not in report["baseline_selection"].lower():
        raise ValueError("Target-aware reference limitation must be disclosed")
    return {"status":"PASS_REAL_SINGLE_TISSUE_MEASURED_RECORD_CONSISTENCY",
      "rows":len(rows),"source_photographs":1,"vae_lower_error":vwins,
      "oracle_lower_error":owins,"vae_mean_mse":vae,"oracle_mean_mse":oracle,
      "production_approved":False,
      "limitation":"Target-aware nearest training patch is an oracle, not a fair deployable reference; 217 source patches are correlated."}
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--records",required=True,type=Path)
    parser.add_argument("--summary",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.records,args.summary),indent=2))
