"""Verify all saved real NCI SMILES-VAE outputs after run.py."""
from pathlib import Path
import csv,json,hashlib
from collections import Counter
from rdkit import Chem,RDConfig
ROOT=Path(__file__).resolve().parent/"results"
def rows(name):
 with (ROOT/name).open() as f:return list(csv.DictReader(f))
def main():
 m=json.loads((ROOT/"metrics.json").read_text())
 source=Path(RDConfig.RDDataDir)/"NCI"/"first_5K.smi"
 assert hashlib.sha256(source.read_bytes()).hexdigest()==m["source_sha256"]
 entries=rows("source_manifest.csv")
 assert len(entries)==600
 assert Counter(r["split"] for r in entries)=={"train":360,"validation":120,"test":120}
 test=rows("heldout_reconstructions.csv");prior=rows("prior_samples.csv")
 assert len(test)==120 and len(prior)==1024
 assert abs(sum(int(r["valid"]) for r in test)/120-m["test_validity"])<1e-10
 assert abs(sum(int(r["exact_reconstruction"]) for r in test)/120-m["test_exact_reconstruction"])<1e-10
 assert abs(sum(float(r["morgan_tanimoto"]) for r in test)/120-m["test_mean_morgan_tanimoto"])<1e-10
 good=[r["canonical_smiles"] for r in prior if int(r["valid"])]
 assert len(good)==36 and len(set(good))==23
 assert all(Chem.MolFromSmiles(s) is not None for s in good)
 assert (ROOT/"smiles_vae.pt").stat().st_size>500000
 for f in ("molecular_smiles_generation_results.png","molecular_smiles_vae_training.png"):
  assert (ROOT/"figures"/f).stat().st_size>10000
 print("PASS: 600 unique real NCI molecules, disjoint split, 120 test predictions, 1024 prior strings")
if __name__=="__main__":main()
