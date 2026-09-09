"""Core computations for Scientific AI Figure 6.

The production figure combines: (a) Morgan-fingerprint chemical-space projection
from Drug Repurposing Hub structures, (b) RDKit structures and Tanimoto novelty
for halicin comparators, and (c) published Stokes et al. MIC values.
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
HALICIN = "[O-][N+](=O)c1cnc(s1)Sc1nnc(s1)N"
NITHIAMIDE = "CC(=O)NC1=NC=C(S1)[N+]([O-])=O"
METRONIDAZOLE = "CC1=NC=C(N1CCO)[N+](=O)[O-]"

def tanimoto(a,b):
    return TanimotoSimilarity(GEN.GetFingerprint(Chem.MolFromSmiles(a)),
                              GEN.GetFingerprint(Chem.MolFromSmiles(b)))
print("halicin–nithiamide", tanimoto(HALICIN,NITHIAMIDE))
print("halicin–metronidazole", tanimoto(HALICIN,METRONIDAZOLE))
mic = pd.Series({"E. coli BW25113":2.0, "M. tuberculosis H37Rv":16.0, "C. difficile 630":0.5})
print(mic)
