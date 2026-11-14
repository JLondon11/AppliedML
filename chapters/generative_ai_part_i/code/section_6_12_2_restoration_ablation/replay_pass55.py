"""Reproduce and scientifically verify the original-photo Pass55 experiment.

Run: python replay_pass55.py --output results
The underlying complete actual PyTorch model implementation is run.py.
All source, verification, and figure-generation code is committed in this section.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from verify_scoped_restoration import audit

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=ROOT/"results")
    args=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/"run.py"),"--steps","65","--output",str(args.output)],check=True)
    evidence=audit(args.output)
    subprocess.run([sys.executable,str(ROOT/"plot_measured_restoration.py"),
                    str(args.output/"metrics.csv")],check=True)
    print(json.dumps(evidence,indent=2))

if __name__=="__main__":
    main()
