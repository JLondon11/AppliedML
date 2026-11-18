"""Unified Chapter 1 capstone orchestrator."""
from project01_optimizer_benchmark import TASKS, run
from dataclasses import asdict
import json

def main():
    results=[]
    for task in TASKS:
        r=run(task); results.append(asdict(r)); print(task,r.status)
    with open('capstone_results.json','w') as f: json.dump(results,f,indent=2)
    failed=[r for r in results if r['status']!='PASS']
    raise SystemExit(1 if failed else 0)
if __name__=='__main__': main()
