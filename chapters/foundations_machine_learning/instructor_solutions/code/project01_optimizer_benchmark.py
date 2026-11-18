"""Configuration-driven benchmark orchestrator for heterogeneous ML optimization tasks."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json, time, subprocess, sys

@dataclass
class RunResult:
    task:str; optimizer:str; status:str; runtime_s:float; output:str

TASKS={
'convex':'ce01_german_credit_optimizers.py','cnn':'ce02_cifar100_optimizers.py','transformer':'ce03_wikitext103_transformer.py',
'pinn':'ce04_burgers_pinn.py','rl':'ce05_ppo_halfcheetah.py'}

def run(task, extra=None):
    script=Path(__file__).with_name(TASKS[task]); cmd=[sys.executable,str(script)]+(extra or []); t=time.perf_counter(); p=subprocess.run(cmd,capture_output=True,text=True)
    return RunResult(task,'script-defined','PASS' if p.returncode==0 else 'FAIL',time.perf_counter()-t,(p.stdout+p.stderr)[-4000:])

def main():
    results=[run(t) for t in TASKS]; Path('project01_results.json').write_text(json.dumps([asdict(r) for r in results],indent=2)); print(results)
if __name__=='__main__': main()
