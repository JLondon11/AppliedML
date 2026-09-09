"""Regenerate all code-rendered Scientific AI figures from this directory."""
import subprocess, sys
SCRIPTS=[
"figure_02_scientific_discovery.py","figure_03_scientific_ai_evolution.py",
"figure_04_data_provenance.py","figure_05_whole_slide_imaging.py",
"figure_06_drug_discovery.py","figure_07_multimodal_oncology.py",
"figure_08_wdbc_baselines.py","figure_09_digital_twin.py",
"figure_10_pinn_workflow.py","figure_11_pinn_residual.py",
"figure_12_retinal_pipeline.py"]
for s in SCRIPTS:
    print("running",s,flush=True)
    subprocess.run([sys.executable,s],check=True)
