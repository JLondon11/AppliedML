"""Resolve Computer Vision Part I frozen Figures 13-14 from Alam & Islam (2019)."""
from pathlib import Path
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve(); REPO=next(p for p in HERE.parents if (p/'chapters').is_dir())
sys.path.insert(0,str(REPO/'chapters'/'_shared'))
from pass14_closure_utils import extract_html_figure, save_provenance
ROOT=Path(os.environ.get('PASS14_ARTIFACT_ROOT','pass14_artifact'))
CH='01_Computer_Vision_Part_I_Foundations'; OUT=ROOT/CH/'Figures_Production'; OUT.mkdir(parents=True,exist_ok=True)
prov={}
# Published Table 3, Healthcare Technology Letters 2019, DOI 10.1049/htl.2018.5098.
models=['Tiny YOLO','VGG-16','ResNet50','InceptionV3','MobileNet']
rbc=[96.09,72.98,79.80,87.75,74.24]; wbc=[86.89,100.0,95.08,100.0,93.44]; plt_acc=[96.36,90.91,87.27,96.36,83.64]
x=np.arange(len(models)); w=.24
fig,ax=plt.subplots(figsize=(8.6,4.2))
ax.bar(x-w,rbc,w,label='RBC'); ax.bar(x,wbc,w,label='WBC'); ax.bar(x+w,plt_acc,w,label='Platelet')
ax.set_xticks(x,models,rotation=18,ha='right'); ax.set_ylabel('Detection/counting accuracy (%)'); ax.set_ylim(0,105); ax.legend(frameon=False,ncol=3); ax.grid(False)
fig.tight_layout(); p=OUT/'Figure_013_Cell_type_detection_accuracy_using_different_CNN_architectures_with_YO.png'; fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.03); plt.close(fig)
prov['13']={'source':'Alam & Islam 2019 Table 3','doi':'10.1049/htl.2018.5098','values':{'Tiny YOLO':[96.09,86.89,96.36],'VGG-16':[72.98,100.0,90.91],'ResNet50':[79.80,95.08,87.27],'InceptionV3':[87.75,100.0,96.36],'MobileNet':[74.24,93.44,83.64]}}
# Exact published Figure 6 from the open-access article; caption is external to image in PMC.
p=OUT/'Figure_014_Training_loss_curves_for_the_CNN_backbone_architectures_Tiny_YOLO_VGG_.png'
prov['14']=extract_html_figure('https://pmc.ncbi.nlm.nih.gov/articles/PMC6718065/','CNN models loss curves with YOLO algorithm along with their validation mAP',p)
prov['14'].update({'source':'Alam & Islam 2019 Figure 6','doi':'10.1049/htl.2018.5098','license':'CC BY 3.0'})
save_provenance(ROOT/CH/'pass14_provenance.json',prov)
