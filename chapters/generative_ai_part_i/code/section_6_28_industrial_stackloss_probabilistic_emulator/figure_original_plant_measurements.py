"""Plot the genuine Brownlee ammonia-oxidation plant measurements (Chapter 6 §6.28).

All 21 points are sourced from statsmodels.datasets.stackloss; no invented
values or augmented observations are used.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.datasets import stackloss

root=Path(__file__).resolve().parent
results=root/'results'
results.mkdir(exist_ok=True)
data=stackloss.load_pandas().data[['STACKLOSS','AIRFLOW','WATERTEMP','ACIDCONC']]
actual=data.to_numpy(dtype='<f8')
assert actual.shape==(21,4)
digest=hashlib.sha256(actual.tobytes()).hexdigest()
assert digest=='d999d98d68f6cd1ad64dec41476b32961a63e60a3e81433bca361601b5612e6d'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.4,
                     'axes.edgecolor':'#A8B5C0','axes.labelcolor':'#253746',
                     'xtick.color':'#526473','ytick.color':'#526473',
                     'pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(10.9,3.9),constrained_layout=True)
specs=[('AIRFLOW','Airflow (source units)','#176B79'),
       ('WATERTEMP','Water temperature (source units)','#A96842'),
       ('ACIDCONC','Acid concentration (source units)','#597D72')]
for ax,(col,xlabel,color) in zip(axs,specs):
    ax.scatter(data[col],data.STACKLOSS,s=42,color=color,alpha=.91,
               edgecolor='white',linewidth=.65,zorder=3)
    ax.set(xlabel=xlabel,ylabel='Measured stack loss (source units)')
    ax.grid(color='#E4EBEF',linewidth=.7,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.text(.015,.985,'n = 21 measured records',transform=ax.transAxes,
            fontsize=8,color='#435463',va='top')
for ext in ('pdf','png'):
    fig.savefig(results/('stackloss_real_input_output_relationships.'+ext),
                dpi=230,bbox_inches='tight')
plt.close(fig)
data.to_csv(results/'stackloss_source_observations.csv',index_label='source_record_index')
print('Verified source dataset SHA-256',digest)
