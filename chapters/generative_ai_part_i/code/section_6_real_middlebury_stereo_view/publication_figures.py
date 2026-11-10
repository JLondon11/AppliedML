"""Produce publication figures from the genuine stereo pair and saved CVAE checkpoint.
Model predictions are sampled from latent prior, never conditioned on the test target.
"""
from pathlib import Path
import json,numpy as np,torch
from skimage import data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import CondVAE,patchify,SEED
root=Path(__file__).resolve().parent/'results';figs=root/'figures';figs.mkdir(exist_ok=True,parents=True)
left,right,disp=data.stereo_motorcycle()
L=left.astype('float32')/255;R=right.astype('float32')/255
m=json.loads((root/'manifest.json').read_text())
ck=torch.load(root/'conditional_stereo_vae.pt',map_location='cpu',weights_only=False)
model=CondVAE();model.load_state_dict(ck['state']);model.eval();torch.set_num_threads(2)
te_l,te_r,coords=patchify(L,R,400,500)
outputs=[]
with torch.no_grad():
 for start in range(0,len(te_l),32):
  x=te_l[start:start+32];gen=torch.Generator().manual_seed(SEED+start+900)
  outputs.append(torch.stack([model.decode(x,torch.randn((len(x),8),generator=gen))
                              for j in range(8)]).mean(0))
pred=torch.cat(outputs).numpy().transpose(0,2,3,1)
x=te_l.numpy().transpose(0,2,3,1);y=te_r.numpy().transpose(0,2,3,1)
gradient=np.abs(np.diff(x,axis=2)).mean((1,2,3))+np.abs(np.diff(x,axis=1)).mean((1,2,3))
chosen=[]
for i in np.argsort(gradient)[::-1]:
 if all(abs(coords[i][0]-coords[j][0])>=16 or abs(coords[i][1]-coords[j][1])>=64 for j in chosen):chosen.append(int(i))
 if len(chosen)==6:break
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(11.8,4.2),layout='constrained')
for ax,im,title in zip(axes[:2],(left,right),('Real left camera','Real right camera (target)')):
 ax.imshow(im);ax.axhline(320,color='#F6C35B',lw=2);ax.axhline(400,color='#4ACCBF',lw=2)
 ax.set(title=title);ax.set_axis_off()
D=np.ma.masked_invalid(disp)
im=axes[2].imshow(D,cmap='viridis');axes[2].set(title='Ground-truth stereo disparity');axes[2].set_axis_off()
fig.colorbar(im,ax=axes[2],fraction=.046,pad=.03,label='Disparity (pixels)')
fig.suptitle('Middlebury 2014: one genuine calibrated stereo scene',fontsize=12,fontweight='bold')
fig.text(.51,.05,'Row boundaries: training 0–319  |  validation 320–399  |  spatial test 400–499',ha='center',fontsize=8.5,color='#445767')
for ext in ('png','pdf'):fig.savefig(figs/('stereo_real_scene_split.'+ext),dpi=210,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(4,6,figsize=(11,7.1),constrained_layout=True)
for j,i in enumerate(chosen):
 for row,im in enumerate([x[i],y[i],pred[i],np.abs(pred[i]-y[i])]):
  ax=axs[row,j];ax.imshow(im if row!=3 else im/max(.6,im.max()),vmin=0,vmax=1);ax.set_axis_off()
 axs[0,j].set_title(f'Test ({coords[i][0]}, {coords[i][1]})',fontsize=8)
for row,label in enumerate(['Real left input','Real right target','Generated right view','Absolute RGB error']):
 axs[row,0].text(-.08,.5,label,rotation=90,ha='right',va='center',fontsize=8.5,
                 transform=axs[row,0].transAxes,color='#315769')
fig.suptitle('Edge-rich held-out patches selected using left-image gradients only',fontsize=11.3,fontweight='bold')
for ext in ('png','pdf'):fig.savefig(figs/('stereo_edge_rich_outputs.'+ext),dpi=190,bbox_inches='tight')
plt.close(fig)
print('PASS: publication figures generated from genuine Middlebury images and learned model, selected patches',chosen)
