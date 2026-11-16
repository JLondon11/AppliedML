"""Real held-out tissue VAE versus oracle-selected nearest original training patches.

Runs actual checkpoint inference on original immunohistochemistry pixels and
exports all 217 measured paired errors plus original/model/oracle image plate.
The oracle uses the known test target pixels and is NOT a deployable predictor.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from skimage import data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE_SHA="c5b3ef509a92f16d4c29be8cf0300fe75d53e13a3ce650159db932caea8dcc1b"
class VAE(nn.Module):
    def __init__(self,z=24):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.ReLU(),
            nn.Conv2d(16,32,4,2,1),nn.ReLU(),nn.Conv2d(32,64,4,2,1),nn.ReLU(),nn.Flatten())
        self.mu=nn.Linear(64*4*4,z);self.lv=nn.Linear(64*4*4,z)
        self.fc=nn.Linear(z,64*4*4)
        self.dec=nn.Sequential(nn.Unflatten(1,(64,4,4)),
            nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(16,3,4,2,1),nn.Sigmoid())
    def encode(self,x):
        h=self.enc(x)
        return self.mu(h),self.lv(h).clamp(-8,8)
    def decode(self,z):
        return self.dec(self.fc(z))

def patches(image,start,stop):
    values=[];locations=[]
    for row in range(start,stop-31,16):
        for col in range(0,512-31,16):
            values.append(np.moveaxis(image[row:row+32,col:col+32],-1,0))
            locations.append((row,col))
    return np.array(values,dtype=np.float32)/255,np.array(locations,dtype=int)

def execute(checkpoint,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    original=data.immunohistochemistry()
    identity=hashlib.sha256(original.tobytes()).hexdigest()
    if identity!=SOURCE_SHA:raise ValueError("Source differs from authentic observed photo")
    training,_=patches(original,0,320)
    test,positions=patches(original,384,512)
    if len(training)!=589 or len(test)!=217:raise ValueError("Invalid original patches")
    checkpoint=Path(checkpoint)
    artifact=torch.load(checkpoint,map_location="cpu",weights_only=True)
    weights=artifact["model"] if "model" in artifact else artifact["weights"]
    model=VAE();model.load_state_dict(weights);model.eval()
    with torch.no_grad():
        observed=torch.from_numpy(test)
        predictions=torch.cat([model.decode(model.encode(batch)[0]) for batch in observed.split(64)]).numpy()
    originals=test.reshape(len(test),-1)
    library=training.reshape(len(training),-1)
    nearest_ids=[];nearest_errors=[]
    for start in range(0,len(test),16):
        targets=originals[start:start+16]
        squared=((targets[:,None,:]-library[None,:,:])**2).mean(2)
        ids=np.argmin(squared,axis=1)
        nearest_ids.extend(ids.tolist())
        nearest_errors.extend(squared[np.arange(len(ids)),ids].tolist())
    nearest=np.asarray(nearest_errors)
    nearest_ids=np.asarray(nearest_ids)
    model_mse=((predictions-test)**2).mean(axis=(1,2,3))
    delta=nearest-model_mse
    if not np.isfinite(delta).all():raise ValueError("Nonfinite real-data error")
    columns=["test_index","source_row","source_col","nearest_train_index",
             "vae_mse","nearest_real_train_patch_mse","nearest_minus_vae_mse","vae_lower_error"]
    with (out/"actual_217_patch_paired_errors.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for i,(row,col) in enumerate(positions):
            writer.writerow(dict(test_index=i,source_row=int(row),source_col=int(col),
                 nearest_train_index=int(nearest_ids[i]),vae_mse=float(model_mse[i]),
                 nearest_real_train_patch_mse=float(nearest[i]),
                 nearest_minus_vae_mse=float(delta[i]),vae_lower_error=bool(delta[i]>0)))
    positive=np.argsort(delta)[-4:][::-1].tolist()
    negative=np.argsort(delta)[:4].tolist()
    chosen=positive+negative
    fig,axes=plt.subplots(4,8,figsize=(15.2,8.3),facecolor="white")
    to_rgb=lambda x:np.moveaxis(x,0,-1)
    for j,i in enumerate(chosen):
        real=to_rgb(test[i]);recon=to_rgb(predictions[i])
        neighbor=to_rgb(training[nearest_ids[i]])
        error=np.abs(real-recon).mean(2)
        for row,image,cmap in ((0,real,None),(1,recon,None),
                               (2,neighbor,None),(3,error,"magma")):
            a=axes[row,j]
            a.imshow(image,cmap=cmap,vmin=0,vmax=(.4 if row==3 else 1),
                     interpolation="nearest")
            a.set_xticks([]);a.set_yticks([])
        axes[0,j].set_title(
            f'{"VAE better" if j<4 else "Oracle better"}\\npatch {i}; Δ={delta[i]:+.4f}',
            fontsize=8.2,color=("#157b82" if j<4 else "#aa6447"))
    labels=["Original measured\\ntissue patch","Real trained VAE\\nreconstruction",
            "Oracle-nearest real\\ntraining patch","VAE absolute RGB\\nreconstruction error"]
    for row,label in enumerate(labels):
        axes[row,0].set_ylabel(label,fontsize=10,labelpad=18)
    fig.suptitle("Real histology · actual VAE reconstructions vs oracle real-image reference",
                 x=.042,y=.976,ha="left",fontsize=16,color="#203f55",weight="bold")
    count=int((delta>0).sum())
    fig.text(.045,.921,
        f"One original micrograph; 217 spatial test patches; VAE lower error on {count}, oracle reference on {int((delta<0).sum())}",
        fontsize=9.8,color="#405f72")
    fig.text(.045,.045,
        "Illustrated extremes are not random samples. Oracle matching uses observed test target pixels (not a deployable predictor).",
        fontsize=9,color="#405f72")
    fig.text(.045,.020,
        "Error: mean absolute RGB difference, fixed 0–0.4 color scale. Not an independent tissue/patient evaluation.",
        fontsize=8.8,color="#405f72")
    fig.subplots_adjust(left=.145,right=.985,top=.83,bottom=.102,wspace=.10,hspace=.32)
    for ext in ("pdf","png","svg"):
        fig.savefig(out/f"actual_tissue_paired_failure_success_atlas.{ext}",
                    dpi=210,facecolor="white")
    plt.close(fig)
    result=dict(source_sha256=identity,source_images=1,
      original_train_patches=len(training),original_test_patches=len(test),
      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      vae_mean_test_mse=float(model_mse.mean()),
      oracle_nearest_mean_test_mse=float(nearest.mean()),
      mean_oracle_minus_vae_mse=float(delta.mean()),
      median_oracle_minus_vae_mse=float(np.median(delta)),
      vae_lower_mse_count=count,oracle_lower_mse_count=int((delta<0).sum()),
      illustrated_indices={"vae_better":positive,"oracle_better":negative},
      baseline_warning="Oracle nearest training patch uses held-out test target pixels; it is a nondeployable descriptor, not a fair operational baseline.",
      scope_warning="217 correlated patches from one source photograph; no patient- or slide-level generalization.")
    (out/"actual_paired_error_summary.json").write_text(json.dumps(result,indent=2)+"\\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();execute(a.checkpoint,a.output)
