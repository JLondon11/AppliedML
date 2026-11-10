"""Figure 11: Physics-regularization test against REAL laboratory thermocouple data.

The original Figure 11 simulated heat-equation targets. This version analyzes
actual temperatures measured at eight positions along an electrically heated
aluminum rod, rather than claiming synthetic values are observations.

Data: TariqSawaqed1/Heat-Conduction-Aerospace-Lab
      Part 1/Aluminum_26V_250mA.txt, experimental laboratory log.
Run: python figure_11_pinn_residual.py --data Aluminum_26V_250mA.txt
Optional --output-dir DIR --epochs 600

Simplified diffusion regularization is tested, not assumed correct. In this
specific experiment it reduces held-out interpolation accuracy; fitted alpha
is dimensionless and NOT a measured material thermal diffusivity.
"""
import argparse, copy, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED=29
TRAIN=(0,1,3,4,6,7)
VALID=2
TEST=5
LAMBDA=(0.,0.002,0.02)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(2,32),nn.Tanh(),
             nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,z):
        return self.model(z)

def load_measurements(path):
    # No internal synthetic fallback. Absence of original data is an error.
    array=np.genfromtxt(path,delimiter=",",skip_header=1,dtype=np.float64)
    if array.ndim!=2 or array.shape[1]!=9 or array.shape[0]<100:
        raise ValueError("Expected real eight-channel thermocouple log")
    time,temperature=array[:,0],array[:,1:]
    if not np.isfinite(array).all() or not np.allclose(np.diff(time),10):
        raise ValueError("Unrecognized measured timestamps or nonfinite values")
    if not np.isclose(time[0],0):
        raise ValueError("Time index should start at zero seconds")
    return time,temperature

def experiment(source,output,epochs=600):
    torch.set_num_threads(2)
    time,temperature=load_measurements(source)
    count=len(time); output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Sensor geometry is 0.5 inches per channel, but the absolute sensor
    # position relative to heater is not re-inferred from temperatures.
    x=np.arange(8)/7
    t=(time-time[0])/(time[-1]-time[0])
    coord=np.stack(np.meshgrid(x,t),axis=-1)
    mean=float(temperature[:,TRAIN].mean())
    scale=float(temperature[:,TRAIN].std())
    Y=(temperature-mean)/scale
    train_x=torch.tensor(coord[:,TRAIN].reshape(-1,2),dtype=torch.float32)
    train_y=torch.tensor(Y[:,TRAIN].reshape(-1,1),dtype=torch.float32)
    grid=torch.tensor(coord.reshape(-1,2),dtype=torch.float32)
    results=[];predictions={};histories={}
    for weight in LAMBDA:
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        model=Net()
        log_alpha=nn.Parameter(torch.tensor(-3.0))
        opt=torch.optim.Adam([*model.parameters(),log_alpha],lr=0.002)
        history=[]
        for epoch in range(1,epochs+1):
            model.train();opt.zero_grad()
            z=train_x.clone().requires_grad_(weight>0)
            fit=model(z)
            data_loss=torch.mean((fit-train_y)**2)
            if weight>0:
                grad=torch.autograd.grad(fit,z,torch.ones_like(fit),create_graph=True)[0]
                ut=grad[:,1:2]
                uxx=torch.autograd.grad(
                    grad[:,0:1],z,torch.ones_like(grad[:,0:1]),
                    create_graph=True)[0][:,0:1]
                alpha=torch.nn.functional.softplus(log_alpha)
                physics_loss=torch.mean((ut-alpha*uxx)**2)
                loss=data_loss+weight*physics_loss
            else:
                physics_loss=None
                loss=data_loss
            loss.backward();opt.step()
            if epoch==1 or epoch%50==0 or epoch==epochs:
                history.append({"epoch":epoch,
                    "train_normalized_mse":float(data_loss.detach()),
                    "physics_normalized_residual_mse":
                        float(physics_loss.detach()) if weight>0 else None})
        model.eval()
        with torch.no_grad():
            estimates=model(grid).numpy().reshape(count,8)*scale+mean
        predictions[str(weight)]=estimates.tolist()
        histories[str(weight)]=history
        rmse=lambda channels:float(np.sqrt(np.mean(
            (temperature[:,channels]-estimates[:,channels])**2)))
        results.append({"physics_weight":weight,"train_rmse_c":rmse(list(TRAIN)),
            "validation_rmse_c":rmse(VALID),
            "heldout_test_rmse_c":rmse(TEST),
            "alpha_dimensionless":float(torch.nn.functional.softplus(
                log_alpha.detach())) if weight>0 else None,
            "epochs":epochs})
        print(results[-1],flush=True)
    choice=min(results,key=lambda r:r["validation_rmse_c"])["physics_weight"]
    base=np.asarray(predictions["0.0"])
    selected=np.asarray(predictions[str(choice)])
    record={
      "dataset":"Original measured heated aluminum rod thermocouple temperatures",
      "origin":"TariqSawaqed1/Heat-Conduction-Aerospace-Lab",
      "original_file":"Part 1/Aluminum_26V_250mA.txt",
      "original_github_blob_sha":"968990e38e4fe0a1f1b981d117d8bf92a90f163c",
      "source_url":"https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab",
      "recorded_times_s":time.tolist(),"sensor_spacing_m":0.0127,
      "recorded_temperatures_c":temperature.tolist(),
      "n_recorded_times":count,"n_recorded_sensors":8,
      "n_original_measurements":int(temperature.size),
      "train_channels":["CH1","CH2","CH4","CH5","CH7","CH8"],
      "validation_channel":"CH3","test_channel":"CH6",
      "training_observations":count*6,"validation_observations":count,
      "testing_observations":count,
      "selection_protocol":"Select regularization strength by held-out validation sensor only",
      "model":"2-input 32x32 tanh MLP with optional diffusion residual",
      "physics_assumption":"Simplified u_t = alpha*u_xx on normalized coordinates at observed training sensor/time sites; heating boundary and convection not modeled",
      "physical_limitation":"Fitted alpha is dimensionless and not identifiable as actual aluminum thermal diffusivity; test evaluates one unseen sensor from same run",
      "random_seed":SEED,"epochs":epochs,
      "normalization":"Fit on six training thermocouple channels only",
      "physics_weights":list(LAMBDA),
      "results":results,"selected_weight_by_validation":choice,
      "predictions_c_by_weight":predictions,"loss_histories":histories
    }
    (output/"results.json").write_text(json.dumps(record,indent=2)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,
       "axes.spines.top":False,"axes.spines.right":False})
    navy="#233e57";teal="#337a70";copper="#B67852"
    fig,ax=plt.subplots(2,2,figsize=(12.1,7.9),layout="constrained")
    im=ax[0,0].imshow(temperature.T,origin="lower",aspect="auto",
      extent=(time[0],time[-1],0.5,8.5),cmap="cividis")
    ax[0,0].set(title="(a) Actual temperatures: 8 thermocouples",
      xlabel="Time (s)",ylabel="Sensor channel",yticks=np.arange(1,9))
    fig.colorbar(im,ax=ax[0,0],label="Observed temperature (°C)")
    ax[0,1].plot(time,temperature[:,TEST],color=navy,lw=2.5,label="Observed test CH6")
    for w,c,style in [(0.,teal,"-"),(.002,copper,"--"),(.02,"#78678c",":")]:
        ax[0,1].plot(time,np.asarray(predictions[str(w)])[:,TEST],
            color=c,ls=style,lw=1.6,label=f"MLP + λ={w:g}")
    ax[0,1].set(title="(b) Held-out sensor interpolation",xlabel="Time (s)",ylabel="Temperature (°C)")
    ax[0,1].legend(frameon=False,fontsize=8)
    ax[1,0].plot(time,temperature[:,VALID],color=navy,lw=2.2,label="Observed validation CH3")
    ax[1,0].plot(time,selected[:,VALID],color=teal,lw=1.8,label=f"Selected λ={choice:g}")
    ax[1,0].set(title="(c) Independent validation channel",xlabel="Time (s)",ylabel="Temperature (°C)")
    ax[1,0].legend(frameon=False,fontsize=8)
    for r,c in zip(results,[navy,copper,teal]):
        ax[1,1].scatter(r["validation_rmse_c"],r["heldout_test_rmse_c"],
           s=90,c=c,label=f'λ={r["physics_weight"]:g}')
    ax[1,1].set(title="(d) Validation–test error tradeoff",
      xlabel="CH3 validation RMSE (°C)",ylabel="CH6 test RMSE (°C)")
    ax[1,1].legend(frameon=False)
    for a in [ax[0,1],ax[1,0],ax[1,1]]:a.grid(alpha=.15)
    for ext in ["png","pdf","svg"]:
        fig.savefig(output/f"figure_11_pinn_residual_field_error.{ext}",
            dpi=230,bbox_inches="tight")
    plt.close(fig)
    return record

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,
       default=Path(__file__).resolve().parent/"outputs")
    parser.add_argument("--epochs",type=int,default=600)
    args=parser.parse_args()
    experiment(args.data,args.output_dir,args.epochs)
