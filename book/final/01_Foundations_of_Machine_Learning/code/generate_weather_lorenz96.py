from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="book/final/01_Foundations_of_Machine_Learning/figures")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    N=20; forcing=8.0; dt=.01; steps=3000
    x=np.ones(N)*forcing; x[0]+=0.01
    hist=[]
    for k in range(steps):
        dx=(np.roll(x,-1)-np.roll(x,2))*np.roll(x,1)-x+forcing
        x=x+dt*dx
        if k%10==0:
            hist.append(x.copy())
    hist=np.asarray(hist)
    t=np.arange(len(hist))*dt*10

    fig,axs=plt.subplots(1,2,figsize=(10.0,3.8))
    palette=["#355C7D","#4F7C6E","#B26E3B","#7A6AA6","#657A8A"]
    for i in range(5):
        axs[0].plot(t,hist[:,i],lw=1.1,color=palette[i],label=f"State {i+1}")
    axs[0].set_xlabel("Model time")
    axs[0].set_ylabel("State value")
    axs[0].legend(frameon=False,fontsize=7,ncol=2)
    im=axs[1].imshow(hist.T,aspect="auto",origin="lower",cmap="viridis",
                     extent=[t[0],t[-1],0,N])
    axs[1].set_xlabel("Model time")
    axs[1].set_ylabel("State index")
    cb=fig.colorbar(im,ax=axs[1],fraction=.038,pad=.03)
    cb.set_label("State value")
    for i,ax in enumerate(axs):
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.text(.5,-.22,f"({chr(97+i)})",transform=ax.transAxes,ha="center",va="top",fontsize=10)
    fig.tight_layout(w_pad=1.8,pad=.25)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"ch01_weather_lorenz96.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)

    meta={
      "system":"Lorenz-96",
      "dimension":N,
      "forcing":forcing,
      "time_step":dt,
      "integration_steps":steps,
      "purpose":"Numerical surrogate for chaotic forecast rollout and verification concepts; not ERA5."
    }
    (out/"ch01_weather_lorenz96_provenance.json").write_text(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
