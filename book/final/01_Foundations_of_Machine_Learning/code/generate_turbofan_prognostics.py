from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='book/final/01_Foundations_of_Machine_Learning/figures')
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    rng=np.random.default_rng(31)
    cycles=np.arange(1,181)
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.25))
    palette=['#355C7D','#4F7C6E','#B26E3B','#7A6AA6','#657A8A','#8A6F5A','#6C8066','#85789C']

    for i in range(8):
        life=170+rng.normal(0,12)
        exponent=1.5+rng.uniform(-.15,.15)
        health=np.clip(1-(cycles/life)**exponent+rng.normal(0,.015,len(cycles)),0,1)
        axs[0].plot(cycles,health,lw=1.0,color=palette[i],alpha=.9)
    axs[0].set_xlabel('Operating cycle')
    axs[0].set_ylabel('Normalized health indicator')
    axs[0].set_ylim(-.02,1.03)

    reference=np.maximum(0,180-cycles)
    estimate=np.clip(reference+rng.normal(0,8,len(cycles)),0,None)
    axs[1].plot(cycles,reference,label='Reference RUL',lw=2,color='#355C7D')
    axs[1].plot(cycles,estimate,label='Estimated RUL',lw=1.2,color='#B26E3B',alpha=.78)
    axs[1].set_xlabel('Operating cycle')
    axs[1].set_ylabel('Remaining useful life (cycles)')
    axs[1].legend(frameon=False,fontsize=8)

    for i,ax in enumerate(axs):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out')
        ax.text(.5,-.23,f'({chr(97+i)})',transform=ax.transAxes,ha='center',va='top',fontsize=10)

    fig.tight_layout(w_pad=2.2)
    for ext in ('png','svg','pdf'):
        fig.savefig(out/f'ch01_turbofan_prognostics.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    main()
