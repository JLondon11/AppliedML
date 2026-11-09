"""Compose DDPM figure 24 from *measured* independent test scores and generated outputs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(ddpm_dir: Path, dest: Path) -> None:
    ddpm_dir=Path(ddpm_dir)
    result=json.loads((ddpm_dir/'metrics.json').read_text())
    fig,ax=plt.subplots(1,3,figsize=(13.5,4.5),
                        gridspec_kw={'width_ratios':[1.1,1.1,1.35]})
    for axis,file in zip(ax[:2],['real_test_digits.png','ddpm_generated_samples.png']):
        axis.imshow(plt.imread(ddpm_dir/file))
        axis.axis('off')
    measurements=result['independent_test_mse']
    steps=[int(k)+1 for k in measurements]
    model=[v['predicted_noise_mse'] for v in measurements.values()]
    baseline=[v['zero_predictor_mse'] for v in measurements.values()]
    ax[2].plot(steps,model,'o-',label='Trained DDPM')
    ax[2].plot(steps,baseline,'s--',label='Zero-predictor baseline')
    ax[2].set(xlabel='Forward diffusion timestep',
              ylabel='Held-out noise-prediction MSE',
              title='Independent test evaluation')
    ax[2].legend(frameon=False,fontsize=8)
    ax[2].grid(alpha=.2)
    fig.suptitle('Real-data DDPM training and sampling: sklearn handwritten digits',fontsize=12)
    fig.tight_layout()
    fig.savefig(dest,dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('ddpm_dir',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    render(args.ddpm_dir,args.destination)
