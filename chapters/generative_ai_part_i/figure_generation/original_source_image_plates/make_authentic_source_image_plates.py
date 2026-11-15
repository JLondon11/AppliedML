"""Generate image-first scientific source-image galleries from authentic data.

These are INPUT-data visualizations, not reconstructions or neural generations.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import data
from sklearn.datasets import load_digits

OUT=Path(__file__).resolve().parent
def main(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    im=data.immunohistochemistry()
    assert im.shape==(512,512,3) and im.dtype==np.uint8
    fig=plt.figure(figsize=(12.3,7.5),facecolor='white')
    gs=fig.add_gridspec(3,5,left=.05,right=.975,bottom=.105,top=.865,hspace=.28,wspace=.16,
                       width_ratios=[2.15,1,1,1,1])
    ax=fig.add_subplot(gs[:,:1]);ax.imshow(im);ax.set_axis_off()
    for first,last,color,name in ((0,320,'#16938a','TRAIN'),
                                  (320,384,'#d79e48','VALIDATE'),(384,512,'#a85c69','TEST')):
        ax.add_patch(Rectangle((-.5,first-.5),512,last-first,fill=False,ec=color,lw=2.2))
        ax.text(8,first+15,name,color='white',fontsize=10,weight='bold',
             bbox={'facecolor':color,'alpha':.92,'boxstyle':'round,pad=.2','edgecolor':'none'})
    ax.set_title('A   Original tissue / source partitions',loc='left',
                 fontsize=11,weight='bold',color='#173e5b',pad=11)
    positions=((32,32),(112,144),(208,320),(336,48),(336,224),(352,416),
               (384,48),(416,224),(464,416),(448,48),(480,208),(384,416))
    for idx,(r,c) in enumerate(positions):
        axis=fig.add_subplot(gs[idx//4,idx%4+1])
        axis.imshow(im[r:r+32,c:c+32],interpolation='nearest');axis.set_axis_off()
        axis.text(.03,.04,f'{r}:{r+32}, {c}:{c+32}',transform=axis.transAxes,
              fontsize=6.4,color='white',
              bbox={'facecolor':'#1f3444','alpha':.86,'edgecolor':'none','pad':1.5})
    fig.text(.05,.952,'Authentic microscopy input • spatial separation and observed tissue texture',
          color='#163f5a',fontsize=16.2,weight='bold')
    fig.text(.05,.907,'Original scikit-image immunohistochemistry photograph; source-photo regions are disjoint, but patches share one parent photograph.',
          color='#526777',fontsize=9.6)
    fig.text(.05,.053,'Left: original photograph and source-row partitions. Right: actual 32 × 32 input image crops (not VAE reconstructions).',
          color='#4e6576',fontsize=9)
    fig.text(.05,.027,'No generated tissue, clinical validation, patient-level generalization or model inference is illustrated.',
          color='#4e6576',fontsize=8.7)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'original_tissue_source_and_patches.{ext}',dpi=225,facecolor='white')
    plt.close(fig)

    d=load_digits();images=d.images;labels=d.target
    fig=plt.figure(figsize=(12.3,5.25),facecolor='white')
    gs=fig.add_gridspec(2,10,left=.05,right=.975,bottom=.17,top=.78,hspace=.12,wspace=.18)
    chosen=[]
    for digit in range(10):
        candidates=np.flatnonzero(labels==digit)
        indices=(int(candidates[0]),int(candidates[len(candidates)//2]))
        chosen.append(indices)
        for k,i in enumerate(indices):
            ax=fig.add_subplot(gs[k,digit])
            ax.imshow(images[i],cmap='gray',vmin=0,vmax=16,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if k==0:ax.set_title(str(digit),fontsize=12,weight='bold',color='#1b485f')
            if digit==0:ax.set_ylabel('Scan A' if k==0 else 'Scan B',fontsize=10,labelpad=10)
            for spine in ax.spines.values():spine.set_edgecolor('#d8e1e6')
    fig.text(.05,.944,'Authentic scanned handwriting • visual diversity within original digit classes',
        fontsize=16.2,color='#183f5b',weight='bold')
    fig.text(.05,.875,'20 distinct original scikit-learn digit scans, two source observations per class. Not diffusion-generated images.',
        fontsize=10,color='#506778')
    fig.text(.05,.07,'Original source intensities (0–16); 8 × 8 pixels per image. First and median-index scan in each class.',
        color='#506778',fontsize=9)
    fig.text(.05,.037,'Show alongside model-generated outputs; never substitute these original scans for neural generations.',
        color='#506778',fontsize=8.6)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'original_scanned_digits_gallery.{ext}',dpi=225,facecolor='white')
    plt.close(fig)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    provenance={'actual_source_not_model_generated':True,
        'tissue_source':'skimage.data.immunohistochemistry()',
        'tissue_shape':list(im.shape),'tissue_source_pixel_sha256':sha(im.tobytes()),
        'tissue_region_rows':{'training':[0,319],'validation':[320,383],'test':[384,511]},
        'digit_source':'sklearn.datasets.load_digits()',
        'digit_source_sample_count':len(images),
        'digit_source_image_array_sha256':sha(images.tobytes()),
        'digit_class_counts':{str(k):int((labels==k).sum()) for k in range(10)},
        'selected_digit_source_indices':chosen,
        'scope':'original data samples only; no executed VAE/diffusion predictions are shown'}
    (out/'source_image_plate_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))
if __name__=='__main__':main()
