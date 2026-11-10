"""Authentic IHC microscopy candidate segmentation and independent WDBC diagnosis.

This script DOES NOT perform or claim nucleus-level malignant/benign classification
on the microscope photo. No ground-truth mask accompanies the photo.
Dependencies: numpy scipy matplotlib scikit-image scikit-learn.
Run: python run.py
"""
import hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data, color, filters, morphology, feature, measure, segmentation
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, accuracy_score, confusion_matrix

OUT=Path(__file__).resolve().parent/"outputs"
OUT.mkdir(parents=True,exist_ok=True)

def main():
    original=data.immunohistochemistry()
    assert original.shape==(512,512,3)
    hematoxylin=color.rgb2hed(original)[:,:,0]
    smoothed=filters.gaussian(hematoxylin,sigma=1.2)
    cutoff=float(np.quantile(smoothed,0.95))
    initial=smoothed>cutoff
    cleaned=morphology.remove_small_objects(initial,min_size=16)
    cleaned=morphology.remove_small_holes(cleaned,area_threshold=16)
    distance=ndi.distance_transform_edt(cleaned)
    peaks=feature.peak_local_max(distance,min_distance=5,
                                 threshold_abs=1.8,labels=cleaned)
    markers=np.zeros(distance.shape,dtype=np.int32)
    if len(peaks):
        markers[tuple(peaks.T)]=np.arange(1,len(peaks)+1)
    markers,_=ndi.label(markers>0)
    instances=segmentation.watershed(-distance,markers,mask=cleaned)
    regions=[p for p in measure.regionprops(instances,intensity_image=smoothed)
             if 16<=p.area<=500]
    kept=np.zeros(instances.shape,dtype=np.int32)
    for i,p in enumerate(regions,1):
        kept[instances==p.label]=i
    area=np.array([p.area for p in regions],dtype=float)
    eccentricity=np.array([p.eccentricity for p in regions],dtype=float)

    wdbc=load_breast_cancer()
    X=wdbc.data
    y=(wdbc.target==0).astype(int) # malignant is positive
    trainval,test=train_test_split(np.arange(569),test_size=114,
                                   stratify=y,random_state=20261109)
    train,validation=train_test_split(trainval,test_size=114,
                                      stratify=y[trainval],random_state=20261110)
    assert (len(train),len(validation),len(test))==(341,114,114)
    candidates=[]
    for C in (.01,.1,1.,10.,100.):
        model=make_pipeline(StandardScaler(),
                            LogisticRegression(C=C,max_iter=3000,
                                               random_state=20261109))
        model.fit(X[train],y[train])
        candidates.append((C,model,float(f1_score(y[validation],
                                                  model.predict(X[validation])))))
    C,model,_=max(candidates,key=lambda row:row[2])
    probability=model.predict_proba(X[test])[:,1]
    fpr,tpr,_=roc_curve(y[test],probability)
    auc=float(roc_auc_score(y[test],probability))
    accuracy=float(accuracy_score(y[test],probability>=.5))
    confusion=confusion_matrix(y[test],probability>=.5).tolist()

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.4,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig=plt.figure(figsize=(13.6,7.6),layout="constrained")
    grids=fig.add_gridspec(2,6,height_ratios=[1.15,1],hspace=.08)
    ax=[fig.add_subplot(grids[0,0:2]),fig.add_subplot(grids[0,2:4]),
        fig.add_subplot(grids[0,4:6]),fig.add_subplot(grids[1,0:3]),
        fig.add_subplot(grids[1,3:6])]
    ax[0].imshow(original)
    ax[0].set_title("(a) Authentic immunohistochemistry photograph")
    ax[0].axis("off")
    ax[1].imshow(smoothed,cmap="magma")
    ax[1].contour(cleaned,levels=[.5],colors=["#f5efe5"],linewidths=.5)
    ax[1].set_title("(b) Hematoxylin-derived candidate mask")
    ax[1].axis("off")
    ax[2].imshow(original,alpha=.7)
    ax[2].imshow(np.ma.masked_equal(kept,0),cmap="tab20",
                 alpha=.56,interpolation="nearest")
    ax[2].set_title(f"(c) Stain-positive candidate regions (n={len(regions)})")
    ax[2].axis("off")
    ax[3].scatter(area,eccentricity,s=15,color="#2d796f",alpha=.65,
                  linewidths=0,rasterized=True)
    ax[3].set(title="(d) Measured candidate-region morphology",
              xlabel="Pixel area per segmented region",ylabel="Eccentricity",
              ylim=(0,1.03))
    ax[4].plot(fpr,tpr,color="#25475c",lw=2.25,
               label=f"Actual held-out WDBC predictions · AUC={auc:.4f}")
    ax[4].plot([0,1],[0,1],ls=":",color="#a3afb6")
    ax[4].set(title="(e) Independent real-data diagnosis task (WDBC)",
              xlabel="False-positive rate",ylabel="True-positive rate",
              xlim=(0,1),ylim=(0,1.025))
    ax[4].legend(frameon=False,fontsize=8)
    for pane in ax[3:]:
        pane.grid(alpha=.16)
    fig.suptitle("Authentic microscopy segmentation and independent patient-level diagnosis",
                 fontweight="bold",fontsize=14.5,color="#25475c")
    stem="Figure_11_22_Watershed_Nuclear_Segmentation_and_Classification"
    for ext in ("png","pdf","svg"):
        fig.savefig(OUT/f"{stem}.{ext}",dpi=230,bbox_inches="tight")
    plt.close(fig)
    record={
       "microscopy_dataset":"scikit-image immunohistochemistry real tissue photograph",
       "image_shape":list(original.shape),
       "image_pixel_sha256":hashlib.sha256(original.tobytes()).hexdigest(),
       "segmentation_kind":"Unsupervised 95th percentile hematoxylin and watershed",
       "candidate_regions_count":len(regions),
       "candidate_region_areas_px":area.tolist(),
       "candidate_region_eccentricities":eccentricity.tolist(),
       "segmentation_has_ground_truth":False,
       "segmentation_dice":None,"segmentation_aji":None,
       "classification_dataset":"Wisconsin Diagnostic Breast Cancer 569 authentic patient cases",
       "classification_has_pairing_to_photograph":False,
       "classification_train":341,"classification_val":114,
       "classification_test":114,"classification_C":float(C),
       "classification_test_auc":auc,"classification_test_accuracy":accuracy,
       "classification_confusion_matrix_benign_malignant":confusion,
       "classification_test_ground_truth":y[test].tolist(),
       "classification_test_probabilities":probability.tolist(),
       "limitations":"Candidate stain-positive regions not validated nuclei; WDBC patient labels do not belong to microscopy photograph. MoNuSeg benchmark NOT performed."
    }
    (OUT/"measured_results.json").write_text(json.dumps(record,indent=2)+"\n")
    assert len(record["classification_test_probabilities"])==114
    assert np.isclose(roc_auc_score(record["classification_test_ground_truth"],
                                   record["classification_test_probabilities"]),auc)
    print(json.dumps({k:record[k] for k in
         ("candidate_regions_count","classification_train","classification_val",
          "classification_test","classification_C","classification_test_auc",
          "classification_test_accuracy")},indent=2))

if __name__=="__main__":
    main()
