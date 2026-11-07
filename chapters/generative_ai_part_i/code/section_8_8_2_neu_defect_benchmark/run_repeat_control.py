"""Class-balanced equal-size augmentation control for the NEU-DET benchmark."""
import csv
from pathlib import Path
import torch
from run import prepare_dataset, SEED, CLASSES
from run_multiseed import load_splits,train_classifier,seed_all

def main():
    root=Path(__file__).resolve().parent
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=root/'results_multiseed')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    seed_all(SEED)
    data,splits,_=prepare_dataset(args.data_root,args.output,False)
    subsets=load_splits(data,splits)
    metrics=[];traces=[];predictions=[]
    for seed in (20261106,20261107,20261108):
        seed_all(seed+501)
        x,y=subsets['train'];extra=[]
        for label in range(6):
            possible=torch.where(y==label)[0]
            extra.append(possible[torch.randint(0,len(possible),(120,))])
        selected=torch.cat(extra)
        images=x[selected].clone();labels=y[selected].clone()
        flip=torch.rand(len(images))<0.5
        images[flip]=images[flip].flip(-1)
        result,trace,preds,_=train_classifier(subsets['train'],subsets['validation'],
            subsets['test'],'real_repeat_augmented',seed,12,32,args.output,(images,labels))
        metrics.append(result);traces.extend(trace);predictions.extend(preds)
    with (args.output/'repeat_control_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    with (args.output/'repeat_control_training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','arm','epoch','train_loss','val_balanced_accuracy']);w.writerows(traces)
    with (args.output/'repeat_control_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','arm','test_row','true_class','pred_class',*['p_'+k for k in CLASSES]]);w.writerows(predictions)
if __name__=='__main__':main()
