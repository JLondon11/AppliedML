"""NCI molecular fingerprint variational autoencoder with held-out benchmarks.

Real data: RDKit-distributed NCI first_200.props.sdf.
Produces source/dataset provenance, VAE training traces, held-out Tanimoto
metrics and publication figures. Predicts fingerprints only; DOES NOT decode
valid molecular graphs or claim a novel drug discovery benchmark.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs, RDConfig, rdBase
from rdkit.Chem import Crippen, rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
from torch import nn

SEED = 20261106
BITS = 256
LATENT = 12

class FingerprintVAE(nn.Module):
    def __init__(self, features=BITS, hidden=128, latent=LATENT):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, hidden//2), nn.ReLU())
        self.mu = nn.Linear(hidden//2, latent)
        self.logvar = nn.Linear(hidden//2, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden//2), nn.ReLU(), nn.Linear(hidden//2, hidden), nn.ReLU(), nn.Linear(hidden, features))

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 8)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.decoder(z), mu, logvar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    return parser.parse_args()


def tanimoto_bits(reference, reconstructed):
    a = np.asarray(reference, dtype=bool)
    b = np.asarray(reconstructed, dtype=bool)
    union = np.logical_or(a, b).sum(axis=1)
    inter = np.logical_and(a, b).sum(axis=1)
    return np.divide(inter, union, out=np.ones(len(a),dtype=float), where=union != 0)


def main():
    args = parse_args()
    out = args.output
    figdir = out / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    sdfpath = Path(RDConfig.RDDataDir) / 'NCI' / 'first_200.props.sdf'
    if not sdfpath.exists():
        raise FileNotFoundError(f'RDKit NCI source missing: {sdfpath}')
    source_sha = hashlib.sha256(sdfpath.read_bytes()).hexdigest()
    supplier = Chem.SDMolSupplier(str(sdfpath), removeHs=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=BITS)
    vectors = []
    properties = []
    smiles = []
    invalid = 0
    for mol in supplier:
        if mol is None:
            invalid += 1
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        vectors.append(arr.astype(np.float32))
        properties.append(float(Crippen.MolLogP(mol)))
        smiles.append(Chem.MolToSmiles(mol, canonical=True))
    X = np.stack(vectors)
    y = np.asarray(properties)
    assert X.shape[0] >= 140
    ids = np.arange(len(X))
    trainval, test = train_test_split(ids, test_size=0.20, random_state=SEED, shuffle=True)
    train, valid = train_test_split(trainval, test_size=0.25, random_state=SEED, shuffle=True)
    # Approximately 60/20/20 train/validation/test. No held-out examples used in training.
    split = {str(i): ('train' if i in set(train) else 'validation' if i in set(valid) else 'test') for i in ids}
    with (out/'dataset_manifest.csv').open('w',newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['source_row','split','canonical_smiles','rdkit_logP'])
        writer.writerows((int(i), split[str(i)], smiles[i], y[i]) for i in ids)

    model = FingerprintVAE()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    Xtrain = torch.from_numpy(X[train]); Xvalid = torch.from_numpy(X[valid]); Xtest = torch.from_numpy(X[test])
    trace = []
    best = float('inf'); best_state = None; best_epoch = -1
    beta = 0.03
    for epoch in range(args.epochs):
        model.train()
        for batch in torch.randperm(len(train)).split(32):
            t = Xtrain[batch]
            logits, mu, lv = model(t)
            bce = nn.functional.binary_cross_entropy_with_logits(logits,t,reduction='none').sum(dim=-1).mean()
            kl = (-0.5*(1+lv-mu.square()-lv.exp()).sum(-1)).mean()
            loss = bce + beta*kl
            optim.zero_grad(); loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            tr, m, lv = model(Xtrain)
            vl, mv, lvv = model(Xvalid)
            tr_bce = nn.functional.binary_cross_entropy_with_logits(tr,Xtrain,reduction='none').sum(dim=-1).mean().item()
            vl_bce = nn.functional.binary_cross_entropy_with_logits(vl,Xvalid,reduction='none').sum(dim=-1).mean().item()
            vl_kl = (-0.5*(1+lvv-mv.square()-lvv.exp()).sum(-1)).mean().item()
            score = vl_bce + beta*vl_kl
        trace.append([epoch+1, tr_bce, vl_bce, vl_kl, score])
        if score < best:
            best=score; best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, mus, lv = model(Xtest)
        p = torch.sigmoid(logits).numpy()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, Xtest, reduction='none').sum(-1).numpy()
    reconstructed = p >= 0.5
    truth = X[test].astype(bool)
    vae_similarity = tanimoto_bits(truth, reconstructed)
    pca = PCA(n_components=LATENT, random_state=SEED).fit(X[train])
    rec_pca = np.clip(pca.inverse_transform(pca.transform(X[test])),0,1)
    pca_similarity = tanimoto_bits(truth, rec_pca>=0.5)
    pairwise = np.asarray([tanimoto_bits(np.repeat(x[None,:],len(train),axis=0),X[train]) for x in X[test]])
    nearest = X[train][pairwise.argmax(axis=1)]
    nearest_similarity = tanimoto_bits(truth,nearest)
    results=[]
    for j, source_idx in enumerate(test):
        results.append([int(source_idx),float(vae_similarity[j]),float(pca_similarity[j]),float(nearest_similarity[j]),float(bce[j]),float(y[source_idx])])
    with (out/'heldout_predictions.csv').open('w',newline='') as fh:
        writer=csv.writer(fh)
        writer.writerow(['source_row','vae_tanimoto','pca_tanimoto','nearest_training_tanimoto','vae_reconstruction_BCE','rdkit_logP'])
        writer.writerows(results)
    with (out/'training_trace.csv').open('w',newline='') as fh:
        writer=csv.writer(fh)
        writer.writerow(['epoch','train_BCE','validation_BCE','validation_KL','validation_objective'])
        writer.writerows(trace)

    fig, ax = plt.subplots(figsize=(7.0,3.6))
    ax.plot([r[0] for r in trace],[r[1] for r in trace],label='Training reconstruction BCE')
    ax.plot([r[0] for r in trace],[r[2] for r in trace],label='Validation reconstruction BCE')
    ax.axvline(best_epoch,color='0.5',linestyle='--',linewidth=1,label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='Binary cross-entropy per fingerprint (summed bits)')
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir/'nci_vae_training.png',dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0,3.9))
    vals=[vae_similarity,pca_similarity,nearest_similarity]
    titles=['VAE','PCA','Nearest training molecule']
    ax.boxplot(vals,tick_labels=titles,showmeans=True)
    ax.set(ylabel='Held-out Morgan-fingerprint Tanimoto similarity',ylim=(0,1.05))
    fig.tight_layout()
    fig.savefig(figdir/'nci_vae_heldout_benchmark.png',dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2,4.2))
    points = pca.transform(X[test])
    plot=ax.scatter(points[:,0],points[:,1],c=y[test],cmap='viridis',s=32,edgecolors='none')
    fig.colorbar(plot,ax=ax,label='RDKit calculated logP')
    ax.set(xlabel='PCA coordinate 1',ylabel='PCA coordinate 2')
    fig.tight_layout()
    fig.savefig(figdir/'nci_molecular_structure_property_map.png',dpi=180)
    plt.close(fig)

    metrics={
      'source':'RDKit Data/NCI/first_200.props.sdf','source_sha256':source_sha,
      'original_molecule_count':len(supplier),'valid_molecule_count':len(X),'invalid_count':invalid,
      'n_train':len(train),'n_validation':len(valid),'n_test':len(test),'split_seed':SEED,
      'morgan_radius':2,'fingerprint_bits':BITS,'latent_dimensions':LATENT,
      'epochs_requested':args.epochs,'selected_epoch':best_epoch,'kl_weight_beta':beta,
      'vae_mean_tanimoto':float(vae_similarity.mean()),'vae_sd_tanimoto':float(vae_similarity.std(ddof=1)),
      'pca_mean_tanimoto':float(pca_similarity.mean()),'pca_sd_tanimoto':float(pca_similarity.std(ddof=1)),
      'nearest_train_mean_tanimoto':float(nearest_similarity.mean()),
      'vae_mean_reconstruction_bce':float(bce.mean()),
      'limitations':['Fingerprints, not molecular graph/SMILES reconstruction',
                     'No novel molecule generation, validity, or wet-lab measurements',
                     'Small convenience dataset; random rather than scaffold split',
                     'Metrics only compare representations on 20% held-out NCI subset'],
      'versions':{'rdkit':rdBase.rdkitVersion,'torch':torch.__version__,'numpy':np.__version__}
    }
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    torch.save({'model':best_state,'metrics':metrics},out/'nci_fingerprint_vae.pt')
    print(json.dumps(metrics,indent=2))

if __name__ == '__main__':
    main()
