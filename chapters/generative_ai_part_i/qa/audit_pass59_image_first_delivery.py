"""Validate exact locally delivered original Chapter 6 scientific source/model image assets.

Requires the three actual delivered PDFs and source/data manifests; this is
a staged-asset integrity certificate, not approval of absent manuscript PDF.
"""
import argparse,hashlib,json
from pathlib import Path
from skimage import data
from sklearn.datasets import load_digits

PDF_SHA256={
 'original_tissue_source_and_patches.pdf':'3211bfab2579ed7b246677f93b88130f8ffbb8e71eb26adc74c3379eb530f416',
 'original_scanned_digits_gallery.pdf':'e74cfde0be705c00fda8c1c48fa441cd2e07abb112fedff4335c36ff05785264',
 'actual_tissue_cvae_reconstruction_gallery.pdf':'e1845abf02988cb5d2dbc4c9a52d764ab3d740bb2457c9e56e2466b1c38b6c36'
}
def audit(stage):
    stage=Path(stage)
    tissue=data.immunohistochemistry()
    digits=load_digits()
    if hashlib.sha256(tissue.tobytes()).hexdigest()!='c5b3ef509a92f16d4c29be8cf0300fe75d53e13a3ce650159db932caea8dcc1b':
        raise ValueError('Original tissue source differs from observed-data experiment')
    if hashlib.sha256(digits.images.tobytes()).hexdigest()!='20def7f70a702f0af9732fbba4375e147a7d54fe70d8c45569b8e7c1c7010c10':
        raise ValueError('Original 1797 scanned-digit image source differs')
    actual={
       'original_tissue_source_and_patches.pdf':stage/'original_tissue_source_and_patches.pdf',
       'original_scanned_digits_gallery.pdf':stage/'original_scanned_digits_gallery.pdf',
       'actual_tissue_cvae_reconstruction_gallery.pdf':stage/'model_verified_gallery'/'actual_tissue_cvae_reconstruction_gallery.pdf'
    }
    files={}
    for name,path in actual.items():
        if not path.is_file() or path.stat().st_size<10000:
            raise ValueError('Missing actual published figure source: '+str(path))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=PDF_SHA256[name]:
            raise ValueError('The original delivered PDF no longer matches its provenance: '+name)
        files[name]={'bytes':path.stat().st_size,'sha256':digest}
    source=json.loads((stage/'source_image_plate_provenance.json').read_text())
    model=json.loads((stage/'model_verified_gallery'/'actual_tissue_gallery_measured_metrics.json').read_text())
    if not source['actual_source_not_model_generated'] or source['digit_source_sample_count']!=1797:
        raise ValueError('False original-image/model-output identity')
    if model['source_images']!=1 or model['spatially_heldout_patches']!=217:
        raise ValueError('One-photograph original VAE scope corrupted')
    if abs(model['actual_model_test_PSNR_dB']-20.15526594713861)>1e-6:
        raise ValueError('Recorded original real-data model result inconsistency')
    return {'status':'PASS_STAGED_SCIENTIFIC_FIGURE_ASSET_INTEGRITY',
        'original_real_source_images_verified':True,
        'actual_model_outputs_verified':True,
        'PDFs':files,'production_approved':False,
        'why_not_production_approved':'No actual current Chapter 6 PDF/TeX or exact figure-page/caption mapping.'}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(audit(a.stage),indent=2))
