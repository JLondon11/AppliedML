"""Image-first print plates from the actually executed original Penn–Fudan detector.

Source is the successful real-data Faster R-CNN result archive OR its verified
unpacked contents from a GitHub Actions artifact. Source-figure hashes are
fixed: NEVER replace real photos/predictions with illustrations. This does NOT
measure occlusion-stratified detection or certify its broader parent case study.
"""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORIGINAL_ZIP='1bc0a2e7690c47c4a51a7fd1b09603134d7690cfb78bccab9dae22cb7f353fb5'
ORIGINAL_PNGS={
'FudanPed00001':'4e9d8ad9b485b68875cd47d74888d430c44d10d9b4c79610877742c8d0aaa0ea',
'FudanPed00002':'6ef68ff8eed47069494ae20efbd3ca0f220d57e51ae029e501c230d93e32f11e',
'FudanPed00003':'4477208a97e1171e1640ec229c5de8ac589b8c2f8876aec57abaa5c159d4b267',
'FudanPed00004':'592aabb8d9a95367a72495541e357f8b7f21b63c25f4358428ca4b7bfb951a98',
'FudanPed00005':'bc17b338759cc20ef811394158433f2ee54c3dde1599307c0d0720434b458533',
'FudanPed00006':'6d1148fbfb6c1fed21341c3931950a6a944323241efb3535f709d2dcbd5ed3f6'}
def digest(data):return hashlib.sha256(data).hexdigest()
def load_source(src):
 src=Path(src)
 if src.is_file():
  if digest(src.read_bytes())!=ORIGINAL_ZIP:
   raise ValueError('Original artifact ZIP differs from the scientifically verified archive')
  with zipfile.ZipFile(src) as z:
   if z.testzip() is not None:raise ValueError('Archive corruption')
   files={n:z.read('Figures/'+n+'.png') for n in ORIGINAL_PNGS}
   metrics=z.read('metrics.csv');provenance=z.read('provenance.json')
 else:
  files={n:(src/'Figures'/f'{n}.png').read_bytes() for n in ORIGINAL_PNGS}
  metrics=(src/'metrics.csv').read_bytes()
  provenance=(src/'provenance.json').read_bytes()
 for n,raw in files.items():
  if digest(raw)!=ORIGINAL_PNGS[n]:
   raise ValueError('Original executed-model photographed figure mutated: '+n)
 return files,metrics,json.loads(provenance)
def main(src,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 images,metrics_content,provenance=load_source(src)
 metrics={r['metric']:float(r['value']) for r in csv.DictReader(io.StringIO(metrics_content.decode()))}
 if (int(metrics['ground_truth_persons']),int(metrics['predicted_boxes']),int(metrics['true_positives']))!=(423,1199,417):
  raise ValueError('Actual original original-image/GT/detections quantities changed')
 if not abs(metrics['local_ap50']-.9681834843380837)<1e-12:
  raise ValueError('Original evaluator AP50 differs from recorded source')
 images={name:Image.open(io.BytesIO(raw)).convert('RGB') for name,raw in images.items()}
 figdata=[]
 for part in (0,1):
  fig,axes=plt.subplots(3,1,figsize=(6.5,9.4),facecolor='white')
  for row in range(3):
   key=f'FudanPed{part*3+row+1:05d}'
   image=images[key]
   a=axes[row];a.imshow(image);a.axis('off')
   a.set_title(f'{chr(65+row)}  {key} · original annotations (left) / actual detections (right)',
     fontsize=8.6,color='#1e435b',weight='bold',loc='left',pad=6)
   figdata.append(dict(page=part+1,source_image=key,
      original_png_byte_sha256=ORIGINAL_PNGS[key],width=image.width,height=image.height))
  fig.text(.055,.972,'Original Penn–Fudan pedestrian detection',fontsize=13.3,weight='bold',color='#143e56')
  fig.text(.055,.935,f'Genuine source photographs and executed Faster R-CNN outputs · set {part+1}/2',fontsize=8.6,color='#516a78')
  fig.text(.055,.064,'Original person annotations: teal; actual predicted boxes: orange. No occlusion GT.',fontsize=8.2,color='#526b79')
  fig.text(.055,.038,'170 genuine photographs · 423 original persons · 417 local IoU matches.',fontsize=8.2,color='#526b79')
  fig.subplots_adjust(left=.045,right=.985,top=.887,bottom=.105,hspace=.28)
  for ext in ('pdf','png','svg'):
   fig.savefig(out/f'pennfudan_original_detection_gallery_{part+1}.{ext}',
      dpi=210,facecolor='white')
  plt.close(fig)
 report=dict(original_experiment='COCO-pretrained Faster R-CNN MobileNetV3 320 on original PennFudanPed',
  original_source_archive_sha256=ORIGINAL_ZIP,
  actual_original_checkpoint_sha256=provenance['checkpoint_sha256'],
  actual_original_evaluator_metrics=metrics,
  original_annotated_photo_results=figdata,
  model_predictions_are_real_inferences=True,
  original_groundtruth_masks_in_archive=False,
  no_occlusion_stratification=True,
  certified_scientific_scope='CV5-D02 Level A genuine general pedestrian detection only',
  broader_parent_cv5_cs11_certified=False,
  production_manuscript_approved=False)
 (out/'pennfudan_original_figure_source_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'actual_annotated_photographs':len(figdata),'source_original_archive_sha256':ORIGINAL_ZIP,'original_evaluator_metrics':metrics},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--source',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();main(a.source,a.output)
