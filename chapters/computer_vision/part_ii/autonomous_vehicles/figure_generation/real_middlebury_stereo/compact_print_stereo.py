"""Print-width faithful view of actually executed Middlebury stereo BM/SGBM."""
from pathlib import Path
import hashlib,json,runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
s=runpy.run_path(str(base/'real_middlebury_stereo_certification.py'))
out=base/'compact';out.mkdir(exist_ok=True)
left,right,gt,ref=s['left'],s['right'],s['gt'],s['ref']
rows,preds=s['rows'],s['preds']
assert int(ref.sum())==343274
assert rows[0]['evaluated_predicted_valid_pixels']==254330 and rows[1]['evaluated_predicted_valid_pixels']==284723
ink='#173e57';muted='#516878'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
bm,bmv=preds['StereoBM'];sg,sgv=preds['StereoSGBM']
def assemble(name,title,panels,notes):
 fig=plt.figure(figsize=(6.5,7.25),facecolor='white')
 gs=fig.add_gridspec(3,2,left=.075,right=.97,top=.83,bottom=.16,hspace=.24,wspace=.12)
 for i,(head,im,cmap,lo,hi) in enumerate(panels):
  ax=fig.add_subplot(gs[i//2,i%2]);ax.imshow(im,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
  ax.set_title(head,fontsize=9,loc='left',color=ink,weight='bold');ax.axis('off')
 fig.text(.075,.962,title,fontsize=13.4,weight='bold',color=ink)
 fig.text(.075,.915,'Authentic Middlebury stereo pair · 343,274 valid reference pixels',fontsize=8.2,color=muted)
 for y,n in zip((.111,.083,.052),notes):fig.text(.075,y,n,fontsize=8.2,color=muted)
 for ext in ('pdf','svg','png'):fig.savefig(out/f'{name}.{ext}',dpi=230,facecolor='white')
 plt.close(fig)
assemble('middlebury_stereo_source_and_actual_predictions','Original images → actual disparities',[
 ('A  Original left photo',left,None,None,None),('B  Original right photo',right,None,None,None),
 ('C  Original GT disparity',np.ma.masked_where(~ref,gt),'viridis',0,90),
 ('D  StereoBM disparity',np.ma.masked_where(~bmv,bm),'viridis',0,90),
 ('E  StereoSGBM disparity',np.ma.masked_where(~sgv,sg),'viridis',0,90),
 ('F  Source photograph crop',left[160:390,180:520],None,None,None)],
 ('Original photos and original measured disparity; trained models not required.',
  'Disparity maps share a 0–90 px scale; invalid estimates masked.',
  'One scene only; this is NOT camera–LiDAR sensor fusion.'))
ebm=np.ma.masked_where(~(ref&bmv),np.abs(bm-gt));esg=np.ma.masked_where(~(ref&sgv),np.abs(sg-gt))
mb=np.where(~ref,np.nan,np.where(bmv,1,0));ms=np.where(~ref,np.nan,np.where(sgv,1,0))
assemble('middlebury_stereo_observed_error_and_missing','Measured errors and missing estimates',[
 ('A  BM disparity error',ebm,'magma',0,12),('B  SGBM disparity error',esg,'magma',0,12),
 ('C  BM prediction coverage',mb,'gray',0,1),('D  SGBM prediction coverage',ms,'gray',0,1),
 ('E  BM residual crop',ebm[160:390,180:520],'magma',0,12),
 ('F  SGBM residual crop',esg[160:390,180:520],'magma',0,12)],
 (f"BM observed coverage {rows[0]['coverage']:.1%}, Bad-3 incl. misses {rows[0]['bad3_all_reference_including_missing']:.1%}",
  f"SGBM observed coverage {rows[1]['coverage']:.1%}, Bad-3 incl. misses {rows[1]['bad3_all_reference_including_missing']:.1%}",
  'Error 0–12 pixels; reference-invalid regions excluded from residuals.'))
result={'original_pixel_sha256':{a:hashlib.sha256(x.tobytes()).hexdigest() for a,x in [('left',left),('right',right),('reference',gt)]},
 'observed_metrics':rows,'source_scenes':1,'figure_numbers_assigned':False,
 'parent_camera_lidar_fusion_certified':False,'production_approved':False}
(out/'compact_figure_source_provenance.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
