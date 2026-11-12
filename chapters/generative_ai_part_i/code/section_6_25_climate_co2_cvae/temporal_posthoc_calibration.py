"""Post-hoc Mauna Loa interval evaluation on strictly later real measured CO2.

No retraining: input test_predictions.csv from the executed 65-epoch Chapter 6
CVAE, with model/checkpoint selection ending by 1995. Calibrate from complete
1996–1998 target horizons, test from forecast starts in 1999–2001.
Overlapping forecast windows invalidate standard exchangeable conformal
coverage guarantees. Report observed test coverage only.
"""
import argparse,json,math
from pathlib import Path
import pandas as pd
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--predictions',type=Path,required=True)
ap.add_argument('--outdir',type=Path,default=Path('temporal_calibration'))
args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv(args.predictions,parse_dates=['first_forecast_week'])
required={'seed','first_forecast_week','horizon_week','true_co2_ppm','cvae_mean_ppm','cvae_q05_ppm','cvae_q95_ppm'}
assert required.issubset(raw.columns)
raw['forecast_end']=raw.first_forecast_week+pd.to_timedelta(7*7,unit='D')
cut=pd.Timestamp('1998-12-31')
cal=raw[(raw.first_forecast_week>=pd.Timestamp('1996-01-01')) & (raw.forecast_end<=cut)]
held=raw[raw.first_forecast_week>cut]
assert cal.forecast_end.max()<=cut and held.first_forecast_week.min()>cut
report=[];predictions=[]
for seed in sorted(raw.seed.unique()):
 for lead in range(1,9):
  a=cal[(cal.seed==seed)&(cal.horizon_week==lead)]
  b=held[(held.seed==seed)&(held.horizon_week==lead)]
  assert len(a)==149 and len(b)==150,(seed,lead,len(a),len(b))
  residual=np.abs(a.true_co2_ppm-a.cvae_mean_ppm).to_numpy()
  q=min(1.0,math.ceil((len(a)+1)*.9)/len(a))
  radius=float(np.quantile(residual,q,method='higher'))
  original=((b.true_co2_ppm>=b.cvae_q05_ppm)&(b.true_co2_ppm<=b.cvae_q95_ppm))
  after=(abs(b.true_co2_ppm-b.cvae_mean_ppm)<=radius)
  report.append({'seed':int(seed),'lead':lead,'calibration_windows':len(a),'later_test_windows':len(b),
    'radius_ppm':radius,'original_90pct_test_coverage':float(original.mean()),
    'posthoc_90pct_test_coverage':float(after.mean()),'interval_width_ppm':radius*2})
  for _,r in b.iterrows():
   predictions.append({'seed':int(seed),'first_forecast_week':r.first_forecast_week.date().isoformat(),
     'horizon_week':lead,'measured_ppm':float(r.true_co2_ppm),'forecast_mean_ppm':float(r.cvae_mean_ppm),
     'lower_ppm':float(r.cvae_mean_ppm-radius),'upper_ppm':float(r.cvae_mean_ppm+radius),
     'covered':int(abs(r.true_co2_ppm-r.cvae_mean_ppm)<=radius)})
out=pd.DataFrame(report);out.to_csv(args.outdir/'per_horizon_observed_coverage.csv',index=False)
pd.DataFrame(predictions).to_csv(args.outdir/'independent_later_test_predictions.csv',index=False)
summary={'calibration_windows':149,'later_test_windows':150,
 'mean_original_test_coverage':float(out.original_90pct_test_coverage.mean()),
 'mean_posthoc_test_coverage':float(out.posthoc_90pct_test_coverage.mean()),
 'mean_interval_width_ppm':float(out.interval_width_ppm.mean()),
 'caveat':'Overlapping dependent observations, not distribution-free conformal guarantee'}
(args.outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
