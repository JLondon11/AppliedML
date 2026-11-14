/* Original measured Yeh held-out results only; do not substitute generated mix predictions.
 * Run: node plot_verified_heldout.mjs heldout_test_predictions.csv figures/concrete_verified_heldout.svg
 * Zero external dependencies.
 */
import fs from 'node:fs';
import path from 'node:path';

export function render(csv) {
  const lines = csv.trim().split(/\r?\n/);
  const header = lines.shift().split(',');
  const iy = header.indexOf('strength_measured'), ip = header.indexOf('strength_predicted');
  if (iy < 0 || ip < 0) throw Error('Required observed and predicted columns missing');
  const pts = lines.map(line => {
    const parts = line.split(',');
    return [Number(parts[iy]), Number(parts[ip])];
  });
  if (pts.length !== 229 || pts.some(([a,b])=>!Number.isFinite(a)||!Number.isFinite(b)))
    throw Error('Expected 229 finite held-out original measurement/prediction pairs');
  const rmse = Math.sqrt(pts.reduce((s,[a,b])=>s+(a-b)**2,0)/pts.length);
  const mae = pts.reduce((s,[a,b])=>s+Math.abs(a-b),0)/pts.length;
  const esc = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
  const left = {x:100,y:108,w:455,h:300},right={x:695,y:108,w:455,h:300};
  const mapX = (v,p,min,max)=>p.x+p.w*(v-min)/(max-min);
  const mapY = (v,p,min,max)=>p.y+p.h*(max-v)/(max-min);
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="525" viewBox="0 0 1240 525"><rect width="1240" height="525" fill="#ffffff"/><style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#293c52}.heading{font-size:23px;font-weight:600}.subtitle{font-size:13px;fill:#617489}.panel{font-size:15px;font-weight:600}.tick{font-size:11px;fill:#607285}.axis{font-size:13px}.note{font-size:12px;fill:#3a596c}</style><text x="66" y="40" class="heading">Concrete compressive strength | held-out observations</text><text x="66" y="64" class="subtitle">Original Yeh laboratory measurements • 229 independent test rows • no generated-mix strength measurements</text>`];
  function frame(p,title,xlabel,ylabel,xmin,xmax,ymin,ymax) {
    parts.push(`<text x="${p.x}" y="${p.y-21}" class="panel">${esc(title)}</text><rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" fill="#fbfcfd" stroke="#bdc8d2"/>`);
    for (let i=0;i<=4;i++){
      const vx=xmin+i*(xmax-xmin)/4,vy=ymin+i*(ymax-ymin)/4;
      const x=mapX(vx,p,xmin,xmax),y=mapY(vy,p,ymin,ymax);
      parts.push(`<line x1="${x}" y1="${p.y}" x2="${x}" y2="${p.y+p.h}" stroke="#e2e8ec"/><line x1="${p.x}" y1="${y}" x2="${p.x+p.w}" y2="${y}" stroke="#e2e8ec"/><text x="${x}" y="${p.y+p.h+19}" text-anchor="middle" class="tick">${vx.toFixed(0)}</text><text x="${p.x-9}" y="${y+4}" text-anchor="end" class="tick">${vy.toFixed(0)}</text>`);
    }
    parts.push(`<text x="${p.x+p.w/2}" y="${p.y+p.h+47}" text-anchor="middle" class="axis">${esc(xlabel)}</text><text x="${p.x-65}" y="${p.y+p.h/2}" transform="rotate(-90 ${p.x-65} ${p.y+p.h/2})" text-anchor="middle" class="axis">${esc(ylabel)}</text>`);
  }
  frame(left,'A  Predicted versus measured strength','Measured MPa','Predicted MPa',0,90,0,90);
  frame(right,'B  Prediction error by measured strength','Measured MPa','Prediction − measurement (MPa)',0,90,-50,50);
  parts.push(`<line x1="${mapX(0,left,0,90)}" y1="${mapY(0,left,0,90)}" x2="${mapX(90,left,0,90)}" y2="${mapY(90,left,0,90)}" stroke="#ad735d" stroke-width="2" stroke-dasharray="6 5"/><line x1="${right.x}" y1="${mapY(0,right,-50,50)}" x2="${right.x+right.w}" y2="${mapY(0,right,-50,50)}" stroke="#ad735d" stroke-width="2" stroke-dasharray="6 5"/>`);
  for (const [actual,pred] of pts){
    parts.push(`<circle cx="${mapX(actual,left,0,90).toFixed(2)}" cy="${mapY(pred,left,0,90).toFixed(2)}" r="2.7" fill="#237b89" fill-opacity=".58"/>`);
    parts.push(`<circle cx="${mapX(actual,right,0,90).toFixed(2)}" cy="${mapY(pred-actual,right,-50,50).toFixed(2)}" r="2.7" fill="#385f89" fill-opacity=".58"/>`);
  }
  parts.push(`<text x="100" y="494" class="note">Held-out RMSE ${rmse.toFixed(3)} MPa  |  MAE ${mae.toFixed(3)} MPa  |  dashed lines show ideal agreement / zero error</text></svg>`);
  return parts.join('');
}
if (process.argv[1] && import.meta.url.endsWith(path.basename(process.argv[1]))) {
  const source = process.argv[2] || new URL('./heldout_test_predictions.csv',import.meta.url);
  const dest = process.argv[3] || path.join(path.dirname(new URL(import.meta.url).pathname),'figures','concrete_verified_heldout.svg');
  const svg=render(fs.readFileSync(source,'utf8'));
  fs.mkdirSync(path.dirname(String(dest)),{recursive:true});
  fs.writeFileSync(dest,svg);
  console.log('Measured-record scientific SVG written:',String(dest));
}
