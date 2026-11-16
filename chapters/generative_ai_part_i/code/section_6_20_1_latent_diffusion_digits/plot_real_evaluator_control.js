/** Regenerate a real-evaluator / saved-generator comparison using measured CSV evidence.
 * Run: node plot_real_evaluator_control.js
 * Requires the genuine original-evaluator experiment outputs already committed.
 * Does NOT train models, create data, or certify generation fidelity.
 */
const fs=require("node:fs"),path=require("node:path");
const root=__dirname;
const dir=path.join(root,"original_evaluator_control");
const observed=fs.readFileSync(path.join(dir,"real_evaluator_vs_generated_class.csv"),"utf8");
const confusion=fs.readFileSync(path.join(dir,"real_evaluator_confusion.csv"),"utf8");
function render(realText, confusionText) {
  const rowlines=realText.trim().split(/\r?\n/).slice(1);const rows=rowlines.map(s=>{const p=s.split(",");return {k:+p[0],n:+p[1],real:+p[3],gen:+p[6]};});
  const cm=confusionText.trim().split(/\r?\n/).slice(1).map(s=>s.split(",").slice(1).map(Number));
  if(rows.length!==10||cm.length!==10||cm.some(x=>x.length!==10)||rows.reduce((x,r)=>x+r.n,0)!==270||cm.reduce((x,row)=>x+row.reduce((a,b)=>a+b,0),0)!==270)throw Error("Invalid original evaluator evidence");
  const realhits=cm.reduce((n,row,i)=>n+row[i],0);if(realhits!==257)throw Error("Real-control accuracy discrepancy");
  const xtext=(x,y,s,size=15,color="#284558",w=400,anchor="start")=>`<text x="${x}" y="${y}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" fill="${color}" font-weight="${w}" text-anchor="${anchor}">${s}</text>`;
  const ln=(x,y,u,v,color="#ccd7df",width=1)=>`<line x1="${x}" y1="${y}" x2="${u}" y2="${v}" stroke="${color}" stroke-width="${width}"/>`;
  const dots=(x,y,color,r=6)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" stroke="#fff" stroke-width="1.6"/>`;
  let o=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 640" width="1240" height="640" role="img" aria-label="Original-digit real-test classifier recall compared with saved conditional-generator label agreement for each digit, and real-test confusion matrix"><rect width="1240" height="640" fill="white"/>`];
  o.push(xtext(58,50,"Real-data classifier control for digit-conditional latent diffusion",25,"#19394a",700));
  o.push(xtext(58,76,"Original 1,797 digit scans; 1,257 training / 270 validation / 270 independent real tests",14,"#506979"));
  o.push(xtext(58,101,"Real held-out evaluator: 257 / 270 correct (95.19%)    •    Saved conditional generated agreement: 778 / 1,000 (77.8%)",14,"#506979"));
  o.push(xtext(58,146,"A  Real-test recognition vs generated-output agreement",19,"#23465c",700));
  const start=85, end=602, top=194, gap=33, xp=p=>start+p*(end-start);
  for(const t of [0,.2,.4,.6,.8,1]){const x=xp(t);o.push(ln(x,179,x,511,"#e2e9ed"),xtext(x,530,`${Math.round(t*100)}%`,12,"#647884",400,"middle"));}
  for(const r of rows){const y=top+r.k*gap;o.push(xtext(65,y+5,String(r.k),13,"#3a5769",700,"end"),ln(xp(r.real),y,xp(r.gen),y,"#9fb5c0",2),dots(xp(r.real),y,"#23748c"),dots(xp(r.gen),y,"#b66b4d"));}
  o.push(dots(76,566,"#23748c"),xtext(91,571,"Real held-out digit recall",14));
  o.push(dots(315,566,"#b66b4d"),xtext(330,571,"Conditional generated agreement",14));
  o.push(xtext(58,606,"Different source populations and denominators; comparison is descriptive, not paired.",12,"#647884"));
  o.push(ln(650,132,650,610,"#dfe7eb"));
  o.push(xtext(685,146,"B  Evaluator confusion on original held-out scans",19,"#23465c",700));
  const cell=42, sx=730,sy=188;
  for(let i=0;i<10;i++){o.push(xtext(sx+cell*i+cell/2,179,String(i),12,"#4c6374",400,"middle"),xtext(sx-10,sy+cell*i+26,String(i),12,"#4c6374",400,"end"));for(let j=0;j<10;j++){const v=cm[i][j],intensity=v/30,shade=Math.round(246-178*intensity),fill=`rgb(${shade},${Math.min(251,shade+10)},${Math.min(255,shade+19)})`;o.push(`<rect x="${sx+cell*j}" y="${sy+cell*i}" width="${cell-2}" height="${cell-2}" rx="3" fill="${fill}"/>`);if(v)o.push(xtext(sx+cell*j+(cell-2)/2,sy+cell*i+25,String(v),13,v>=18?"#fff":"#24516a",v>=18?700:400,"middle"));}}
  o.push(xtext(sx+208,630,"Predicted digit on observed test images",14,"#586f7e",400,"middle"));
  o.push(xtext(699,606,"Confusion is measured on 270 actual scans, not generated images.",12,"#647884"));
  o.push(`</svg>\n`);return o.join("\n");
}
const generated=render(observed,confusion);
fs.writeFileSync(path.join(dir,"real_classifier_control_summary.svg"),generated,"utf8");
console.log("Regenerated authentic data evaluator control figure");
