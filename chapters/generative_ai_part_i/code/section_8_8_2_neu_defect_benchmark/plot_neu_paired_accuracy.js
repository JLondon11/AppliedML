/** Produce a measured-data NEU steel-defect seed and paired-control SVG.
 * Node 18+, no dependencies.
 * Input: replicated_seed_metrics_pass13.csv (authentic held-out 360-image results).
 * Output: figures/neu_seed_paired_accuracy.svg
 * Does not train models or imply GAN generalization.
 */
const fs=require("node:fs");
const path=require("node:path");
const root=__dirname;
const text=fs.readFileSync(path.join(root,"replicated_seed_metrics_pass13.csv"),"utf8").trim();
const [header,...lines]=text.split(/\r?\n/);
if(header!=="seed,arm,test_accuracy,test_macro_f1")throw Error("wrong CSV schema");
const rows=lines.map(l=>{const [seed,arm,acc,f1]=l.split(",");return {seed:Number(seed),arm,acc:Number(acc),f1:Number(f1)};});
const seeds=[20261106,20261107,20261108];
const arms=["real_only","flip_augmentation","gan_augmented","real_repeat_augmented"];
if(rows.length!==12||new Set(rows.map(x=>x.seed+"|"+x.arm)).size!==12||
  !seeds.every(s=>arms.every(a=>rows.some(x=>x.seed===s&&x.arm===a)))||
  !rows.every(x=>x.acc>=0&&x.acc<=1&&x.f1>=0&&x.f1<=1))
  throw Error("incomplete or invalid authentic per-seed scores");
function renderFigure(rows){
  const seeds=[20261106,20261107,20261108], arms=["real_only","flip_augmentation","gan_augmented","real_repeat_augmented"];
  const palette={real_only:"#26718a",flip_augmentation:"#8a8d99",gan_augmented:"#b65a41",real_repeat_augmented:"#23856f"};
  const labels={real_only:"Real only",flip_augmentation:"Horizontal flip",gan_augmented:"Conditional GAN",real_repeat_augmented:"Real repeats + flips"};
  const get=(s,a)=>{const r=rows.find(x=>x.seed===s&&x.arm===a);if(!r)throw Error("missing score");return r.acc;};
  const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  const t=(x,y,str,size=16,color="#203444",anchor="start",weight=400)=>`<text x="${x}" y="${y}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" fill="${color}">${esc(str)}</text>`;
  const line=(x1,y1,x2,y2,color="#d9e2e7",w=1,dash="")=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash?` stroke-dasharray="${dash}"`:""}/>`;
  const circ=(x,y,c,r=6)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" stroke="#fff" stroke-width="1.6"/>`;
  let svg=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 690" width="1280" height="690" role="img" aria-label="Measured held-out NEU defect classification accuracy by three random seeds with paired GAN accuracy deficits against real-image baselines">`,`<rect width="1280" height="690" fill="#fff"/>`,t(58,55,"NEU steel-defect classification • real held-out evidence",27,"#172c3b","start",700),t(58,84,"1,800 original photographs (1 duplicate excluded)  |  1,080 train · 359 validation · 360 held-out test",15,"#526778"),t(58,110,"Three training seeds; same 360 held-out photographs scored in each arm. Generated GAN images used only for training.",15,"#526778")];
  const X=[115,315,515], top=185,bottom=525, ymin=.35,ymax=.88, ypos=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top);
  svg.push(t(58,154,"A   Original held-out classification accuracy",19,"#243b4b","start",700));
  for(let v=.4;v<=.8+1e-6;v+=.1){const y=ypos(v);svg.push(line(95,y,565,y),t(84,y+5,`${Math.round(v*100)}%`,13,"#687986","end"))}
  svg.push(line(95,bottom,565,bottom,"#83919c"),t(330,578,"Independent training seed",15,"#536777","middle"));
  seeds.forEach((s,i)=>svg.push(t(X[i],549,`Seed ${i+1}`,14,"#526778","middle")));
  for(const arm of arms){const pts=seeds.map((s,i)=>[X[i],ypos(get(s,arm))]);svg.push(`<polyline points="${pts.map(p=>p.join(",")).join(" ")}" fill="none" stroke="${palette[arm]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`);pts.forEach(p=>svg.push(circ(...p,palette[arm])));}
  svg.push(t(650,154,"B   Paired accuracy effect of GAN augmentation",19,"#243b4b","start",700));
  const x0=700,x1=1202, zero=1175, factor=15, xp=v=>zero+factor*v; const deltas=seeds.map(s=>({s,real:(get(s,"gan_augmented")-get(s,"real_only"))*100,repeat:(get(s,"gan_augmented")-get(s,"real_repeat_augmented"))*100}));
  for(let tick=-30;tick<=0;tick+=5){const x=xp(tick);svg.push(line(x,200,x,bottom,tick===0?"#8496a4":"#e2e8eb",tick===0?2:1),t(x,543,`${tick}`,13,"#637889","middle"))}
  svg.push(t(934,579,"GAN minus reference accuracy (percentage points)",15,"#536777","middle"));
  deltas.forEach((d,i)=>{const y=240+i*93;svg.push(t(664,y+6,`Seed ${i+1}`,14,"#536777"),line(xp(d.repeat),y,xp(d.real),y,"#adb9c1",2),circ(xp(d.real),y,"#26718a",7),circ(xp(d.repeat),y,"#b65a41",7));});
  const mean=key=>deltas.reduce((v,d)=>v+d[key],0)/3;
  svg.push(t(650,633,`Mean GAN − real-only: ${mean("real").toFixed(2)} pp    |    GAN − matched real repeats: ${mean("repeat").toFixed(2)} pp`,16,"#8c4533","start",700));
  let lx=72;for(const arm of arms){svg.push(line(lx,615,lx+28,615,palette[arm],4),circ(lx+14,615,palette[arm],4),t(lx+37,620,labels[arm],14,"#4a6070"));lx+={"real_only":185,"flip_augmentation":210,"gan_augmented":205,"real_repeat_augmented":245}[arm];}
  svg.push(t(58,669,"Source: replicated_seed_metrics_pass13.csv (12 recorded source-model evaluations). No independent factory/generalization claim.",12,"#748390"));
  svg.push(`</svg>`);return svg.join("\n")+"\n";
}
const out=path.join(root,"figures","neu_seed_paired_accuracy.svg");
fs.mkdirSync(path.dirname(out),{recursive:true});
fs.writeFileSync(out,renderFigure(rows),"utf8");
console.log("Wrote original-data NEU paired-accuracy SVG:",out);
