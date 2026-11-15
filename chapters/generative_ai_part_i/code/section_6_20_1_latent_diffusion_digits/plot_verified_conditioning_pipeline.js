/** Render an SVG from ORIGINAL recorded digit-generation class-count evidence.
 * Run: node plot_verified_conditioning_pipeline.js
 * This is a classifier-based generator diagnostic, not a human image-fidelity
 * benchmark, a text-to-image experiment, or a new model-training run.
 */
const fs=require("node:fs");
const path=require("node:path");
const root=__dirname;
const raw=fs.readFileSync(path.join(root,"figure_6_44_class_counts_pass27.csv"),"utf8").trim();
const [header,...lines]=raw.split(/\r?\n/);
if(header!=="arm,label,correct,n")throw Error("Wrong original evidence schema");
const rows=lines.map(s=>{const [arm,label,correct,n]=s.split(",");return{arm,label,correct:Number(correct),n:Number(n)};});
if(rows.some(r=>!Number.isInteger(r.correct)||!Number.isInteger(r.n)||r.correct<0||r.correct>r.n))throw Error("Invalid source counts");
function render(rows) {
 const s=(x)=>String(x).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
 const tx=(x,y,t,z=16,c="#294252",bold=false,anchor="start")=>`<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${z}" font-family="Arial,Helvetica,sans-serif" font-weight="${bold?700:400}" fill="${c}">${s(t)}</text>`;
 const line=(x1,y1,x2,y2,color="#dbe4e8",width=1)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>`;
 const get=(a,l)=>rows.find(x=>x.arm===a&&x.label===l);
 const labels=Array.from({length:10},(_,i)=>String(i));
 if(rows.length!==22||!labels.every(l=>get("conditional",l)?.n===100&&get("unconditional",l)?.n===100))throw Error("Invalid digit class sample counts");
 for(const a of ["conditional","unconditional"]){const total=labels.reduce((v,l)=>v+get(a,l).correct,0);if(total!==get(a,"overall")?.correct||get(a,"overall")?.n!==1000)throw Error("Overall and class totals differ");}
 const rate=(a,l)=>100*get(a,l).correct/get(a,l).n;
 const clrC="#287b87",clrU="#bd7453";
 let out=[`<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="695" viewBox="0 0 1300 695" role="img" aria-label="Real scanned digit latent diffusion classifier agreement by requested/reference class, and authentic experiment pipeline">`,`<rect width="1300" height="695" fill="#ffffff"/>`,tx(58,52,"Class-conditioned latent diffusion on authentic handwritten digits",26,"#173448",true),tx(58,79,"Measured output agreement with a classifier trained only on genuine digit images",15,"#526b7c"),tx(58,124,"A  Per-digit classifier agreement",20,"#1c4257",true)];
 const xmin=132,xmax=721,y0=191,step=39;const xp=v=>xmin+(xmax-xmin)*v/100;
 for(const tick of [0,20,40,60,80,100]){const x=xp(tick);out.push(line(x,166,x,568,"#e6ecf0"),tx(x,593,`${tick}%`,12,"#647988",false,"middle"));}
 out.push(tx(58,180,"Digit",14,"#526b7c"),tx(440,623,"Agreement rate across 100 generated records per digit and arm",14,"#526b7c",false,"middle"));
 for(let i=0;i<10;i++){const l=String(i),y=y0+i*step,c=rate("conditional",l),u=rate("unconditional",l);out.push(tx(83,y+5,l,15,"#365468",true,"middle"),line(xp(u),y,xp(c),y,"#b8c8d0",2),`<circle cx="${xp(c)}" cy="${y}" r="6.4" fill="${clrC}" stroke="#fff" stroke-width="1.5"/>`,`<circle cx="${xp(u)}" cy="${y}" r="5.8" fill="${clrU}" stroke="#fff" stroke-width="1.5"/>`);}
 out.push(`<circle cx="85" cy="648" r="6" fill="${clrC}"/>`,tx(100,653,"Conditional",13,"#34546b"),`<circle cx="221" cy="648" r="6" fill="${clrU}"/>`,tx(236,653,"Unconditional, comparison labels",13,"#34546b"));
 out.push(line(762,132,762,656,"#e1e8eb"),tx(800,124,"B  Actually evaluated procedure",20,"#1c4257",true));
 const boxes=[
 {y:155,title:"Original observed scans",detail:"1,797 real 8×8 handwritten digits"},
 {y:235,title:"Partitioned source data",detail:"1,257 train / 270 validation / 270 test"},
 {y:315,title:"Trained 16-dimensional VAE",detail:"25 epochs; single recorded model seed"},
 {y:395,title:"Latent diffusion sampling",detail:"48 steps; conditional and unconditional"},
 {y:475,title:"Independent digit classifier",detail:"Trained on original real training scans"},
 ];
 boxes.forEach((b,i)=>{out.push(`<rect x="802" y="${b.y}" width="432" height="64" rx="9" fill="${i%2?"#eff5f7":"#f4f7f9"}" stroke="#d1dde3"/>`,tx(822,b.y+25,b.title,16,"#204556",true),tx(822,b.y+47,b.detail,13,"#607786"));if(i<boxes.length-1)out.push(line(1018,b.y+64,1018,b.y+79,"#8096a1",2),`<path d="M1013 ${b.y+74} l5 6 l5 -6" fill="none" stroke="#8096a1" stroke-width="2"/>`);});
 out.push(tx(802,574,"77.8% conditional  •  9.4% unconditional",17,"#245f71",true),tx(802,601,"Classifier-label agreement, not human-rated image fidelity.",13,"#526b7c"),tx(802,626,"Only one fitted generator seed; classes 1 and 2 are weak.",13,"#526b7c"));
 out.push(line(58,675,1240,675,"#e5ebee"),tx(58,690,"Source: figure_6_44_class_counts_pass27.csv; model output classes, not original test-image accuracy or natural-image generation.",11,"#6a7e8a"));
 return out.join("\n")+"\n</svg>\n";
}
const out=path.join(root,"figures","verified_digit_conditioning_science.svg");
fs.mkdirSync(path.dirname(out),{recursive:true});
fs.writeFileSync(out,render(rows),"utf8");
console.log("Wrote verified source-count digit conditioning graphic",out);
