/** Exact finite-population ESOL random-selection tail figure.
 * Numerical population size, observed successes and two ranking hit counts are
 * independently verified in verified_original_ESOL_random_null.json and actual
 * measured 226-test record CSV. No invented results or new molecule synthesis.
 */
const fs=require("node:fs"),path=require("node:path");
const ROOT=__dirname;
const nullData=JSON.parse(fs.readFileSync(path.join(ROOT,"verified_original_ESOL_random_null.json"),"utf8"));
if(nullData.n_original_heldout_assays!==226||nullData.observed_high_solubility_assays!==59||
 nullData.model_arms.ridge.measured_hits_top20!==14||
 nullData.model_arms.published_ESOL.measured_hits_top20!==17)
 throw Error("Source original measured-assay null has changed; revalidate experiment first");
function render(){
const N=226,K=59,n=20,c=(a,b)=>{let x=1;for(let i=1;i<=b;i++)x*=((a-b+i)/i);return x};
const p=Array.from({length:n+1},(_,k)=>c(K,k)*c(N-K,n-k)/c(N,n)),sf=k=>p.slice(k).reduce((a,b)=>a+b,0);
const txt=(x,y,t,z=15,col="#274459",w=400,anchor="start")=>`<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${col}" font-family="Arial,Helvetica,sans-serif" font-size="${z}" font-weight="${w}">${t}</text>`;
const line=(x,y,u,v,col="#cbd7dd",stroke=1)=>`<line x1="${x}" y1="${y}" x2="${u}" y2="${v}" stroke="${col}" stroke-width="${stroke}"/>`;
let svg=[`<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="650" viewBox="0 0 1140 650" role="img" aria-label="Exact random-rank null for measured ESOL held-out compounds: hypergeometric survival probabilities for top-20 measured hits and observed ridge 14 and published-source score 17"><rect width="1140" height="650" fill="#fff"/>`,txt(56,50,"ESOL • measured hit enrichment against exact random selection",25,"#193d4c",700),txt(56,77,"226 existing measured test compounds • 59 above the training-derived solubility threshold • select 20 without replacement",14,"#5e7481")];
const x0=125,x1=1020,y0=143,y1=507,x=k=>x0+k*(x1-x0)/20,y=v=>y1-Math.log10(1/v)*(y1-y0)/13.0;
for(const exp of [0,2,4,6,8,10,12]){const yy=y(10**(-exp));svg.push(line(x0,yy,x1,yy,"#e2e9ed"),txt(105,yy+4,exp===0?"1":`10⁻${exp}`,12,"#627988",400,"end"))}
svg.push(txt(56,129,"P(random top-20 includes at least k observed high-solubility assays)",15,"#244b5c"));
const pts=Array.from({length:21},(_,k)=>[x(k),y(sf(k))]);
svg.push(`<polyline points="${pts.map(v=>v.join(",")).join(" ")}" fill="none" stroke="#287b8a" stroke-width="3" stroke-linejoin="round"/>`);
for(let k=0;k<=20;k++)svg.push(`<circle cx="${x(k)}" cy="${y(sf(k))}" r="3.2" fill="#287b8a"/>`);
const yRef=y(1e-13); // lower than chart?
for(const [k,col] of [[14,"#b45e49"],[17,"#725b9b"]]){svg.push(line(x(k),y(sf(k)),x(k),y1,col,2),`<circle cx="${x(k)}" cy="${y(sf(k))}" r="6.8" fill="${col}" stroke="white" stroke-width="2"/>`)}
for(let k=0;k<=20;k+=2)svg.push(txt(x(k),y1+24,String(k),12,"#5e7686",400,"middle"));
svg.push(txt((x0+x1)/2,553,"Number of observed high-solubility assays in 20 selected compounds",15,"#3e5c6b",400,"middle"),txt(126,588,"Random expected: 5.22 hits; central 95% random range: 2–9 hits.",15,"#24506c",700));
svg.push(txt(680,191,"Ridge: 14 / 20 hits",18,"#b45e49",700),txt(680,214,"Exact tail: 1.96 × 10⁻⁵",14,"#7d514c"));
svg.push(txt(790,311,"Published source score: 17 / 20",16,"#725b9b",700),txt(790,333,"Exact tail: 1.06 × 10⁻⁸",13,"#675c83"));
svg.push(txt(56,622,"Observed source labels only. Published ESOL predictor training independence unknown; neither arm tested newly generated molecules.",12,"#657d8a"));
svg.push(`</svg>\n`);return svg.join("\n");
}
const output=path.join(ROOT,"figures","esol_exact_random_ranking_tail.svg");
fs.mkdirSync(path.dirname(output),{recursive:true});
fs.writeFileSync(output,render(),"utf8");
console.log(output);
