/* UCI 1030 concrete laboratory test data, grouped-mix holdout. Native Node JS, no dependencies. */
import fs from 'node:fs';
const compute=function compute(csv){
  const input=csv.trim().split(/\r?\n/).slice(1).map((l,i)=>({id:i+1,v:l.split(",").map(Number)}));
  if(input.length!==1030||input.some(o=>o.v.length!==9||o.v.some(x=>!Number.isFinite(x))))throw Error("source format");
  let state=20261110;function rand(){state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296}
  function normal(){return Math.sqrt(-2*Math.log(Math.max(1e-10,rand())))*Math.cos(2*Math.PI*rand())}
  const group=new Map();for(const r of input){const key=r.v.slice(0,7).join("|");if(!group.has(key))group.set(key,[]);group.get(key).push(r)}
  let keys=[...group.keys()];for(let i=keys.length-1;i>0;i--){let j=Math.floor(rand()*(i+1));[keys[i],keys[j]]=[keys[j],keys[i]]}
  const ntrain=Math.floor(keys.length*.65),nval=Math.floor(keys.length*.15);
  const split={train:keys.slice(0,ntrain).flatMap(k=>group.get(k)),validation:keys.slice(ntrain,ntrain+nval).flatMap(k=>group.get(k)),test:keys.slice(ntrain+nval).flatMap(k=>group.get(k))};
  const mean=Array(8).fill(0),sd=Array(8).fill(0);
  const enc=r=>r.v.slice(0,8).map((x,i)=>i===7?Math.log1p(x):x);
  for(const r of split.train){const a=enc(r);for(let i=0;i<8;i++)mean[i]+=a[i]/split.train.length}
  for(const r of split.train){const a=enc(r);for(let i=0;i<8;i++)sd[i]+=(a[i]-mean[i])**2/split.train.length}
  for(let i=0;i<8;i++)sd[i]=Math.sqrt(sd[i])||1;
  const xs=r=>enc(r).map((v,i)=>(v-mean[i])/sd[i]);
  const tr=split.train.map(r=>({id:r.id,x:xs(r),y:r.v[8]}));
  function knn(row,k){const x=xs(row), d=tr.map(r=>{let dist=0;for(let i=0;i<8;i++)dist+=(r.x[i]-x[i])**2;return {dist,y:r.y}}).sort((a,b)=>a.dist-b.dist).slice(0,k);return d.reduce((s,v)=>s+v.y/(.1+Math.sqrt(v.dist)),0)/d.reduce((s,v)=>s+1/(.1+Math.sqrt(v.dist)),0)}
  function scores(rows,k){const vals=rows.map(r=>[r.v[8],knn(r,k)]);return {rmse:Math.sqrt(vals.reduce((s,v)=>s+(v[0]-v[1])**2,0)/vals.length),mae:vals.reduce((s,v)=>s+Math.abs(v[0]-v[1]),0)/vals.length,n:vals.length}}
  const kCandidates=[3,5,9,15],validation=kCandidates.map(k=>({k,...scores(split.validation,k)}));
  const bestK=[...validation].sort((a,b)=>a.rmse-b.rmse)[0].k;
  const scoreTest=scores(split.test,bestK);
  const trainStrengthMean=split.train.reduce((s,r)=>s+r.v[8],0)/split.train.length;
  const baseline={rmse:Math.sqrt(split.test.reduce((s,r)=>s+(r.v[8]-trainStrengthMean)**2,0)/split.test.length),mae:split.test.reduce((s,r)=>s+Math.abs(r.v[8]-trainStrengthMean),0)/split.test.length};
  const held=split.test.map(r=>({id:r.id,age:r.v[7],cement:r.v[0],strength_measured:r.v[8],strength_predicted:knn(r,bestK),water:r.v[3],binder:r.v[0]+r.v[1]+r.v[2]}));
  const day28=held.filter(r=>r.age===28),cementMedian=[...split.train.map(r=>r.v[0])].sort((a,b)=>a-b)[Math.floor(split.train.length/2)];
  const feasible=day28.filter(r=>r.cement<=cementMedian && r.water/r.binder<=.65);
  const cut=40, topN=Math.min(15,feasible.length);
  const best=[...feasible].sort((a,b)=>b.strength_predicted-a.strength_predicted).slice(0,topN);
  const hits=best.filter(r=>r.strength_measured>=cut).length;
  const measuredEligible=feasible.filter(r=>r.strength_measured>=cut).length;
  const train28=split.train.filter(r=>r.v[7]===28),high=train28.filter(r=>r.v[8]>=40);
  const designStats={};const designrows=[];
  for(const arm of ["high_strength_conditioned","unconditioned_training_mixes"]){
    let nvalid=0,passed=0,above=0,cementSum=0,novel=0;
    const bases=arm==="high_strength_conditioned"?high:train28;
    const cov=Array(7).fill(0);for(const r of bases)for(let j=0;j<7;j++)cov[j]+=r.v[j]/bases.length;
    const dev=Array(7).fill(0);for(const r of bases)for(let j=0;j<7;j++)dev[j]+=(r.v[j]-cov[j])**2/bases.length;
    for(let j=0;j<7;j++)dev[j]=Math.sqrt(dev[j]);
    for(let i=0;i<300;i++){
      const parent=bases[Math.floor(rand()*bases.length)],vals=parent.v.slice(0,7).map((v,j)=>Math.max(0,v+normal()*.08*dev[j]));
      const binder=vals[0]+vals[1]+vals[2],ratio=binder>0?vals[3]/binder:Infinity;
      const physical=ratio>=.25&&ratio<=.65&&binder>=200&&binder<=650&&vals.reduce((s,v)=>s+v,0)>=1500&&vals.reduce((s,v)=>s+v,0)<=2700;
      const surrogate=knn({v:[...vals,28,0]},bestK);
      if(physical)passed++;if(physical&&surrogate>=cut)above++;
      cementSum+=vals[0];nvalid++;
      if(i<100)designrows.push({arm,idx:i+1,cement:vals[0],slag:vals[1],flyash:vals[2],water:vals[3],superplasticizer:vals[4],coarseaggregate:vals[5],fineaggregate:vals[6],age:28,predicted_strength:surrogate,physical_screen_pass:physical?1:0,source:"model generated, not laboratory tested"});
    }
    designStats[arm]={n_generated:nvalid,n_rule_feasible:passed,n_rule_feasible_with_predicted_strength_40_or_more:above,mean_cement:cementSum/nvalid};
  }
  return {source_rows:input.length,unique_mix_groups:keys.length,split_rows:Object.fromEntries(Object.entries(split).map(([k,v])=>[k,v.length])),split_groups:{train:ntrain,validation:nval,test:keys.length-ntrain-nval},validation,chosen_k:bestK,test_knn:scoreTest,test_mean_baseline:baseline,day28:{n_heldout_28day:day28.length,cement_median_train:cementMedian,n_feasible_measured_test:feasible.length,top_n:topN,observed_high_strength_hits_selected:hits,observed_high_strength_eligible:measuredEligible,expected_random_hits:topN*measuredEligible/Math.max(1,feasible.length),selection:best},designStats,designrows,heldout:held,source_constraints:{age_target:28,water_binder:[.25,.65],binder:[200,650],total_materials:[1500,2700]},warnings:["generated candidate strengths are surrogate predictions, never laboratory measured","retrospective held-out selection uses only measured source recipes","age and mix group split prevents identical mixes spanning partitions","physical screen is simplified material bounds, not verified production constructability"]};
};
const csv=fs.readFileSync(process.argv[2]||new URL('./Concrete_Data_Yeh.csv',import.meta.url),'utf8');
const results=compute(csv);
fs.writeFileSync(process.argv[3]||'results.json',JSON.stringify(results,null,2)+'\n');
console.log(JSON.stringify({split:results.split_rows,validation:results.validation,model:results.test_knn,baseline:results.test_mean_baseline,screen:results.day28,generated:results.designStats},null,2));
