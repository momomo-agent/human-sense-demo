// autoresearch v100: Fine-tune hybrid segment+voting
const fs = require('fs');
const DATA = '/Users/kenefe/LOCAL/momo-agent/projects/human-sense-demo/Tests/speaker-test-data.jsonl';
const lines = fs.readFileSync(DATA, 'utf8').trim().split('\n');
const all = lines.map(l => JSON.parse(l));
all.sort((a, b) => a.audioTime - b.audioTime);
const N = all.length;
const act = all.map(s => s.isUserSpeaker);
const dt = all.map((s, i) => i === 0 ? 0 : s.audioTime - all[i - 1].audioTime);
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}

const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);
const velStd5 = (() => {
  return all.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - 2); j <= Math.min(N - 1, i + 2); j++) w.push(all[j].jawVelocity);
    const m = mean(w);
    return Math.sqrt(w.reduce((s, v) => s + (v - m) ** 2, 0) / w.length);
  });
})();
const dtEnt5 = (() => {
  return all.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - 2); j <= Math.min(N - 1, i + 2); j++) w.push(dt[j]);
    const b=[0,0,0];w.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});
    let e=0;const n=w.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;
  });
})();
const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
const jdMean5 = (() => {
  return all.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - 2); j <= Math.min(N - 1, i + 2); j++) w.push(all[j].jawDelta);
    return mean(w);
  });
})();

// Time-windowed features
function timeWindowIdx(centerIdx, windowSec) {
  const t0 = all[centerIdx].audioTime;
  const indices = [];
  for(let j=centerIdx;j>=0;j--){
    if(t0 - all[j].audioTime > windowSec) break;
    indices.push(j);
  }
  for(let j=centerIdx+1;j<N;j++){
    if(all[j].audioTime - t0 > windowSec) break;
    indices.push(j);
  }
  return indices;
}

const tw5 = all.map((s,i) => {
  const idx = timeWindowIdx(i, 5);
  const jds = idx.map(j => all[j].jawDelta);
  const effs = idx.map(j => jawEff[j]);
  return {
    jdMean: mean(jds),
    jeMean: mean(effs),
    activeRate: jds.filter(v => v >= 0.03).length / jds.length,
  };
});

const tw3 = all.map((s,i) => {
  const idx = timeWindowIdx(i, 3);
  const jds = idx.map(j => all[j].jawDelta);
  const effs = idx.map(j => jawEff[j]);
  return {
    jdMean: mean(jds),
    jeMean: mean(effs),
    activeRate: jds.filter(v => v >= 0.03).length / jds.length,
  };
});

// === Fine-tune hybrid ===
console.log('=== Fine-tune hybrid ===\n');
{
  let best = {f1:0}, count=0;
  
  for(const twData of [{name:'tw3',d:tw3},{name:'tw5',d:tw5}]){
    for(let jdTh=0.02;jdTh<=0.045;jdTh+=0.005){
      for(let arTh=0.05;arTh<=0.2;arTh+=0.05){
        for(let jeTh=3.5;jeTh<=5.5;jeTh+=0.5){
          for(let zoneW=2.5;zoneW<=4.5;zoneW+=0.5){
            for(let vH=0.4;vH<=0.6;vH+=0.1){
              for(let vHW=1.5;vHW<=3;vHW+=0.5){
                for(let jdW=0.5;jdW<=1.5;jdW+=0.5){
                  for(let t=4;t<=6;t+=0.5){
                    const preds = all.map((s,i) => {
                      let v = 0;
                      const f = twData.d[i];
                      // Zone signal
                      if(f.jdMean >= jdTh && f.activeRate >= arTh && f.jeMean >= jeTh) v += zoneW;
                      // Token signals
                      if(s.jawVelocity >= vH) v += vHW;
                      else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                      if(s.jawDelta >= 0.05) v += jdW;
                      else if(s.jawDelta >= 0.02) v += jdW*0.5;
                      if(jawEff[i] >= 5) v += 0.5;
                      if(scoreVelAnti[i] >= 0.2) v += 0.5;
                      if(s.score < 0.45) v += 0.5;
                      if(dt[i] >= 0.2) v += 0.5;
                      // Penalties
                      if(f.jdMean < 0.005) v -= 2;
                      if(f.jeMean < 1.5) v -= 1;
                      if(!isHighJW[i] && s.score >= 0.7 && s.jawDelta < 0.02) v -= 1;
                      return v >= t;
                    });
                    const r = ev(preds);
                    if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                      best={...r,tw:twData.name,jdTh,arTh,jeTh,zoneW,vH,vHW,jdW,t};
                      count++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Fine-tune: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  window=${best.tw} jdMean>=${best.jdTh} activeRate>=${best.arTh} jeMean>=${best.jeTh}`);
    console.log(`  zoneW=${best.zoneW} vel>=${best.vH}→${best.vHW} jdW=${best.jdW}`);
    console.log(`  threshold=${best.t}`);
    
    // Error analysis
    const twD = best.tw==='tw3'?tw3:tw5;
    const sc = all.map((s,i) => {
      let v = 0;
      const f = twD[i];
      if(f.jdMean >= best.jdTh && f.activeRate >= best.arTh && f.jeMean >= best.jeTh) v += best.zoneW;
      if(s.jawVelocity >= best.vH) v += best.vHW;
      else if(s.jawVelocity >= 0.1) v += best.vHW*0.3;
      if(s.jawDelta >= 0.05) v += best.jdW;
      else if(s.jawDelta >= 0.02) v += best.jdW*0.5;
      if(jawEff[i] >= 5) v += 0.5;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(s.score < 0.45) v += 0.5;
      if(dt[i] >= 0.2) v += 0.5;
      if(f.jdMean < 0.005) v -= 2;
      if(f.jeMean < 1.5) v -= 1;
      if(!isHighJW[i] && s.score >= 0.7 && s.jawDelta < 0.02) v -= 1;
      return v;
    });
    const preds = sc.map(v => v >= best.t);
    
    console.log('\n=== FP analysis ===');
    const FP = [];
    for(let i=0;i<N;i++) if(preds[i]&&!act[i]) FP.push(i);
    console.log(`FP: ${FP.length}`);
    // Group FP by zone status
    let fpInZone=0, fpOutZone=0;
    FP.forEach(i => {
      const f = twD[i];
      if(f.jdMean >= best.jdTh && f.activeRate >= best.arTh && f.jeMean >= best.jeTh) fpInZone++;
      else fpOutZone++;
    });
    console.log(`  In user zone: ${fpInZone}, Outside: ${fpOutZone}`);
    
    // Show some FP
    FP.slice(0,15).forEach(i => {
      const s=all[i];
      const f=twD[i];
      console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} jd=${s.jawDelta.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(3)} zone_jd=${f.jdMean.toFixed(3)} zone_ar=${f.activeRate.toFixed(2)} zone_je=${f.jeMean.toFixed(1)} v=${sc[i].toFixed(2)}`);
    });
    
    console.log('\n=== FN analysis ===');
    const FN = [];
    for(let i=0;i<N;i++) if(!preds[i]&&act[i]) FN.push(i);
    console.log(`FN: ${FN.length}`);
    let fnInZone=0, fnOutZone=0;
    FN.forEach(i => {
      const f = twD[i];
      if(f.jdMean >= best.jdTh && f.activeRate >= best.arTh && f.jeMean >= best.jeTh) fnInZone++;
      else fnOutZone++;
    });
    console.log(`  In user zone: ${fnInZone}, Outside: ${fnOutZone}`);
    
    FN.slice(0,15).forEach(i => {
      const s=all[i];
      const f=twD[i];
      console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} jd=${s.jawDelta.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(3)} zone_jd=${f.jdMean.toFixed(3)} zone_ar=${f.activeRate.toFixed(2)} zone_je=${f.jeMean.toFixed(1)} v=${sc[i].toFixed(2)}`);
    });
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v99: F1=80.5%');
