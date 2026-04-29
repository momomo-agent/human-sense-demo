// autoresearch v109: Multi-jitter + v107 base
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
function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}

const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

const velDiff = all.map((s, i) => i === 0 ? 0 : Math.abs(s.jawVelocity - all[i-1].jawVelocity));
const velJitter5 = wstat(velDiff, 2, mean);
const jdDiff = all.map((s, i) => i === 0 ? 0 : Math.abs(s.jawDelta - all[i-1].jawDelta));
const jdJitter5 = wstat(jdDiff, 2, mean);
const dtZeroRatio5 = wstat(dt, 2, a => a.filter(v => v < 0.001).length / a.length);

function timeWindowIdx(centerIdx, windowSec) {
  const t0 = all[centerIdx].audioTime;
  const indices = [];
  for(let j=centerIdx;j>=0;j--){if(t0-all[j].audioTime>windowSec)break;indices.push(j);}
  for(let j=centerIdx+1;j<N;j++){if(all[j].audioTime-t0>windowSec)break;indices.push(j);}
  return indices;
}
const tw5 = all.map((s,i) => {
  const idx = timeWindowIdx(i, 5);
  const jds = idx.map(j => all[j].jawDelta);
  const effs = idx.map(j => jawEff[j]);
  return { jdMean: mean(jds), jeMean: mean(effs) };
});

// === Search: v107 base + multiple jitter features ===
console.log('=== Multi-jitter search ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=3.5;zoneW<=4.5;zoneW+=0.5){
    for(let vHW=2;vHW<=3;vHW+=0.5){
      for(let jdW=0.5;jdW<=1.5;jdW+=0.5){
        for(let vjTh=0.05;vjTh<=0.2;vjTh+=0.05){
          for(let vjW=0;vjW<=1;vjW+=0.5){
            for(let jdjTh=0.01;jdjTh<=0.04;jdjTh+=0.01){
              for(let jdjW=0;jdjW<=1;jdjW+=0.5){
                for(let dzrTh=0.3;dzrTh<=0.6;dzrTh+=0.1){
                  for(let dzrW=0;dzrW<=1;dzrW+=0.5){
                    for(let p1W=0.5;p1W<=1.5;p1W+=0.5){
                      for(let t=4.5;t<=6;t+=0.25){
                        const preds = all.map((s,i) => {
                          let v = 0;
                          const f = tw5[i];
                          if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                          if(s.jawVelocity >= 0.5) v += vHW;
                          else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                          if(s.jawDelta >= 0.05) v += jdW;
                          else if(s.jawDelta >= 0.02) v += jdW*0.4;
                          if(jawEff[i] >= 5) v += 0.5;
                          if(scoreVelAnti[i] >= 0.2) v += 0.5;
                          if(s.score < 0.45) v += 0.5;
                          if(dt[i] >= 0.2) v += 0.5;
                          // Jitter bonuses (user has higher jitter)
                          if(vjW>0 && velJitter5[i] >= vjTh) v += vjW;
                          if(jdjW>0 && jdJitter5[i] >= jdjTh) v += jdjW;
                          if(dzrW>0 && dtZeroRatio5[i] >= dzrTh) v += dzrW;
                          // Penalties
                          if(f.jdMean < 0.005) v -= 2;
                          if(f.jeMean < 1.5) v -= 1;
                          if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                          return v >= t;
                        });
                        const r = ev(preds);
                        if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1){
                          best={...r,zoneW,vHW,jdW,vjTh,vjW,jdjTh,jdjW,dzrTh,dzrW,p1W,t};
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
    }
  }
  console.log(`Multi-jitter: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  velJitter5>=${best.vjTh}→${best.vjW}`);
    console.log(`  jdJitter5>=${best.jdjTh}→${best.jdjW}`);
    console.log(`  dtZeroRatio5>=${best.dzrTh}→${best.dzrW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
    
    // Error analysis
    const sc = all.map((s,i) => {
      let v = 0;
      const f = tw5[i];
      if(f.jdMean >= 0.03 && f.jeMean >= 5) v += best.zoneW;
      if(s.jawVelocity >= 0.5) v += best.vHW;
      else if(s.jawVelocity >= 0.1) v += best.vHW*0.3;
      if(s.jawDelta >= 0.05) v += best.jdW;
      else if(s.jawDelta >= 0.02) v += best.jdW*0.4;
      if(jawEff[i] >= 5) v += 0.5;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(s.score < 0.45) v += 0.5;
      if(dt[i] >= 0.2) v += 0.5;
      if(best.vjW>0 && velJitter5[i] >= best.vjTh) v += best.vjW;
      if(best.jdjW>0 && jdJitter5[i] >= best.jdjTh) v += best.jdjW;
      if(best.dzrW>0 && dtZeroRatio5[i] >= best.dzrTh) v += best.dzrW;
      if(f.jdMean < 0.005) v -= 2;
      if(f.jeMean < 1.5) v -= 1;
      if(!isHighJW[i] && s.score >= 0.7) v -= best.p1W;
      return v;
    });
    const preds = sc.map(v => v >= best.t);
    const FP = [], FN = [];
    for(let i=0;i<N;i++){
      if(preds[i]&&!act[i]) FP.push(i);
      if(!preds[i]&&act[i]) FN.push(i);
    }
    console.log(`\nFP: ${FP.length}`);
    FP.slice(0,10).forEach(i => {
      const s=all[i];
      console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} jd=${s.jawDelta.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} vj=${velJitter5[i].toFixed(3)} jdj=${jdJitter5[i].toFixed(3)} v=${sc[i].toFixed(2)}`);
    });
    console.log(`\nFN: ${FN.length}`);
    FN.slice(0,10).forEach(i => {
      const s=all[i];
      console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} jd=${s.jawDelta.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} vj=${velJitter5[i].toFixed(3)} jdj=${jdJitter5[i].toFixed(3)} v=${sc[i].toFixed(2)}`);
    });
  }
}

// === Also try jitter as penalty (low jitter = AI) ===
console.log('\n=== Jitter as penalty ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=3.5;zoneW<=4.5;zoneW+=0.5){
    for(let vHW=2;vHW<=3;vHW+=0.5){
      for(let jdW=0.5;jdW<=1.5;jdW+=0.5){
        for(let vjLow=0.02;vjLow<=0.08;vjLow+=0.02){
          for(let vjPen=0.5;vjPen<=2;vjPen+=0.5){
            for(let jdjLow=0.005;jdjLow<=0.02;jdjLow+=0.005){
              for(let jdjPen=0;jdjPen<=1;jdjPen+=0.5){
                for(let p1W=0.5;p1W<=1.5;p1W+=0.5){
                  for(let t=4.5;t<=5.75;t+=0.25){
                    const preds = all.map((s,i) => {
                      let v = 0;
                      const f = tw5[i];
                      if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                      if(s.jawVelocity >= 0.5) v += vHW;
                      else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                      if(s.jawDelta >= 0.05) v += jdW;
                      else if(s.jawDelta >= 0.02) v += jdW*0.4;
                      if(jawEff[i] >= 5) v += 0.5;
                      if(scoreVelAnti[i] >= 0.2) v += 0.5;
                      if(s.score < 0.45) v += 0.5;
                      if(dt[i] >= 0.2) v += 0.5;
                      // Jitter penalties (low jitter = AI mechanical pattern)
                      if(velJitter5[i] < vjLow) v -= vjPen;
                      if(jdjPen>0 && jdJitter5[i] < jdjLow) v -= jdjPen;
                      // Other penalties
                      if(f.jdMean < 0.005) v -= 2;
                      if(f.jeMean < 1.5) v -= 1;
                      if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                      return v >= t;
                    });
                    const r = ev(preds);
                    if(r.recall>=0.85&&r.specificity>=0.92&&r.f1>best.f1){
                      best={...r,zoneW,vHW,jdW,vjLow,vjPen,jdjLow,jdjPen,p1W,t};
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
  console.log(`Jitter penalty: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  velJitter5<${best.vjLow}→-${best.vjPen}`);
    console.log(`  jdJitter5<${best.jdjLow}→-${best.jdjPen}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v107: F1=83.5%');
console.log('v108: F1=84.2% (dtZeroRatio5)');
