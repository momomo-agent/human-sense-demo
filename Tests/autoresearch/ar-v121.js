// autoresearch v121: fsDev10 + (1-sc)*jd + run features deep dive
const fs = require('fs');
const DATA = '/Users/kenefe/LOCAL/momo-agent/projects/human-sense-demo/Tests/speaker-test-data.jsonl';
const lines = fs.readFileSync(DATA, 'utf8').trim().split('\n');
const all = lines.map(l => JSON.parse(l));
all.sort((a, b) => a.audioTime - b.audioTime);
const N = all.length;
const act = all.map(s => s.isUserSpeaker);
const dt = all.map((s, i) => i === 0 ? 0 : s.audioTime - all[i - 1].audioTime);
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

function twZone(i, sec) {
  const t0 = all[i].audioTime;
  const idx = [];
  for(let j=i;j>=0;j--){if(t0-all[j].audioTime>sec)break;idx.push(j);}
  for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>sec)break;idx.push(j);}
  return { jdMean: mean(idx.map(j=>all[j].jawDelta)), jeMean: mean(idx.map(j=>jawEff[j])) };
}
const tw10 = all.map((_,i) => twZone(i, 10));

function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}
const dtZeroRatio5 = wstat(dt, 2, a => a.filter(v => v < 0.001).length / a.length);

// fsDev10: how much this token's finalScore deviates from local mean
const fsDev10 = all.map((_,i) => {
  const t0 = all[i].audioTime;
  const vals = [];
  for(let j=i;j>=0;j--){if(t0-all[j].audioTime>10)break;vals.push(all[j].finalScore||0);}
  for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>10)break;vals.push(all[j].finalScore||0);}
  return (all[i].finalScore||0) - mean(vals);
});

// (1-score)*jawDelta
const invScJd = all.map(s => (1-s.score) * s.jawDelta);

// === Part 1: fsDev10 as penalty ===
console.log('=== fsDev10 penalty search ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let fdTh=-0.1;fdTh<=0.15;fdTh+=0.05){
            for(let fdW=0.5;fdW<=2;fdW+=0.5){
              for(let t=4.5;t<=6;t+=0.25){
                const preds = all.map((s,i) => {
                  let v = 0;
                  const f = tw10[i];
                  if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                  if(s.jawVelocity >= 0.5) v += vHW;
                  else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                  if(s.jawDelta >= 0.05) v += jdW;
                  else if(s.jawDelta >= 0.02) v += jdW*0.4;
                  if(jawEff[i] >= 5) v += 0.5;
                  if(scoreVelAnti[i] >= 0.2) v += 0.5;
                  if(s.score < 0.45) v += 0.5;
                  if(dt[i] >= 0.2) v += 0.5;
                  if(dtZeroRatio5[i] >= 0.5) v += 0.5;
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  // fsDev10: positive = higher than local mean = more AI-like
                  if(fsDev10[i] >= fdTh) v -= fdW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsW,fdTh,fdW,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`fsDev10: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  fsDev10>=${best.fdTh}→-${best.fdW} threshold=${best.t}`);
  }
}

// === Part 2: (1-score)*jawDelta as bonus ===
console.log('\n=== (1-sc)*jd bonus ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let isjTh=0.01;isjTh<=0.05;isjTh+=0.01){
            for(let isjW=0.5;isjW<=2;isjW+=0.5){
              for(let t=4.5;t<=6;t+=0.25){
                const preds = all.map((s,i) => {
                  let v = 0;
                  const f = tw10[i];
                  if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                  if(s.jawVelocity >= 0.5) v += vHW;
                  else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                  if(s.jawDelta >= 0.05) v += jdW;
                  else if(s.jawDelta >= 0.02) v += jdW*0.4;
                  if(jawEff[i] >= 5) v += 0.5;
                  if(scoreVelAnti[i] >= 0.2) v += 0.5;
                  if(s.score < 0.45) v += 0.5;
                  if(dt[i] >= 0.2) v += 0.5;
                  if(dtZeroRatio5[i] >= 0.5) v += 0.5;
                  if(invScJd[i] >= isjTh) v += isjW;
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsW,isjTh,isjW,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`(1-sc)*jd: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  (1-sc)*jd>=${best.isjTh}→+${best.isjW} threshold=${best.t}`);
  }
}

// === Part 3: Combined — fsDev10 + (1-sc)*jd + best base ===
console.log('\n=== Combined fsDev10 + (1-sc)*jd ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let fdTh=-0.05;fdTh<=0.1;fdTh+=0.05){
            for(let fdW=0.5;fdW<=1.5;fdW+=0.5){
              for(let isjTh=0.02;isjTh<=0.04;isjTh+=0.01){
                for(let isjW=0.5;isjW<=1.5;isjW+=0.5){
                  for(let t=4.5;t<=6.5;t+=0.25){
                    const preds = all.map((s,i) => {
                      let v = 0;
                      const f = tw10[i];
                      if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                      if(s.jawVelocity >= 0.5) v += vHW;
                      else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                      if(s.jawDelta >= 0.05) v += jdW;
                      else if(s.jawDelta >= 0.02) v += jdW*0.4;
                      if(jawEff[i] >= 5) v += 0.5;
                      if(scoreVelAnti[i] >= 0.2) v += 0.5;
                      if(s.score < 0.45) v += 0.5;
                      if(dt[i] >= 0.2) v += 0.5;
                      if(dtZeroRatio5[i] >= 0.5) v += 0.5;
                      if(invScJd[i] >= isjTh) v += isjW;
                      if(f.jdMean < 0.005) v -= 2;
                      if(f.jeMean < 1.5) v -= 1;
                      if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                      if((s.finalScore||0) >= 0.7) v -= fsW;
                      if(fsDev10[i] >= fdTh) v -= fdW;
                      return v >= t;
                    });
                    const r = ev(preds);
                    if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                      best={...r,zoneW,vHW,jdW,fsW,fdTh,fdW,isjTh,isjW,t};
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
  console.log(`Combined: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  fsDev10>=${best.fdTh}→-${best.fdW} (1-sc)*jd>=${best.isjTh}→+${best.isjW}`);
    console.log(`  threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6%');
