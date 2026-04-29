// autoresearch v115: Cross-features + asymmetric threshold + ensemble
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

// === Part 1: Cross-feature penalties (finalScore × physical) ===
console.log('=== Cross-feature penalties ===\n');
{
  let best = {f1:0}, count=0;
  
  // Base: v112 best
  function base(s, i) {
    let v = 0;
    const f = tw10[i];
    if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 5;
    if(s.jawVelocity >= 0.5) v += 2;
    else if(s.jawVelocity >= 0.1) v += 0.6;
    if(s.jawDelta >= 0.05) v += 2;
    else if(s.jawDelta >= 0.02) v += 0.8;
    if(jawEff[i] >= 5) v += 0.5;
    if(scoreVelAnti[i] >= 0.2) v += 0.5;
    if(s.score < 0.45) v += 0.5;
    if(dt[i] >= 0.2) v += 0.5;
    if(dtZeroRatio5[i] >= 0.5) v += 0.5;
    if(f.jdMean < 0.005) v -= 2;
    if(f.jeMean < 1.5) v -= 1;
    if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
    return v;
  }
  
  // Cross penalties: fs×score, fs×vel, fs×jd
  for(let fsTh=0.5;fsTh<=0.7;fsTh+=0.1){
    for(let fsW=1.5;fsW<=3;fsW+=0.5){
      // Cross: high fs + high score = very likely AI
      for(let xScTh=0.5;xScTh<=0.7;xScTh+=0.1){
        for(let xScW=0;xScW<=1.5;xScW+=0.5){
          // Cross: high fs + low vel = AI lip sync
          for(let xVelTh=0.3;xVelTh<=0.5;xVelTh+=0.1){
            for(let xVelW=0;xVelW<=1;xVelW+=0.5){
              for(let t=4.75;t<=5.75;t+=0.25){
                const preds = all.map((s,i) => {
                  let v = base(s, i);
                  const fsc = s.finalScore||0;
                  if(fsc >= fsTh) v -= fsW;
                  if(xScW>0 && fsc >= 0.5 && s.score >= xScTh) v -= xScW;
                  if(xVelW>0 && fsc >= 0.5 && s.jawVelocity < xVelTh) v -= xVelW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.93&&r.f1>best.f1){
                  best={...r,fsTh,fsW,xScTh,xScW,xVelTh,xVelW,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Cross penalties: ${count} qualifying (S>=93%)`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  fs>=${best.fsTh}→-${best.fsW}`);
    console.log(`  fs>=0.5 && sc>=${best.xScTh}→-${best.xScW}`);
    console.log(`  fs>=0.5 && vel<${best.xVelTh}→-${best.xVelW}`);
    console.log(`  threshold=${best.t}`);
  }
  
  // Also check S>=92%
  let best92 = {f1:0};
  for(let fsTh=0.5;fsTh<=0.7;fsTh+=0.1){
    for(let fsW=1.5;fsW<=3;fsW+=0.5){
      for(let xScTh=0.5;xScTh<=0.7;xScTh+=0.1){
        for(let xScW=0;xScW<=1.5;xScW+=0.5){
          for(let xVelTh=0.3;xVelTh<=0.5;xVelTh+=0.1){
            for(let xVelW=0;xVelW<=1;xVelW+=0.5){
              for(let t=4.75;t<=5.75;t+=0.25){
                const preds = all.map((s,i) => {
                  let v = base(s, i);
                  const fsc = s.finalScore||0;
                  if(fsc >= fsTh) v -= fsW;
                  if(xScW>0 && fsc >= 0.5 && s.score >= xScTh) v -= xScW;
                  if(xVelW>0 && fsc >= 0.5 && s.jawVelocity < xVelTh) v -= xVelW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best92.f1){
                  best92={...r,fsTh,fsW,xScTh,xScW,xVelTh,xVelW,t};
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`\nBest S>=92%: R=${(best92.recall*100).toFixed(1)}% S=${(best92.specificity*100).toFixed(1)}% F1=${(best92.f1*100).toFixed(1)}% FP=${best92.FP} FN=${best92.FN}`);
  console.log(`  fs>=${best92.fsTh}→-${best92.fsW} xSc>=${best92.xScTh}→-${best92.xScW} xVel<${best92.xVelTh}→-${best92.xVelW} t=${best92.t}`);
}

// === Part 2: Ensemble — vote between two classifiers ===
console.log('\n=== Ensemble voting ===\n');
{
  // Classifier A: physical features (no finalScore)
  function classA(s, i, t) {
    let v = 0;
    const f = tw10[i];
    if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 4;
    if(s.jawVelocity >= 0.5) v += 3;
    else if(s.jawVelocity >= 0.1) v += 0.9;
    if(s.jawDelta >= 0.05) v += 1;
    else if(s.jawDelta >= 0.02) v += 0.4;
    if(jawEff[i] >= 5) v += 0.5;
    if(scoreVelAnti[i] >= 0.2) v += 0.5;
    if(s.score < 0.45) v += 0.5;
    if(dt[i] >= 0.2) v += 0.5;
    if(dtZeroRatio5[i] >= 0.5) v += 0.5;
    if(f.jdMean < 0.005) v -= 2;
    if(f.jeMean < 1.5) v -= 1;
    if(!isHighJW[i] && s.score >= 0.7) v -= 1.5;
    return v >= t;
  }
  
  // Classifier B: finalScore-heavy
  function classB(s, i, t) {
    let v = 0;
    const f = tw10[i];
    if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 5;
    if(s.jawVelocity >= 0.5) v += 2;
    else if(s.jawVelocity >= 0.1) v += 0.6;
    if(s.jawDelta >= 0.05) v += 2;
    else if(s.jawDelta >= 0.02) v += 0.8;
    if(jawEff[i] >= 5) v += 0.5;
    if(scoreVelAnti[i] >= 0.2) v += 0.5;
    if(s.score < 0.45) v += 0.5;
    if(dt[i] >= 0.2) v += 0.5;
    if(f.jdMean < 0.005) v -= 2;
    if(f.jeMean < 1.5) v -= 1;
    if((s.finalScore||0) >= 0.7) v -= 2.5;
    return v >= t;
  }
  
  let best = {f1:0};
  for(let tA=4;tA<=6;tA+=0.5){
    for(let tB=4;tB<=6;tB+=0.5){
      // AND: both must agree
      const predsAnd = all.map((s,i) => classA(s,i,tA) && classB(s,i,tB));
      const rAnd = ev(predsAnd);
      if(rAnd.recall>=0.85&&rAnd.specificity>=0.93&&rAnd.f1>best.f1) best={...rAnd,mode:'AND',tA,tB};
      
      // OR: either agrees
      const predsOr = all.map((s,i) => classA(s,i,tA) || classB(s,i,tB));
      const rOr = ev(predsOr);
      if(rOr.recall>=0.85&&rOr.specificity>=0.93&&rOr.f1>best.f1) best={...rOr,mode:'OR',tA,tB};
    }
  }
  if(best.f1>0){
    console.log(`Ensemble: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  mode=${best.mode} tA=${best.tA} tB=${best.tB}`);
  } else {
    console.log('No qualifying ensemble');
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
