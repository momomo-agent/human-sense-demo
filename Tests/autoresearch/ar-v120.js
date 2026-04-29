// autoresearch v120: Causal window + run-level + token position features
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

// Symmetric tw10
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

// === Part 1: Run-level features ===
// Detect runs of similar tokens (consecutive tokens with similar physical characteristics)
// A "run" = consecutive tokens where jawDelta > threshold
const runs = [];
let curRun = null;
for(let i=0;i<N;i++){
  const active = all[i].jawDelta >= 0.02 || all[i].jawVelocity >= 0.1;
  if(active){
    if(!curRun) curRun = {start:i, tokens:[]};
    curRun.tokens.push(i);
  } else {
    if(curRun) { runs.push(curRun); curRun = null; }
  }
}
if(curRun) runs.push(curRun);

// For each token, compute run-level features
const runLen = new Array(N).fill(0);
const runPos = new Array(N).fill(0); // position within run (0=start, 1=end)
const runVelMean = new Array(N).fill(0);
const runJdMean = new Array(N).fill(0);
const runFsMean = new Array(N).fill(0);

for(const run of runs){
  const len = run.tokens.length;
  const vels = run.tokens.map(i=>all[i].jawVelocity);
  const jds = run.tokens.map(i=>all[i].jawDelta);
  const fss = run.tokens.map(i=>all[i].finalScore||0);
  const vm = mean(vels), jm = mean(jds), fm = mean(fss);
  for(let k=0;k<len;k++){
    const i = run.tokens[k];
    runLen[i] = len;
    runPos[i] = len > 1 ? k/(len-1) : 0.5;
    runVelMean[i] = vm;
    runJdMean[i] = jm;
    runFsMean[i] = fm;
  }
}

console.log('=== Run-level feature analysis ===\n');
const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

for(const [name, arr] of [['runLen', runLen], ['runPos', runPos], ['runVelMean', runVelMean], ['runJdMean', runJdMean], ['runFsMean', runFsMean]]){
  const uV = userIdx.map(i=>arr[i]), aV = aiIdx.map(i=>arr[i]);
  const uM=mean(uV), uS=std(uV), aM=mean(aV), aS=std(aV);
  const d = Math.sqrt((uS**2+aS**2)/2) > 0 ? Math.abs(uM-aM)/Math.sqrt((uS**2+aS**2)/2) : 0;
  console.log(`${name.padEnd(14)} User=${uM.toFixed(3)}±${uS.toFixed(3)} AI=${aM.toFixed(3)}±${aS.toFixed(3)} d=${d.toFixed(3)}`);
}

// === Part 2: Run features + v112 base ===
console.log('\n=== v112 + run features ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let rlTh=3;rlTh<=7;rlTh+=2){
            for(let rlW=0;rlW<=1;rlW+=0.5){
              for(let rfmTh=0.4;rfmTh<=0.6;rfmTh+=0.1){
                for(let rfmW=0;rfmW<=1;rfmW+=0.5){
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
                      // Run features
                      if(rlW>0 && runLen[i] >= rlTh) v += rlW;
                      if(rfmW>0 && runFsMean[i] >= rfmTh) v -= rfmW;
                      // Penalties
                      if(f.jdMean < 0.005) v -= 2;
                      if(f.jeMean < 1.5) v -= 1;
                      if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                      if((s.finalScore||0) >= 0.7) v -= fsW;
                      return v >= t;
                    });
                    const r = ev(preds);
                    if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                      best={...r,zoneW,vHW,jdW,fsW,rlTh,rlW,rfmTh,rfmW,t};
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
  console.log(`v112+run: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  runLen>=${best.rlTh}→${best.rlW} runFsMean>=${best.rfmTh}→-${best.rfmW}`);
    console.log(`  threshold=${best.t}`);
  }
}

// === Part 3: Causal-only window (past only, more realistic for real-time) ===
console.log('\n=== Causal window (past-only) ===\n');
{
  function causalZone(i, sec) {
    const t0 = all[i].audioTime;
    const idx = [i];
    for(let j=i-1;j>=0;j--){if(t0-all[j].audioTime>sec)break;idx.push(j);}
    return { jdMean: mean(idx.map(j=>all[j].jawDelta)), jeMean: mean(idx.map(j=>jawEff[j])) };
  }
  const cw10 = all.map((_,i) => causalZone(i, 10));
  
  let best = {f1:0};
  for(let zoneW=4;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let t=4.5;t<=6;t+=0.25){
            const preds = all.map((s,i) => {
              let v = 0;
              const f = cw10[i];
              if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
              if(s.jawVelocity >= 0.5) v += vHW;
              else if(s.jawVelocity >= 0.1) v += vHW*0.3;
              if(s.jawDelta >= 0.05) v += jdW;
              else if(s.jawDelta >= 0.02) v += jdW*0.4;
              if(jawEff[i] >= 5) v += 0.5;
              if(scoreVelAnti[i] >= 0.2) v += 0.5;
              if(s.score < 0.45) v += 0.5;
              if(dt[i] >= 0.2) v += 0.5;
              if(f.jdMean < 0.005) v -= 2;
              if(f.jeMean < 1.5) v -= 1;
              if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
              if((s.finalScore||0) >= 0.7) v -= fsW;
              return v >= t;
            });
            const r = ev(preds);
            if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
              best={...r,zoneW,vHW,jdW,fsW,t};
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Causal: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW} t=${best.t}`);
  }
}

// === Part 4: Score difference from local mean ===
console.log('\n=== Score deviation from local mean ===\n');
{
  const scoreDev10 = all.map((_,i) => {
    const t0 = all[i].audioTime;
    const vals = [];
    for(let j=i;j>=0;j--){if(t0-all[j].audioTime>10)break;vals.push(all[j].score);}
    for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>10)break;vals.push(all[j].score);}
    const m = mean(vals);
    return all[i].score - m;
  });
  
  const uV = userIdx.map(i=>scoreDev10[i]), aV = aiIdx.map(i=>scoreDev10[i]);
  console.log(`scoreDev10: User=${mean(uV).toFixed(4)} AI=${mean(aV).toFixed(4)} d=${(Math.abs(mean(uV)-mean(aV))/Math.sqrt((std(uV)**2+std(aV)**2)/2)).toFixed(3)}`);
  
  // finalScore deviation
  const fsDev10 = all.map((_,i) => {
    const t0 = all[i].audioTime;
    const vals = [];
    for(let j=i;j>=0;j--){if(t0-all[j].audioTime>10)break;vals.push(all[j].finalScore||0);}
    for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>10)break;vals.push(all[j].finalScore||0);}
    const m = mean(vals);
    return (all[i].finalScore||0) - m;
  });
  
  const uFD = userIdx.map(i=>fsDev10[i]), aFD = aiIdx.map(i=>fsDev10[i]);
  console.log(`fsDev10: User=${mean(uFD).toFixed(4)} AI=${mean(aFD).toFixed(4)} d=${(Math.abs(mean(uFD)-mean(aFD))/Math.sqrt((std(uFD)**2+std(aFD)**2)/2)).toFixed(3)}`);
  
  // velocity deviation
  const velDev10 = all.map((_,i) => {
    const t0 = all[i].audioTime;
    const vals = [];
    for(let j=i;j>=0;j--){if(t0-all[j].audioTime>10)break;vals.push(all[j].jawVelocity);}
    for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>10)break;vals.push(all[j].jawVelocity);}
    const m = mean(vals);
    return all[i].jawVelocity - m;
  });
  
  const uVD = userIdx.map(i=>velDev10[i]), aVD = aiIdx.map(i=>velDev10[i]);
  console.log(`velDev10: User=${mean(uVD).toFixed(4)} AI=${mean(aVD).toFixed(4)} d=${(Math.abs(mean(uVD)-mean(aVD))/Math.sqrt((std(uVD)**2+std(aVD)**2)/2)).toFixed(3)}`);
}

// === Part 5: Nonlinear combination — product features ===
console.log('\n=== Product features ===\n');
{
  // jd * vel (both high = strong user signal)
  const jdVel = all.map(s => s.jawDelta * s.jawVelocity);
  // (1-fs) * vel (low finalScore + high vel = user)
  const invFsVel = all.map(s => (1-(s.finalScore||0)) * s.jawVelocity);
  // (1-score) * jd (low score + high jd = user)
  const invScJd = all.map(s => (1-s.score) * s.jawDelta);
  
  for(const [name, arr] of [['jd*vel', jdVel], ['(1-fs)*vel', invFsVel], ['(1-sc)*jd', invScJd]]){
    const uV = userIdx.map(i=>arr[i]), aV = aiIdx.map(i=>arr[i]);
    const uM=mean(uV), uS=std(uV), aM=mean(aV), aS=std(aV);
    const d = Math.sqrt((uS**2+aS**2)/2) > 0 ? Math.abs(uM-aM)/Math.sqrt((uS**2+aS**2)/2) : 0;
    console.log(`${name.padEnd(14)} User=${uM.toFixed(4)}±${uS.toFixed(4)} AI=${aM.toFixed(4)}±${aS.toFixed(4)} d=${d.toFixed(3)}`);
  }
  
  // Try (1-fs)*vel as bonus
  let best = {f1:0};
  for(let ifvTh=0.1;ifvTh<=0.5;ifvTh+=0.1){
    for(let ifvW=0.5;ifvW<=2;ifvW+=0.5){
      for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
        for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
          for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
            for(let fsW=1.5;fsW<=3;fsW+=0.5){
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
                  if(invFsVel[i] >= ifvTh) v += ifvW;
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,ifvTh,ifvW,zoneW,vHW,jdW,fsW,t};
                }
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`\n(1-fs)*vel: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  ifv>=${best.ifvTh}→+${best.ifvW} zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW} t=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6%');
