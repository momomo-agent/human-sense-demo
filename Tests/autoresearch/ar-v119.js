// autoresearch v119: Change-point detection + gradient features + combined best
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

// === Change-point features ===
// Gradient: difference between forward and backward windows
const velGrad = all.map((_,i) => {
  const bw = [], fw = [];
  for(let j=Math.max(0,i-3);j<i;j++) bw.push(all[j].jawVelocity);
  for(let j=i+1;j<=Math.min(N-1,i+3);j++) fw.push(all[j].jawVelocity);
  return (fw.length?mean(fw):0) - (bw.length?mean(bw):0);
});

const jdGrad = all.map((_,i) => {
  const bw = [], fw = [];
  for(let j=Math.max(0,i-3);j<i;j++) bw.push(all[j].jawDelta);
  for(let j=i+1;j<=Math.min(N-1,i+3);j++) fw.push(all[j].jawDelta);
  return (fw.length?mean(fw):0) - (bw.length?mean(bw):0);
});

// Absolute gradient (magnitude of change)
const absVelGrad = velGrad.map(Math.abs);
const absJdGrad = jdGrad.map(Math.abs);

// Score gradient
const scoreGrad = all.map((_,i) => {
  const bw = [], fw = [];
  for(let j=Math.max(0,i-3);j<i;j++) bw.push(all[j].score);
  for(let j=i+1;j<=Math.min(N-1,i+3);j++) fw.push(all[j].score);
  return (fw.length?mean(fw):0) - (bw.length?mean(bw):0);
});

// Analyze
console.log('=== Gradient feature analysis ===\n');
const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

for(const [name, arr] of [['absVelGrad', absVelGrad], ['absJdGrad', absJdGrad], ['scoreGrad', scoreGrad.map(Math.abs)]]){
  const uV = userIdx.map(i=>arr[i]), aV = aiIdx.map(i=>arr[i]);
  const uM=mean(uV), uS=std(uV), aM=mean(aV), aS=std(aV);
  const d = Math.sqrt((uS**2+aS**2)/2) > 0 ? Math.abs(uM-aM)/Math.sqrt((uS**2+aS**2)/2) : 0;
  console.log(`${name.padEnd(16)} User=${uM.toFixed(4)}±${uS.toFixed(4)} AI=${aM.toFixed(4)}±${aS.toFixed(4)} d=${d.toFixed(3)}`);
}

// === Part 2: Combine everything — kitchen sink with fine-grained search ===
console.log('\n=== Kitchen sink: all best features ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let xW=0;xW<=1.5;xW+=0.5){
            for(let gradW=0;gradW<=1;gradW+=0.5){
              for(let p1W=0;p1W<=1;p1W+=0.5){
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
                    // Gradient bonus (high change = user onset)
                    if(gradW>0 && absVelGrad[i] >= 0.2) v += gradW;
                    // Penalties
                    if(f.jdMean < 0.005) v -= 2;
                    if(f.jeMean < 1.5) v -= 1;
                    if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                    const fsc = s.finalScore||0;
                    if(fsc >= 0.7) v -= fsW;
                    if(xW>0 && fsc >= 0.5 && s.score >= 0.7) v -= xW;
                    if(s.score >= 0.8 && s.jawDelta < 0.03) v -= 0.5;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                    best={...r,zoneW,vHW,jdW,fsW,xW,gradW,p1W,t};
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
  console.log(`Kitchen sink: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fs>=0.7→-${best.fsW} cross→-${best.xW} grad→+${best.gradW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

// === Part 3: Relaxed S constraint to find R/S tradeoff frontier ===
console.log('\n=== Pareto frontier (R vs S) ===\n');
{
  function score(s, i, fsW, xW) {
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
    const fsc = s.finalScore||0;
    if(fsc >= 0.7) v -= fsW;
    if(xW>0 && fsc >= 0.5 && s.score >= 0.7) v -= xW;
    return v;
  }
  
  const configs = [
    {fsW:2.5, xW:0, label:'fs-2.5'},
    {fsW:2, xW:0, label:'fs-2'},
    {fsW:1.5, xW:0, label:'fs-1.5'},
    {fsW:1.5, xW:1, label:'fs-1.5+x-1'},
    {fsW:2, xW:1, label:'fs-2+x-1'},
    {fsW:2.5, xW:1, label:'fs-2.5+x-1'},
    {fsW:3, xW:0, label:'fs-3'},
    {fsW:3, xW:1, label:'fs-3+x-1'},
  ];
  
  for(const cfg of configs){
    const sc = all.map((s,i) => score(s,i,cfg.fsW,cfg.xW));
    let bestT = {f1:0};
    for(let t=4;t<=6.5;t+=0.25){
      const preds = sc.map(v => v >= t);
      const r = ev(preds);
      if(r.f1>bestT.f1) bestT={...r,t};
    }
    console.log(`${cfg.label.padEnd(16)} t=${bestT.t} R=${(bestT.recall*100).toFixed(1)}% S=${(bestT.specificity*100).toFixed(1)}% F1=${(bestT.f1*100).toFixed(1)}% FP=${bestT.FP} FN=${bestT.FN}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
