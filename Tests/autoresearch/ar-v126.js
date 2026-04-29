// autoresearch v126: finalScore distribution + ratio features + multi-threshold
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
  return { jdMean: mean(idx.map(j=>all[j].jawDelta)), jeMean: mean(idx.map(j=>jawEff[j])), idx };
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

// === New features: finalScore distribution in window ===
const fsStd10 = all.map((_,i) => {
  const vals = tw10[i].idx.map(j => all[j].finalScore || 0);
  return std(vals);
});

const fsMax10 = all.map((_,i) => {
  const vals = tw10[i].idx.map(j => all[j].finalScore || 0);
  return Math.max(...vals);
});

const fsMin10 = all.map((_,i) => {
  const vals = tw10[i].idx.map(j => all[j].finalScore || 0);
  return Math.min(...vals);
});

const fsRange10 = all.map((_,i) => fsMax10[i] - fsMin10[i]);

// Ratio: this token's fs vs window max
const fsRatio10 = all.map((_,i) => {
  const max = fsMax10[i];
  return max > 0.01 ? (all[i].finalScore || 0) / max : 0;
});

// How many tokens in window have high fs?
const fsHighRatio10 = all.map((_,i) => {
  const vals = tw10[i].idx.map(j => all[j].finalScore || 0);
  return vals.filter(v => v >= 0.7).length / vals.length;
});

// Analyze
console.log('=== finalScore distribution features ===\n');
const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

for(const [name, arr] of [
  ['fsStd10', fsStd10], ['fsRange10', fsRange10], ['fsRatio10', fsRatio10], ['fsHighRatio10', fsHighRatio10]
]){
  const uV = userIdx.map(i=>arr[i]), aV = aiIdx.map(i=>arr[i]);
  const uM=mean(uV), uS=std(uV), aM=mean(aV), aS=std(aV);
  const d = Math.sqrt((uS**2+aS**2)/2) > 0 ? Math.abs(uM-aM)/Math.sqrt((uS**2+aS**2)/2) : 0;
  console.log(`${name.padEnd(18)} User=${uM.toFixed(4)}±${uS.toFixed(4)} AI=${aM.toFixed(4)}±${aS.toFixed(4)} d=${d.toFixed(3)}`);
}

// === Part 2: fsHighRatio10 as penalty ===
console.log('\n=== fsHighRatio10 penalty ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let fhrTh=0.3;fhrTh<=0.7;fhrTh+=0.1){
            for(let fhrW=0.5;fhrW<=2;fhrW+=0.5){
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
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  if(fsHighRatio10[i] >= fhrTh) v -= fhrW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsW,fhrTh,fhrW,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`fsHighRatio10: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  fsHighRatio10>=${best.fhrTh}→-${best.fhrW} threshold=${best.t}`);
  }
}

// === Part 3: Multi-threshold finalScore (3 levels) ===
console.log('\n=== Multi-threshold finalScore ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW1=0.5;fsW1<=2;fsW1+=0.5){  // fs >= 0.5
          for(let fsW2=0.5;fsW2<=2;fsW2+=0.5){  // fs >= 0.7
            for(let fsW3=0;fsW3<=1.5;fsW3+=0.5){  // fs >= 0.9
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
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  // Multi-level fs penalty (cumulative)
                  const fsc = s.finalScore||0;
                  if(fsc >= 0.5) v -= fsW1;
                  if(fsc >= 0.7) v -= fsW2;
                  if(fsc >= 0.9) v -= fsW3;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsW1,fsW2,fsW3,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Multi-fs: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fs>=0.5→-${best.fsW1} fs>=0.7→-${best.fsW2} fs>=0.9→-${best.fsW3}`);
    console.log(`  threshold=${best.t}`);
  }
}

// === Part 4: velocity * (1-finalScore) as combined signal ===
console.log('\n=== vel*(1-fs) combined signal ===\n');
{
  const velInvFs = all.map(s => s.jawVelocity * (1 - (s.finalScore||0)));
  const uV = userIdx.map(i=>velInvFs[i]), aV = aiIdx.map(i=>velInvFs[i]);
  const d = Math.abs(mean(uV)-mean(aV))/Math.sqrt((std(uV)**2+std(aV)**2)/2);
  console.log(`vel*(1-fs): User=${mean(uV).toFixed(4)} AI=${mean(aV).toFixed(4)} d=${d.toFixed(3)}`);
  
  // Replace velocity with vel*(1-fs)
  let best = {f1:0};
  for(let zoneW=4;zoneW<=5.5;zoneW+=0.5){
    for(let vifW=1;vifW<=4;vifW+=0.5){
      for(let vifTh=0.1;vifTh<=0.5;vifTh+=0.1){
        for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
          for(let t=4;t<=6.5;t+=0.25){
            const preds = all.map((s,i) => {
              let v = 0;
              const f = tw10[i];
              if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
              // Replace vel with vel*(1-fs)
              if(velInvFs[i] >= 0.3) v += vifW;
              else if(velInvFs[i] >= vifTh) v += vifW*0.3;
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
              return v >= t;
            });
            const r = ev(preds);
            if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
              best={...r,zoneW,vifW,vifTh,jdW,t};
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`vel*(1-fs): R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vif>=${best.vifTh}→${best.vifW} jdW=${best.jdW} t=${best.t}`);
  }
}

// === Part 5: jawDelta * (1-finalScore) ===
console.log('\n=== jd*(1-fs) combined signal ===\n');
{
  const jdInvFs = all.map(s => s.jawDelta * (1 - (s.finalScore||0)));
  const uV = userIdx.map(i=>jdInvFs[i]), aV = aiIdx.map(i=>jdInvFs[i]);
  const d = Math.abs(mean(uV)-mean(aV))/Math.sqrt((std(uV)**2+std(aV)**2)/2);
  console.log(`jd*(1-fs): User=${mean(uV).toFixed(4)} AI=${mean(aV).toFixed(4)} d=${d.toFixed(3)}`);
  
  let best = {f1:0};
  for(let zoneW=4;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jifW=1;jifW<=4;jifW+=0.5){
        for(let jifTh=0.01;jifTh<=0.05;jifTh+=0.01){
          for(let t=4;t<=6.5;t+=0.25){
            const preds = all.map((s,i) => {
              let v = 0;
              const f = tw10[i];
              if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
              if(s.jawVelocity >= 0.5) v += vHW;
              else if(s.jawVelocity >= 0.1) v += vHW*0.3;
              // Replace jd with jd*(1-fs)
              if(jdInvFs[i] >= 0.03) v += jifW;
              else if(jdInvFs[i] >= jifTh) v += jifW*0.4;
              if(jawEff[i] >= 5) v += 0.5;
              if(scoreVelAnti[i] >= 0.2) v += 0.5;
              if(s.score < 0.45) v += 0.5;
              if(dt[i] >= 0.2) v += 0.5;
              if(dtZeroRatio5[i] >= 0.5) v += 0.5;
              if(f.jdMean < 0.005) v -= 2;
              if(f.jeMean < 1.5) v -= 1;
              if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
              return v >= t;
            });
            const r = ev(preds);
            if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
              best={...r,zoneW,vHW,jifW,jifTh,t};
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`jd*(1-fs): R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jif>=${best.jifTh}→${best.jifW} t=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
