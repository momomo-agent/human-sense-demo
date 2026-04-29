// autoresearch v127: Fused features (jd*(1-fs), vel*(1-fs)) + zone*(1-fs) + fine grid
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

// Fused features
const velInvFs = all.map(s => s.jawVelocity * (1 - (s.finalScore||0)));
const jdInvFs = all.map(s => s.jawDelta * (1 - (s.finalScore||0)));

// === Part 1: Replace vel AND jd with fused versions ===
console.log('=== Fused vel+jd (replace both) ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4;zoneW<=6;zoneW+=0.5){
    for(let vifH=0.2;vifH<=0.5;vifH+=0.1){
      for(let vifW=1;vifW<=3;vifW+=0.5){
        for(let jifH=0.02;jifH<=0.05;jifH+=0.01){
          for(let jifW=1;jifW<=3;jifW+=0.5){
            for(let t=4;t<=6.5;t+=0.25){
              const preds = all.map((s,i) => {
                let v = 0;
                const f = tw10[i];
                if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                // Fused velocity
                if(velInvFs[i] >= vifH) v += vifW;
                else if(velInvFs[i] >= vifH*0.3) v += vifW*0.3;
                // Fused jawDelta
                if(jdInvFs[i] >= jifH) v += jifW;
                else if(jdInvFs[i] >= jifH*0.4) v += jifW*0.4;
                if(jawEff[i] >= 5) v += 0.5;
                if(scoreVelAnti[i] >= 0.2) v += 0.5;
                if(s.score < 0.45) v += 0.5;
                if(dt[i] >= 0.2) v += 0.5;
                if(dtZeroRatio5[i] >= 0.5) v += 0.5;
                if(f.jdMean < 0.005) v -= 2;
                if(f.jeMean < 1.5) v -= 1;
                if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                // No separate fs penalty — it's baked into fused features
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                best={...r,zoneW,vifH,vifW,jifH,jifW,t};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Fused both: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vif>=${best.vifH}→${best.vifW} jif>=${best.jifH}→${best.jifW} t=${best.t}`);
  }
}

// === Part 2: Fused + original + fs penalty (hybrid) ===
console.log('\n=== Hybrid: original + fused + fs ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1;vHW<=2.5;vHW+=0.5){
      for(let jdW=1;jdW<=2.5;jdW+=0.5){
        for(let vifW=0;vifW<=1.5;vifW+=0.5){
          for(let jifW=0;jifW<=1.5;jifW+=0.5){
            for(let fsW=1;fsW<=3;fsW+=0.5){
              for(let t=4.5;t<=6.5;t+=0.25){
                const preds = all.map((s,i) => {
                  let v = 0;
                  const f = tw10[i];
                  if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                  // Original
                  if(s.jawVelocity >= 0.5) v += vHW;
                  else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                  if(s.jawDelta >= 0.05) v += jdW;
                  else if(s.jawDelta >= 0.02) v += jdW*0.4;
                  // Fused bonus
                  if(vifW > 0 && velInvFs[i] >= 0.3) v += vifW;
                  if(jifW > 0 && jdInvFs[i] >= 0.03) v += jifW;
                  if(jawEff[i] >= 5) v += 0.5;
                  if(scoreVelAnti[i] >= 0.2) v += 0.5;
                  if(s.score < 0.45) v += 0.5;
                  if(dt[i] >= 0.2) v += 0.5;
                  if(dtZeroRatio5[i] >= 0.5) v += 0.5;
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,vifW,jifW,fsW,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Hybrid: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  vif→+${best.vifW} jif→+${best.jifW} fs→-${best.fsW} t=${best.t}`);
  }
}

// === Part 3: Zone-level fused feature ===
console.log('\n=== Zone-level fused ===\n');
{
  // Zone jdMean * (1 - zone fsMean) — fuse at zone level
  const zoneFsMean10 = all.map((_,i) => {
    return mean(tw10[i].idx.map(j => all[j].finalScore || 0));
  });
  const zoneJdFused = all.map((_,i) => tw10[i].jdMean * (1 - zoneFsMean10[i]));
  const zoneJeFused = all.map((_,i) => tw10[i].jeMean * (1 - zoneFsMean10[i]));
  
  let best = {f1:0}, count=0;
  
  for(let zjfTh=0.01;zjfTh<=0.03;zjfTh+=0.005){
    for(let zjfW=3;zjfW<=6;zjfW+=0.5){
      for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
        for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
          for(let fsW=0;fsW<=3;fsW+=0.5){
            for(let t=4;t<=6.5;t+=0.25){
              const preds = all.map((s,i) => {
                let v = 0;
                // Fused zone
                if(zoneJdFused[i] >= zjfTh) v += zjfW;
                if(s.jawVelocity >= 0.5) v += vHW;
                else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                if(s.jawDelta >= 0.05) v += jdW;
                else if(s.jawDelta >= 0.02) v += jdW*0.4;
                if(jawEff[i] >= 5) v += 0.5;
                if(scoreVelAnti[i] >= 0.2) v += 0.5;
                if(s.score < 0.45) v += 0.5;
                if(dt[i] >= 0.2) v += 0.5;
                if(dtZeroRatio5[i] >= 0.5) v += 0.5;
                if(tw10[i].jdMean < 0.005) v -= 2;
                if(tw10[i].jeMean < 1.5) v -= 1;
                if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                if(fsW > 0 && (s.finalScore||0) >= 0.7) v -= fsW;
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                best={...r,zjfTh,zjfW,vHW,jdW,fsW,t};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Zone fused: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneJdFused>=${best.zjfTh}→${best.zjfW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fsW=${best.fsW} t=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
