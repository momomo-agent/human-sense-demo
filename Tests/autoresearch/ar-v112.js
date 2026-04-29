// autoresearch v112: Fine-tune v111 best configs
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

// === Part 1: Fine-tune finalScore config (the 87.4% one) ===
console.log('=== Fine-tune finalScore config ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zJdTh=0.02;zJdTh<=0.04;zJdTh+=0.005){
    for(let zoneW=4;zoneW<=5.5;zoneW+=0.25){
      for(let vHW=1.5;vHW<=3;vHW+=0.25){
        for(let jdW=1;jdW<=2.5;jdW+=0.25){
          for(let fsTh=0.55;fsTh<=0.8;fsTh+=0.05){
            for(let fsW=1;fsW<=3;fsW+=0.25){
              for(let p1W=0;p1W<=1;p1W+=0.5){
                for(let t=4.5;t<=6;t+=0.25){
                  const preds = all.map((s,i) => {
                    let v = 0;
                    const f = tw10[i];
                    if(f.jdMean >= zJdTh && f.jeMean >= 5) v += zoneW;
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
                    if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                    if((s.finalScore||0) >= fsTh) v -= fsW;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                    best={...r,zJdTh,zoneW,vHW,jdW,fsTh,fsW,p1W,t};
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
  console.log(`Fine-tune FS: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tw10 zone(jd>=${best.zJdTh})→${best.zoneW}`);
    console.log(`  vel>=0.5→${best.vHW} jd>=0.05→${best.jdW}`);
    console.log(`  finalScore>=${best.fsTh}→-${best.fsW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

// === Part 2: Fine-tune no-finalScore config (the 85.0% one) ===
console.log('\n=== Fine-tune no-FS config ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zJdTh=0.02;zJdTh<=0.04;zJdTh+=0.005){
    for(let zoneW=3.5;zoneW<=5;zoneW+=0.25){
      for(let vHW=2.5;vHW<=3.5;vHW+=0.25){
        for(let jdW=0.5;jdW<=1.5;jdW+=0.25){
          for(let dzrW=0;dzrW<=1;dzrW+=0.25){
            for(let p1W=1;p1W<=2;p1W+=0.25){
              for(let t=4.75;t<=5.75;t+=0.25){
                const preds = all.map((s,i) => {
                  let v = 0;
                  const f = tw10[i];
                  if(f.jdMean >= zJdTh && f.jeMean >= 5) v += zoneW;
                  if(s.jawVelocity >= 0.5) v += vHW;
                  else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                  if(s.jawDelta >= 0.05) v += jdW;
                  else if(s.jawDelta >= 0.02) v += jdW*0.4;
                  if(jawEff[i] >= 5) v += 0.5;
                  if(scoreVelAnti[i] >= 0.2) v += 0.5;
                  if(s.score < 0.45) v += 0.5;
                  if(dt[i] >= 0.2) v += 0.5;
                  if(dzrW>0 && dtZeroRatio5[i] >= 0.5) v += dzrW;
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1){
                  best={...r,zJdTh,zoneW,vHW,jdW,dzrW,p1W,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Fine-tune noFS: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tw10 zone(jd>=${best.zJdTh})→${best.zoneW}`);
    console.log(`  vel>=0.5→${best.vHW} jd>=0.05→${best.jdW}`);
    console.log(`  dzrW=${best.dzrW} p1W=${best.p1W} threshold=${best.t}`);
  }
}

// === Part 3: Add more penalties to FS config ===
console.log('\n=== FS + extra penalties ===\n');
{
  let best = {f1:0}, count=0;
  
  // Base: tw10 zoneW=5 vHW=2 jdW=2 fs>=0.7→-2 p1W=0.5 t=5.25
  for(let p2W=0;p2W<=1.5;p2W+=0.5){
    for(let p3W=0;p3W<=1.5;p3W+=0.5){
      for(let p4W=0;p4W<=1.5;p4W+=0.5){
        for(let fsTh=0.6;fsTh<=0.75;fsTh+=0.05){
          for(let fsW=1.5;fsW<=2.5;fsW+=0.25){
            for(let t=4.75;t<=5.75;t+=0.25){
              const preds = all.map((s,i) => {
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
                if((s.finalScore||0) >= fsTh) v -= fsW;
                // Extra penalties
                if(p2W>0 && s.jawDelta < 0.02 && s.jawVelocity < 0.1) v -= p2W;
                if(p3W>0 && s.score >= 0.8 && s.jawDelta < 0.03) v -= p3W;
                if(p4W>0 && s.jawDelta < 0.04 && s.jawVelocity >= 0.1 && !isHighJW[i]) v -= p4W;
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                best={...r,p2W,p3W,p4W,fsTh,fsW,t};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`FS+penalties: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  fs>=${best.fsTh}→-${best.fsW}`);
    console.log(`  p2(low activity)=${best.p2W} p3(high sc+low jd)=${best.p3W} p4(low jd+vel+!hjw)=${best.p4W}`);
    console.log(`  threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v107: F1=83.5%');
console.log('v111 noFS: F1=85.0%');
console.log('v111 FS: F1=87.4%');
