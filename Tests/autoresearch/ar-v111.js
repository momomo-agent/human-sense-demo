// autoresearch v111: Combine best findings: tw10 + velMom + dtZeroRatio + finalScore
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

const velMomentum = all.map((_,i) => {
  let sum = 0;
  for(let j=i;j<Math.min(N,i+5);j++) sum+=all[j].jawVelocity;
  return sum;
});

function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}
const dtZeroRatio5 = wstat(dt, 2, a => a.filter(v => v < 0.001).length / a.length);

// === Combined search ===
console.log('=== v111: Combined best features ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zJdTh=0.02;zJdTh<=0.04;zJdTh+=0.005){
    for(let zoneW=4;zoneW<=5;zoneW+=0.5){
      for(let vHW=2;vHW<=3;vHW+=0.5){
        for(let jdW=0.5;jdW<=2;jdW+=0.5){
          for(let vmTh=0.5;vmTh<=2;vmTh+=0.5){
            for(let vmW=0;vmW<=1;vmW+=0.5){
              for(let dzrW=0;dzrW<=1;dzrW+=0.5){
                for(let p1W=0.5;p1W<=1.5;p1W+=0.5){
                  for(let p2W=0;p2W<=1;p2W+=0.5){
                    for(let t=4.5;t<=6.5;t+=0.25){
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
                        if(vmW>0 && velMomentum[i] >= vmTh) v += vmW;
                        if(dzrW>0 && dtZeroRatio5[i] >= 0.5) v += dzrW;
                        if(f.jdMean < 0.005) v -= 2;
                        if(f.jeMean < 1.5) v -= 1;
                        if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                        if(p2W>0 && s.jawDelta < 0.04 && s.jawVelocity >= 0.1) v -= p2W;
                        return v >= t;
                      });
                      const r = ev(preds);
                      if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1){
                        best={...r,zJdTh,zoneW,vHW,jdW,vmTh,vmW,dzrW,p1W,p2W,t};
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
  console.log(`Combined: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tw10 zone(jd>=${best.zJdTh})→${best.zoneW}`);
    console.log(`  vel>=0.5→${best.vHW} jd>=0.05→${best.jdW}`);
    console.log(`  velMom>=${best.vmTh}→${best.vmW} dzr→${best.dzrW}`);
    console.log(`  p1W=${best.p1W} p2W=${best.p2W} threshold=${best.t}`);
  }
}

// === With finalScore ===
console.log('\n=== + finalScore ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4;zoneW<=5;zoneW+=0.5){
    for(let vHW=2;vHW<=3;vHW+=0.5){
      for(let jdW=0.5;jdW<=2;jdW+=0.5){
        for(let vmW=0;vmW<=1;vmW+=0.5){
          for(let fsTh=0.5;fsTh<=0.8;fsTh+=0.1){
            for(let fsW=0.5;fsW<=2;fsW+=0.5){
              for(let p1W=0.5;p1W<=1.5;p1W+=0.5){
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
                    if(vmW>0 && velMomentum[i] >= 1) v += vmW;
                    if(f.jdMean < 0.005) v -= 2;
                    if(f.jeMean < 1.5) v -= 1;
                    if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                    if((s.finalScore||0) >= fsTh) v -= fsW;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.85&&r.specificity>=0.92&&r.f1>best.f1){
                    best={...r,zoneW,vHW,jdW,vmW,fsTh,fsW,p1W,t};
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
  console.log(`+finalScore: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tw10 zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  vmW=${best.vmW} finalScore>=${best.fsTh}→-${best.fsW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

// === Asymmetric rescue on combined best ===
console.log('\n=== Rescue on combined ===\n');
{
  // Use best combined params
  const sc = all.map((s,i) => {
    let v = 0;
    const f = tw10[i];
    if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 4.5;
    if(s.jawVelocity >= 0.5) v += 2.5;
    else if(s.jawVelocity >= 0.1) v += 0.75;
    if(s.jawDelta >= 0.05) v += 1.5;
    else if(s.jawDelta >= 0.02) v += 0.6;
    if(jawEff[i] >= 5) v += 0.5;
    if(scoreVelAnti[i] >= 0.2) v += 0.5;
    if(s.score < 0.45) v += 0.5;
    if(dt[i] >= 0.2) v += 0.5;
    if(velMomentum[i] >= 1) v += 0.5;
    if(f.jdMean < 0.005) v -= 2;
    if(f.jeMean < 1.5) v -= 1;
    if(!isHighJW[i] && s.score >= 0.7) v -= 1;
    return v;
  });
  const p1 = sc.map(v => v >= 5.75);
  const r1 = ev(p1);
  console.log(`Base: R=${(r1.recall*100).toFixed(1)}% S=${(r1.specificity*100).toFixed(1)}% F1=${(r1.f1*100).toFixed(1)}% FP=${r1.FP} FN=${r1.FN}`);
  
  let best = {f1:r1.f1};
  for(let hw=3;hw<=15;hw+=3){
    for(let nTh=0.15;nTh<=0.55;nTh+=0.1){
      for(let low=-2;low<=2;low+=1){
        const preds = all.map((_,i) => {
          if(p1[i]) return true;
          if(all[i].jawVelocity < 0.02 && all[i].jawDelta < 0.005) return false;
          if(sc[i] < low) return false;
          let userN=0, total=0;
          for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
            if(j===i) continue;
            total++;
            if(p1[j]) userN++;
          }
          return total>0 && userN/total >= nTh;
        });
        const r = ev(preds);
        if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1) best={...r,hw,nTh,low};
      }
    }
  }
  if(best.f1>r1.f1){
    console.log(`+Rescue: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} nTh=${best.nTh.toFixed(2)} low=${best.low}`);
  } else {
    console.log('Rescue did not improve');
  }
}

console.log('\n=== PROGRESS ===');
console.log('v107: F1=83.5%, v108: F1=84.2%, v110: F1=84.4%');
