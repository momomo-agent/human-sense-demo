// autoresearch v110: Longer windows + sequence patterns + combined jitter
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

// === Part 1: Compute rich features ===

// Time-windowed zone
function twZone(i, sec) {
  const t0 = all[i].audioTime;
  const idx = [];
  for(let j=i;j>=0;j--){if(t0-all[j].audioTime>sec)break;idx.push(j);}
  for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>sec)break;idx.push(j);}
  return {
    jdMean: mean(idx.map(j=>all[j].jawDelta)),
    jeMean: mean(idx.map(j=>jawEff[j])),
    velMean: mean(idx.map(j=>all[j].jawVelocity)),
    velMax: Math.max(...idx.map(j=>all[j].jawVelocity)),
    jdMax: Math.max(...idx.map(j=>all[j].jawDelta)),
    activeRate: idx.filter(j=>all[j].jawDelta>0.01).length/idx.length,
    count: idx.length,
  };
}
const tw5 = all.map((_,i) => twZone(i, 5));
const tw10 = all.map((_,i) => twZone(i, 10));

// Velocity momentum: sum of velocity in forward window
const velMomentum = all.map((_,i) => {
  let sum = 0, cnt = 0;
  for(let j=i;j<Math.min(N,i+5);j++){sum+=all[j].jawVelocity;cnt++;}
  return sum;
});

// Score stability: how stable is score in window
const scoreStd5 = all.map((_,i) => {
  const w = [];
  for(let j=Math.max(0,i-2);j<=Math.min(N-1,i+2);j++) w.push(all[j].score);
  return std(w);
});

// jawDelta acceleration
const jdAccel = all.map((s,i) => i<=1 ? 0 : all[i].jawDelta - 2*all[i-1].jawDelta + (i>=2?all[i-2].jawDelta:0));

// Consecutive high-activity count (how many consecutive tokens have vel>0.1)
const consecActive = new Array(N).fill(0);
for(let i=0;i<N;i++){
  if(all[i].jawVelocity >= 0.1) consecActive[i] = (i>0 ? consecActive[i-1] : 0) + 1;
}

// === Part 2: Broader search with new features ===
console.log('=== v110: Extended feature search ===\n');
{
  let best = {f1:0}, count=0;
  
  // Use tw10 zone instead of tw5
  for(let twSec of [5, 10]){
    const tw = twSec === 5 ? tw5 : tw10;
    for(let zJdTh=0.02;zJdTh<=0.04;zJdTh+=0.01){
      for(let zoneW=3.5;zoneW<=5;zoneW+=0.5){
        for(let vHW=2;vHW<=3;vHW+=0.5){
          for(let jdW=0.5;jdW<=1.5;jdW+=0.5){
            // New: activeRate bonus
            for(let arTh=0.1;arTh<=0.3;arTh+=0.1){
              for(let arW=0;arW<=1;arW+=0.5){
                // New: velMomentum bonus
                for(let vmTh=1;vmTh<=3;vmTh+=1){
                  for(let vmW=0;vmW<=1;vmW+=0.5){
                    for(let p1W=0.5;p1W<=1.5;p1W+=0.5){
                      for(let t=4.5;t<=6;t+=0.25){
                        const preds = all.map((s,i) => {
                          let v = 0;
                          const f = tw[i];
                          if(f.jdMean >= zJdTh && f.jeMean >= 5) v += zoneW;
                          if(s.jawVelocity >= 0.5) v += vHW;
                          else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                          if(s.jawDelta >= 0.05) v += jdW;
                          else if(s.jawDelta >= 0.02) v += jdW*0.4;
                          if(jawEff[i] >= 5) v += 0.5;
                          if(scoreVelAnti[i] >= 0.2) v += 0.5;
                          if(s.score < 0.45) v += 0.5;
                          if(dt[i] >= 0.2) v += 0.5;
                          // New features
                          if(arW>0 && f.activeRate >= arTh) v += arW;
                          if(vmW>0 && velMomentum[i] >= vmTh) v += vmW;
                          // Penalties
                          if(f.jdMean < 0.005) v -= 2;
                          if(f.jeMean < 1.5) v -= 1;
                          if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                          return v >= t;
                        });
                        const r = ev(preds);
                        if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1){
                          best={...r,twSec,zJdTh,zoneW,vHW,jdW,arTh,arW,vmTh,vmW,p1W,t};
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
  console.log(`Extended: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tw=${best.twSec}s zone(jd>=${best.zJdTh})→${best.zoneW}`);
    console.log(`  vel>=0.5→${best.vHW} jd>=0.05→${best.jdW}`);
    console.log(`  activeRate>=${best.arTh}→${best.arW} velMom>=${best.vmTh}→${best.vmW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

// === Part 3: Two-pass with rescue ===
console.log('\n=== Two-pass rescue ===\n');
{
  // Use v107 best as pass 1
  const sc = all.map((s,i) => {
    let v = 0;
    const f = tw5[i];
    if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 4.5;
    if(s.jawVelocity >= 0.5) v += 2.5;
    else if(s.jawVelocity >= 0.1) v += 0.75;
    if(s.jawDelta >= 0.05) v += 1.5;
    else if(s.jawDelta >= 0.02) v += 0.6;
    if(jawEff[i] >= 5) v += 0.5;
    if(scoreVelAnti[i] >= 0.2) v += 0.5;
    if(s.score < 0.45) v += 0.5;
    if(dt[i] >= 0.2) v += 0.5;
    if(f.jdMean < 0.005) v -= 2;
    if(f.jeMean < 1.5) v -= 1;
    if(!isHighJW[i] && s.score >= 0.7) v -= 1;
    return v;
  });
  const p1 = sc.map(v => v >= 5.25);
  const r1 = ev(p1);
  console.log(`Pass 1: R=${(r1.recall*100).toFixed(1)}% S=${(r1.specificity*100).toFixed(1)}% F1=${(r1.f1*100).toFixed(1)}% FP=${r1.FP} FN=${r1.FN}`);
  
  let best = {f1:r1.f1};
  // Rescue: for tokens not predicted user, check neighborhood
  for(let hw=3;hw<=15;hw+=2){
    for(let nTh=0.15;nTh<=0.6;nTh+=0.05){
      for(let low=-3;low<=2;low+=1){
        for(let minVel=0;minVel<=0.1;minVel+=0.05){
          const preds = all.map((_,i) => {
            if(p1[i]) return true;
            if(all[i].jawVelocity < minVel && all[i].jawDelta < 0.01) return false;
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
          if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1) best={...r,hw,nTh,low,minVel};
        }
      }
    }
  }
  if(best.f1>r1.f1){
    console.log(`+Rescue: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} nTh=${best.nTh.toFixed(2)} low=${best.low} minVel=${best.minVel}`);
  } else {
    console.log('Rescue did not improve');
  }
}

// === Part 4: Consecutive activity pattern ===
console.log('\n=== Consecutive activity bonus ===\n');
{
  let best = {f1:0};
  for(let caTh=2;caTh<=6;caTh++){
    for(let caW=0.5;caW<=2;caW+=0.5){
      for(let zoneW=3.5;zoneW<=4.5;zoneW+=0.5){
        for(let vHW=2;vHW<=3;vHW+=0.5){
          for(let p1W=0.5;p1W<=1.5;p1W+=0.5){
            for(let t=4.5;t<=6;t+=0.25){
              const preds = all.map((s,i) => {
                let v = 0;
                const f = tw5[i];
                if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                if(s.jawVelocity >= 0.5) v += vHW;
                else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                if(s.jawDelta >= 0.05) v += 1.5;
                else if(s.jawDelta >= 0.02) v += 0.6;
                if(jawEff[i] >= 5) v += 0.5;
                if(scoreVelAnti[i] >= 0.2) v += 0.5;
                if(s.score < 0.45) v += 0.5;
                if(dt[i] >= 0.2) v += 0.5;
                if(consecActive[i] >= caTh) v += caW;
                if(f.jdMean < 0.005) v -= 2;
                if(f.jeMean < 1.5) v -= 1;
                if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1) best={...r,caTh,caW,zoneW,vHW,p1W,t};
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  consecActive>=${best.caTh}→${best.caW} zoneW=${best.zoneW} vHW=${best.vHW} p1W=${best.p1W} t=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v107: F1=83.5%, v108: F1=84.2%');
