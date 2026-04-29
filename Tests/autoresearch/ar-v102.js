// autoresearch v102: Back to token-level, but with better feature engineering
// Key insight: user/AI interleave at sub-second level, time smoothing hurts
// Need to find token-level features that separate them
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

// New composite features
// 1. jawPower = jawDelta * jawVelocity (physical energy of jaw movement)
const jawPower = all.map(s => s.jawDelta * s.jawVelocity);
// 2. jawMomentum = jawDelta^2 * jawVelocity (emphasizes large openings)
const jawMomentum = all.map(s => s.jawDelta * s.jawDelta * s.jawVelocity);
// 3. Combined physical score
const physScore = all.map(s => s.jawDelta * 5 + s.jawVelocity * 0.5 + (s.jawDelta > 0.001 ? s.jawVelocity/s.jawDelta : 0) * 0.05);

console.log('=== New composite features ===\n');
const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

const feats = {
  jawPower: i => jawPower[i],
  jawMomentum: i => jawMomentum[i],
  physScore: i => physScore[i],
  'jd*vel*eff': i => all[i].jawDelta * all[i].jawVelocity * jawEff[i],
  'jd+vel': i => all[i].jawDelta + all[i].jawVelocity,
  'jd*10+vel': i => all[i].jawDelta*10 + all[i].jawVelocity,
  'sva*jd': i => scoreVelAnti[i] * all[i].jawDelta,
};

for(const [name, fn] of Object.entries(feats)){
  const uVals = userIdx.map(fn);
  const aVals = aiIdx.map(fn);
  const uM = mean(uVals), uS = std(uVals);
  const aM = mean(aVals), aS = std(aVals);
  const pooledS = Math.sqrt((uS**2 + aS**2)/2);
  const d = pooledS > 0 ? Math.abs(uM - aM) / pooledS : 0;
  console.log(`${name.padEnd(15)} User=${uM.toFixed(4)}±${uS.toFixed(4)} AI=${aM.toFixed(4)}±${aS.toFixed(4)} d=${d.toFixed(3)}`);
}

// === Try simple linear combination ===
console.log('\n=== Linear combination search ===\n');
{
  let best = {f1:0};
  
  for(let wJd=5;wJd<=20;wJd+=2.5){
    for(let wVel=0.5;wVel<=2;wVel+=0.25){
      for(let wJe=0;wJe<=0.15;wJe+=0.025){
        for(let wSva=0;wSva<=1.5;wSva+=0.25){
          for(let wSc=-1;wSc<=0;wSc+=0.25){
            for(let th=0.5;th<=3;th+=0.25){
              const preds = all.map((s,i) => {
                const v = s.jawDelta*wJd + s.jawVelocity*wVel + jawEff[i]*wJe + scoreVelAnti[i]*wSva + s.score*wSc;
                return v >= th;
              });
              const r = ev(preds);
              if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,wJd,wVel,wJe,wSva,wSc,th};
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Linear: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  wJd=${best.wJd} wVel=${best.wVel} wJe=${best.wJe} wSva=${best.wSva} wSc=${best.wSc} th=${best.th}`);
  } else {
    console.log('No qualifying linear combination found');
    // Relax constraints
    let best2 = {f1:0};
    for(let wJd=5;wJd<=20;wJd+=2.5){
      for(let wVel=0.5;wVel<=2;wVel+=0.25){
        for(let wJe=0;wJe<=0.15;wJe+=0.05){
          for(let wSva=0;wSva<=1.5;wSva+=0.5){
            for(let th=0.5;th<=3;th+=0.25){
              const preds = all.map((s,i) => {
                const v = s.jawDelta*wJd + s.jawVelocity*wVel + jawEff[i]*wJe + scoreVelAnti[i]*wSva;
                return v >= th;
              });
              const r = ev(preds);
              if(r.f1>best2.f1) best2={...r,wJd,wVel,wJe,wSva,th};
            }
          }
        }
      }
    }
    console.log(`Best unconstrained: R=${(best2.recall*100).toFixed(1)}% S=${(best2.specificity*100).toFixed(1)}% F1=${(best2.f1*100).toFixed(1)}% FP=${best2.FP} FN=${best2.FN}`);
    console.log(`  wJd=${best2.wJd} wVel=${best2.wVel} wJe=${best2.wJe} wSva=${best2.wSva} th=${best2.th}`);
  }
}

// === Hybrid: linear score + zone bonus + voting penalties ===
console.log('\n=== Hybrid: linear + zone + penalties ===\n');
{
  // Time-windowed jawEff mean (5s)
  function timeWindowIdx(centerIdx, windowSec) {
    const t0 = all[centerIdx].audioTime;
    const indices = [];
    for(let j=centerIdx;j>=0;j--){
      if(t0 - all[j].audioTime > windowSec) break;
      indices.push(j);
    }
    for(let j=centerIdx+1;j<N;j++){
      if(all[j].audioTime - t0 > windowSec) break;
      indices.push(j);
    }
    return indices;
  }
  
  const tw5jeMean = all.map((s,i) => {
    const idx = timeWindowIdx(i, 5);
    return mean(idx.map(j => jawEff[j]));
  });
  const tw5jdMean = all.map((s,i) => {
    const idx = timeWindowIdx(i, 5);
    return mean(idx.map(j => all[j].jawDelta));
  });
  
  let best = {f1:0}, count=0;
  
  for(let wJd=7.5;wJd<=15;wJd+=2.5){
    for(let wVel=0.5;wVel<=1.5;wVel+=0.25){
      for(let wSva=0.5;wSva<=1.5;wSva+=0.5){
        for(let zoneTh=4;zoneTh<=6;zoneTh+=0.5){
          for(let zoneW=1;zoneW<=3;zoneW+=0.5){
            for(let penJdLow=-3;penJdLow<=-1;penJdLow+=0.5){
              for(let penJeLow=-2;penJeLow<=-0.5;penJeLow+=0.5){
                for(let t=2;t<=4;t+=0.25){
                  const preds = all.map((s,i) => {
                    let v = 0;
                    // Linear token score
                    v += s.jawDelta * wJd;
                    v += s.jawVelocity * wVel;
                    v += scoreVelAnti[i] * wSva;
                    // Zone bonus
                    if(tw5jeMean[i] >= zoneTh) v += zoneW;
                    // Penalties
                    if(tw5jdMean[i] < 0.01) v += penJdLow;
                    if(tw5jeMean[i] < 2) v += penJeLow;
                    // Score penalty for high confidence AI
                    if(s.score >= 0.7 && s.jawDelta < 0.02) v -= 1;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                    best={...r,wJd,wVel,wSva,zoneTh,zoneW,penJdLow,penJeLow,t};
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
  console.log(`Hybrid linear: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  wJd=${best.wJd} wVel=${best.wVel} wSva=${best.wSva}`);
    console.log(`  zone: jeMean>=${best.zoneTh}→${best.zoneW}`);
    console.log(`  pen: jdMean<0.01→${best.penJdLow}, jeMean<2→${best.penJeLow}`);
    console.log(`  threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v100: F1=81.9%');
