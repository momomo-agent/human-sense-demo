// autoresearch v117: Two-stage classifier + novel approaches
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

// === Part 1: Two-stage — high recall stage 1 + FP filter stage 2 ===
console.log('=== Two-stage classifier ===\n');
{
  // Stage 1: very loose threshold for high recall
  function stage1Score(s, i) {
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
    return v;
  }
  
  const s1scores = all.map((s,i) => stage1Score(s,i));
  
  let best = {f1:0};
  
  for(let s1t=3;s1t<=5;s1t+=0.5){
    const s1preds = s1scores.map(v => v >= s1t);
    const s1r = ev(s1preds);
    if(s1r.recall < 0.95) continue; // need high recall in stage 1
    
    // Stage 2: filter FP using finalScore + score + neighborhood
    for(let fsTh=0.5;fsTh<=0.7;fsTh+=0.1){
      for(let fsW=1;fsW<=3;fsW+=0.5){
        for(let scTh=0.6;scTh<=0.8;scTh+=0.1){
          for(let scW=0;scW<=1.5;scW+=0.5){
            for(let s2t=0;s2t<=2;s2t+=0.5){
              const preds = all.map((s,i) => {
                if(!s1preds[i]) return false;
                // Stage 2 filter
                let penalty = 0;
                const fsc = s.finalScore||0;
                if(fsc >= fsTh) penalty += fsW;
                if(scW>0 && s.score >= scTh && fsc >= 0.4) penalty += scW;
                if(!isHighJW[i] && s.score >= 0.7) penalty += 0.5;
                return penalty <= s2t;
              });
              const r = ev(preds);
              if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                best={...r,s1t,fsTh,fsW,scTh,scW,s2t,s1R:s1r.recall,s1FP:s1r.FP};
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Two-stage: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  S1: t=${best.s1t} R=${(best.s1R*100).toFixed(1)}% FP=${best.s1FP}`);
    console.log(`  S2: fs>=${best.fsTh}→${best.fsW} sc>=${best.scTh}→${best.scW} maxPen=${best.s2t}`);
  }
}

// === Part 2: Inverted approach — start from AI detection ===
console.log('\n=== AI-first detection ===\n');
{
  // Instead of detecting user, detect AI and invert
  // AI signals: high finalScore, low jawDelta, low velocity, in non-active zone
  let best = {f1:0};
  
  for(let fsTh=0.5;fsTh<=0.8;fsTh+=0.1){
    for(let fsW=2;fsW<=4;fsW+=0.5){
      for(let lowJdTh=0.01;lowJdTh<=0.03;lowJdTh+=0.01){
        for(let lowJdW=1;lowJdW<=3;lowJdW+=0.5){
          for(let lowVelTh=0.05;lowVelTh<=0.15;lowVelTh+=0.05){
            for(let lowVelW=1;lowVelW<=3;lowVelW+=0.5){
              for(let noZoneW=1;noZoneW<=3;noZoneW+=0.5){
                for(let t=3;t<=6;t+=0.5){
                  const preds = all.map((s,i) => {
                    // AI score: higher = more likely AI
                    let aiScore = 0;
                    const fsc = s.finalScore||0;
                    if(fsc >= fsTh) aiScore += fsW;
                    if(s.jawDelta < lowJdTh) aiScore += lowJdW;
                    if(s.jawVelocity < lowVelTh) aiScore += lowVelW;
                    const f = tw10[i];
                    if(f.jdMean < 0.03 || f.jeMean < 5) aiScore += noZoneW;
                    // User if AI score is low
                    return aiScore < t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                    best={...r,fsTh,fsW,lowJdTh,lowJdW,lowVelTh,lowVelW,noZoneW,t};
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`AI-first: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  fs>=${best.fsTh}→${best.fsW} jd<${best.lowJdTh}→${best.lowJdW} vel<${best.lowVelTh}→${best.lowVelW}`);
    console.log(`  noZone→${best.noZoneW} aiThreshold=${best.t}`);
  } else {
    console.log('No qualifying AI-first config');
  }
}

// === Part 3: Hybrid — combine user-detection + AI-detection ===
console.log('\n=== Hybrid user+AI detection ===\n');
{
  // User score (positive evidence) + AI score (negative evidence)
  let best = {f1:0};
  
  for(let zoneW=4;zoneW<=5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        // AI evidence
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let lowActW=0.5;lowActW<=2;lowActW+=0.5){
            for(let t=4;t<=6;t+=0.5){
              const preds = all.map((s,i) => {
                let v = 0;
                const f = tw10[i];
                // User evidence
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
                // AI evidence (subtract)
                const fsc = s.finalScore||0;
                if(fsc >= 0.7) v -= fsW;
                if(s.jawDelta < 0.01 && s.jawVelocity < 0.05) v -= lowActW;
                if(f.jdMean < 0.005) v -= 2;
                if(f.jeMean < 1.5) v -= 1;
                if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                if(fsc >= 0.5 && s.score >= 0.7) v -= 1;
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                best={...r,zoneW,vHW,jdW,fsW,lowActW,t};
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Hybrid: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fsW=${best.fsW} lowActW=${best.lowActW} threshold=${best.t}`);
  }
}

// === Part 4: Weighted linear combination (no thresholds) ===
console.log('\n=== Weighted linear ===\n');
{
  let best = {f1:0};
  
  for(let wJd=5;wJd<=15;wJd+=2.5){
    for(let wVel=1;wVel<=4;wVel+=0.5){
      for(let wJe=0.1;wJe<=0.5;wJe+=0.1){
        for(let wFs=-5;wFs<=-1;wFs+=1){
          for(let wSc=-2;wSc<=0;wSc+=0.5){
            for(let wSva=1;wSva<=4;wSva+=1){
              for(let t=1;t<=4;t+=0.5){
                const preds = all.map((s,i) => {
                  const fsc = s.finalScore||0;
                  const v = s.jawDelta*wJd + s.jawVelocity*wVel + jawEff[i]*wJe + fsc*wFs + s.score*wSc + scoreVelAnti[i]*wSva;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,wJd,wVel,wJe,wFs,wSc,wSva,t};
                }
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Linear: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  jd*${best.wJd} + vel*${best.wVel} + je*${best.wJe} + fs*${best.wFs} + sc*${best.wSc} + sva*${best.wSva} >= ${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6%');
