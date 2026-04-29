// autoresearch v122: Decision tree + exhaustive rule search
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

// === Part 1: Simple decision tree ===
console.log('=== Decision tree search ===\n');
{
  // Tree: if zone → check physical → check finalScore
  // Non-zone → default AI
  let best = {f1:0};
  
  for(let velTh=0.1;velTh<=0.5;velTh+=0.1){
    for(let jdTh=0.01;jdTh<=0.05;jdTh+=0.01){
      for(let fsTh=0.5;fsTh<=0.8;fsTh+=0.1){
        // In zone: user if (vel >= velTh OR jd >= jdTh) AND fs < fsTh
        // In zone: also user if vel >= 0.5 (strong physical signal overrides fs)
        for(let strongVel=0.3;strongVel<=0.7;strongVel+=0.1){
          const preds = all.map((s,i) => {
            const f = tw10[i];
            if(f.jdMean < 0.03 || f.jeMean < 5) return false; // not in zone
            const fsc = s.finalScore||0;
            // Strong physical signal: always user
            if(s.jawVelocity >= strongVel && s.jawDelta >= jdTh) return true;
            // Moderate signal: user if fs is low
            if((s.jawVelocity >= velTh || s.jawDelta >= jdTh) && fsc < fsTh) return true;
            return false;
          });
          const r = ev(preds);
          if(r.recall>=0.85&&r.specificity>=0.92&&r.f1>best.f1){
            best={...r,velTh,jdTh,fsTh,strongVel};
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Tree: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zone → vel>=${best.strongVel}&&jd>=${best.jdTh} → user (strong)`);
    console.log(`  zone → (vel>=${best.velTh}||jd>=${best.jdTh}) && fs<${best.fsTh} → user`);
  }
}

// === Part 2: Multi-branch tree ===
console.log('\n=== Multi-branch tree ===\n');
{
  let best = {f1:0};
  
  for(let strongVel=0.3;strongVel<=0.6;strongVel+=0.1){
    for(let strongJd=0.03;strongJd<=0.06;strongJd+=0.01){
      for(let fsTh1=0.5;fsTh1<=0.7;fsTh1+=0.1){
        for(let fsTh2=0.6;fsTh2<=0.8;fsTh2+=0.1){
          for(let lowScTh=0.5;lowScTh<=0.7;lowScTh+=0.1){
            const preds = all.map((s,i) => {
              const f = tw10[i];
              if(f.jdMean < 0.03 || f.jeMean < 5) return false;
              const fsc = s.finalScore||0;
              // Branch 1: strong physical → user regardless
              if(s.jawVelocity >= strongVel && s.jawDelta >= strongJd) return true;
              // Branch 2: moderate physical + low fs → user
              if(s.jawVelocity >= 0.1 && s.jawDelta >= 0.02 && fsc < fsTh1) return true;
              // Branch 3: low score (voice mismatch) + any movement → user
              if(s.score < lowScTh && (s.jawVelocity >= 0.1 || s.jawDelta >= 0.02) && fsc < fsTh2) return true;
              // Branch 4: very high jawEff → user
              if(jawEff[i] >= 8 && fsc < fsTh2) return true;
              return false;
            });
            const r = ev(preds);
            if(r.recall>=0.85&&r.specificity>=0.92&&r.f1>best.f1){
              best={...r,strongVel,strongJd,fsTh1,fsTh2,lowScTh};
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Multi-tree: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  B1: vel>=${best.strongVel}&&jd>=${best.strongJd}`);
    console.log(`  B2: vel>=0.1&&jd>=0.02&&fs<${best.fsTh1}`);
    console.log(`  B3: sc<${best.lowScTh}&&(vel>=0.1||jd>=0.02)&&fs<${best.fsTh2}`);
    console.log(`  B4: je>=8&&fs<${best.fsTh2}`);
  }
}

// === Part 3: Soft scoring with continuous finalScore weight ===
console.log('\n=== Continuous fs weight (v2) ===\n');
{
  let best = {f1:0};
  
  // v = physical_score - fs_penalty
  // physical_score = zone + vel + jd + je + sva + ...
  // fs_penalty = finalScore * scale (continuous, not threshold)
  for(let zoneW=4;zoneW<=6;zoneW+=0.5){
    for(let vHW=1;vHW<=3;vHW+=0.5){
      for(let jdW=1;jdW<=3;jdW+=0.5){
        for(let fsScale=1;fsScale<=5;fsScale+=0.5){
          for(let t=3;t<=6;t+=0.5){
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
              if(f.jdMean < 0.005) v -= 2;
              if(f.jeMean < 1.5) v -= 1;
              if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
              // Continuous fs penalty
              v -= (s.finalScore||0) * fsScale;
              return v >= t;
            });
            const r = ev(preds);
            if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
              best={...r,zoneW,vHW,jdW,fsScale,t};
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Continuous fs: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsScale=${best.fsScale} t=${best.t}`);
  }
}

// === Part 4: Ensemble of v112 + tree ===
console.log('\n=== Ensemble: voting + tree ===\n');
{
  // v112 best scores
  const v112sc = all.map((s,i) => {
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
    if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
    if((s.finalScore||0) >= 0.7) v -= 2.5;
    return v;
  });
  
  let best = {f1:0};
  for(let vt=4;vt<=6;vt+=0.5){
    for(let strongVel=0.3;strongVel<=0.5;strongVel+=0.1){
      for(let fsTh=0.5;fsTh<=0.7;fsTh+=0.1){
        // Ensemble: user if voting says yes AND (strong physical OR low fs)
        const preds = all.map((s,i) => {
          const votingUser = v112sc[i] >= vt;
          if(!votingUser) return false;
          const fsc = s.finalScore||0;
          if(s.jawVelocity >= strongVel && s.jawDelta >= 0.03) return true;
          if(fsc < fsTh) return true;
          return false;
        });
        const r = ev(preds);
        if(r.recall>=0.85&&r.specificity>=0.93&&r.f1>best.f1){
          best={...r,vt,strongVel,fsTh};
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Ensemble: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  voting>=${best.vt} && (vel>=${best.strongVel}&&jd>=0.03 || fs<${best.fsTh})`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6%');
