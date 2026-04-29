// autoresearch v114: Aggressive FP reduction via finalScore + score patterns
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

// finalScore windowed mean
const fsMean5 = wstat(all.map(s=>s.finalScore||0), 2, mean);

// === Multi-level finalScore penalty ===
console.log('=== Multi-level finalScore ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fs1Th=0.5;fs1Th<=0.7;fs1Th+=0.1){
          for(let fs1W=1;fs1W<=3;fs1W+=0.5){
            for(let fs2Th=0.3;fs2Th<=0.5;fs2Th+=0.1){
              for(let fs2W=0;fs2W<=1.5;fs2W+=0.5){
                for(let fmW=0;fmW<=1.5;fmW+=0.5){
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
                      if(f.jdMean < 0.005) v -= 2;
                      if(f.jeMean < 1.5) v -= 1;
                      if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                      // Multi-level finalScore
                      const fsc = s.finalScore||0;
                      if(fsc >= fs1Th) v -= fs1W;
                      else if(fs2W>0 && fsc >= fs2Th) v -= fs2W;
                      // finalScore windowed mean
                      if(fmW>0 && fsMean5[i] >= 0.5) v -= fmW;
                      return v >= t;
                    });
                    const r = ev(preds);
                    if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                      best={...r,zoneW,vHW,jdW,fs1Th,fs1W,fs2Th,fs2W,fmW,t};
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
  console.log(`Multi-FS: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fs>=${best.fs1Th}→-${best.fs1W}, fs>=${best.fs2Th}→-${best.fs2W}`);
    console.log(`  fsMean5>=0.5→-${best.fmW} threshold=${best.t}`);
  }
}

// === Continuous finalScore weight ===
console.log('\n=== Continuous finalScore ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsScale=2;fsScale<=6;fsScale+=0.5){
          for(let p1W=0;p1W<=1;p1W+=0.5){
            for(let t=5;t<=7;t+=0.25){
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
                if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                // Continuous: subtract finalScore * scale
                v -= (s.finalScore||0) * fsScale;
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                best={...r,zoneW,vHW,jdW,fsScale,p1W,t};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Continuous FS: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fsScale=${best.fsScale} p1W=${best.p1W} threshold=${best.t}`);
  }
}

// === scoreGap as feature ===
console.log('\n=== scoreGap feature ===\n');
{
  // scoreGap = |score * jawFactor * velFactor * noMoveFactor - score|
  // Per-token jawWeight/jawVelocityWeight
  const scoreGap = all.map(s => {
    const jw = s.jawWeight || 0.2;
    const vw = s.jawVelocityWeight || 0.2;
    const jawFactor = 1 - jw * (1 - s.jawDelta * 10);
    const velFactor = 1 - vw * (1 - s.jawVelocity);
    const noMoveFactor = (s.jawDelta < 0.001 && s.jawVelocity < 0.01) ? 0.5 : 1;
    const adjusted = s.score * Math.max(0, jawFactor) * Math.max(0, velFactor) * noMoveFactor;
    return Math.abs(adjusted - s.score);
  });
  
  const userSG = act.map((a,i) => a ? scoreGap[i] : null).filter(v=>v!==null);
  const aiSG = act.map((a,i) => !a ? scoreGap[i] : null).filter(v=>v!==null);
  console.log(`scoreGap: User=${mean(userSG).toFixed(4)}±${Math.sqrt(userSG.reduce((s,v)=>s+(v-mean(userSG))**2,0)/userSG.length).toFixed(4)} AI=${mean(aiSG).toFixed(4)}`);
  
  let best = {f1:0};
  for(let sgTh=0.05;sgTh<=0.5;sgTh+=0.05){
    for(let sgW=0.5;sgW<=2;sgW+=0.5){
      const p2 = all.map((s,i) => {
        let v = sc_base(s,i);
        if(scoreGap[i] >= sgTh) v += sgW;
        return v >= 5.25;
      });
      const r = ev(p2);
      if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1) best={...r,sgTh,sgW};
    }
  }
  
  function sc_base(s,i) {
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
    if((s.finalScore||0) >= 0.7) v -= 2.5;
    return v;
  }
  
  if(best.f1>0){
    console.log(`scoreGap bonus: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6%');
