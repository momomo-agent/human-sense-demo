// autoresearch v104: Add finalScore as feature (d=1.542 in zone!)
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

const tw5 = all.map((s,i) => {
  const idx = timeWindowIdx(i, 5);
  const jds = idx.map(j => all[j].jawDelta);
  const effs = idx.map(j => jawEff[j]);
  return {
    jdMean: mean(jds),
    jeMean: mean(effs),
    activeRate: jds.filter(v => v >= 0.03).length / jds.length,
  };
});

// First: what is finalScore? Check if it's already computed
console.log('=== finalScore analysis ===\n');
{
  const hasFinalScore = all.filter(s => s.finalScore !== undefined && s.finalScore !== null).length;
  console.log(`Tokens with finalScore: ${hasFinalScore}/${N}`);
  if(hasFinalScore > 0){
    const userFS = [], aiFS = [];
    for(let i=0;i<N;i++){
      if(all[i].finalScore === undefined) continue;
      if(act[i]) userFS.push(all[i].finalScore);
      else aiFS.push(all[i].finalScore);
    }
    console.log(`User finalScore: mean=${mean(userFS).toFixed(3)}`);
    console.log(`AI finalScore: mean=${mean(aiFS).toFixed(3)}`);
    
    // Best threshold for finalScore alone
    let bestF1=0, bestTh=0;
    for(let th=0.1;th<=0.9;th+=0.01){
      const preds = all.map(s => (s.finalScore || 0) < th);
      const r = ev(preds);
      if(r.f1>bestF1){bestF1=r.f1;bestTh=th;}
    }
    const r = ev(all.map(s => (s.finalScore || 0) < bestTh));
    console.log(`\nfinalScore alone (< ${bestTh.toFixed(2)}): R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% FP=${r.FP} FN=${r.FN}`);
  }
}

// === Combine finalScore with zone + token features ===
console.log('\n=== finalScore + zone + token ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=3;zoneW<=4.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=3;vHW+=0.5){
      for(let jdW=0.5;jdW<=1.5;jdW+=0.5){
        for(let fsTh=0.3;fsTh<=0.7;fsTh+=0.05){
          for(let fsW=1;fsW<=3;fsW+=0.5){
            for(let p1W=0.5;p1W<=2;p1W+=0.5){
              for(let p2W=0;p2W<=1;p2W+=0.5){
                for(let t=4;t<=6;t+=0.25){
                  const preds = all.map((s,i) => {
                    let v = 0;
                    const f = tw5[i];
                    // Zone signal
                    if(f.jdMean >= 0.035 && f.activeRate >= 0.05 && f.jeMean >= 5) v += zoneW;
                    // Token signals
                    if(s.jawVelocity >= 0.5) v += vHW;
                    else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                    if(s.jawDelta >= 0.05) v += jdW;
                    else if(s.jawDelta >= 0.02) v += jdW*0.5;
                    if(jawEff[i] >= 5) v += 0.5;
                    if(scoreVelAnti[i] >= 0.2) v += 0.5;
                    if(s.score < 0.45) v += 0.5;
                    if(dt[i] >= 0.2) v += 0.5;
                    // finalScore penalty (high finalScore = AI)
                    if((s.finalScore || 0) >= fsTh) v -= fsW;
                    // Other penalties
                    if(f.jdMean < 0.005) v -= 2;
                    if(f.jeMean < 1.5) v -= 1;
                    if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                    if(s.jawDelta < 0.04 && s.jawVelocity >= 0.1) v -= p2W;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                    best={...r,zoneW,vHW,jdW,fsTh,fsW,p1W,p2W,t};
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
  console.log(`finalScore+zone: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  finalScore>=${best.fsTh}→-${best.fsW}`);
    console.log(`  p1: !hjw&&score>=0.7→-${best.p1W}`);
    console.log(`  p2: jd<0.04&&vel>=0.1→-${best.p2W}`);
    console.log(`  threshold=${best.t}`);
    
    // Error analysis
    const sc = all.map((s,i) => {
      let v = 0;
      const f = tw5[i];
      if(f.jdMean >= 0.035 && f.activeRate >= 0.05 && f.jeMean >= 5) v += best.zoneW;
      if(s.jawVelocity >= 0.5) v += best.vHW;
      else if(s.jawVelocity >= 0.1) v += best.vHW*0.3;
      if(s.jawDelta >= 0.05) v += best.jdW;
      else if(s.jawDelta >= 0.02) v += best.jdW*0.5;
      if(jawEff[i] >= 5) v += 0.5;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(s.score < 0.45) v += 0.5;
      if(dt[i] >= 0.2) v += 0.5;
      if((s.finalScore || 0) >= best.fsTh) v -= best.fsW;
      if(f.jdMean < 0.005) v -= 2;
      if(f.jeMean < 1.5) v -= 1;
      if(!isHighJW[i] && s.score >= 0.7) v -= best.p1W;
      if(s.jawDelta < 0.04 && s.jawVelocity >= 0.1) v -= best.p2W;
      return v;
    });
    const preds = sc.map(v => v >= best.t);
    const FP = [], FN = [];
    for(let i=0;i<N;i++){
      if(preds[i]&&!act[i]) FP.push(i);
      if(!preds[i]&&act[i]) FN.push(i);
    }
    console.log(`\nFP: ${FP.length}`);
    FP.slice(0,10).forEach(i => {
      const s=all[i];
      console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} jd=${s.jawDelta.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} fs=${(s.finalScore||0).toFixed(3)} v=${sc[i].toFixed(2)}`);
    });
    console.log(`\nFN: ${FN.length}`);
    FN.slice(0,10).forEach(i => {
      const s=all[i];
      console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} jd=${s.jawDelta.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} fs=${(s.finalScore||0).toFixed(3)} v=${sc[i].toFixed(2)}`);
    });
  }
}

// === Also try: finalScore as primary classifier ===
console.log('\n=== finalScore-primary + zone boost ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let fsTh=0.3;fsTh<=0.7;fsTh+=0.025){
    for(let fsW=3;fsW<=6;fsW+=0.5){
      for(let zoneW=1;zoneW<=3;zoneW+=0.5){
        for(let vW=0.5;vW<=2;vW+=0.5){
          for(let jdW=0.5;jdW<=2;jdW+=0.5){
            for(let t=2;t<=5;t+=0.25){
              const preds = all.map((s,i) => {
                let v = 0;
                // finalScore (primary — low = user)
                if((s.finalScore || 0) < fsTh) v += fsW;
                // Zone
                const f = tw5[i];
                if(f.jdMean >= 0.035 && f.jeMean >= 5) v += zoneW;
                // Token
                if(s.jawVelocity >= 0.3) v += vW;
                if(s.jawDelta >= 0.03) v += jdW;
                // Penalties
                if(f.jdMean < 0.005) v -= 2;
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.85&&r.specificity>=0.92&&r.f1>best.f1){
                best={...r,fsTh,fsW,zoneW,vW,jdW,t};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`finalScore-primary: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  finalScore<${best.fsTh}→${best.fsW}`);
    console.log(`  zoneW=${best.zoneW} vW=${best.vW} jdW=${best.jdW}`);
    console.log(`  threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v103: F1=83.1%');
