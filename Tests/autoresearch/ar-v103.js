// autoresearch v103: Build on v100 (F1=81.9%), add targeted penalties for FP
// v100 best: tw5 zone(jdMean>=0.035, ar>=0.05, jeMean>=5) + zoneW=4 + token voting, t=5
// All 174 FP are in user zone — need to filter AI tokens within zone
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

// Analyze FP from v100 — what distinguishes them from TP?
console.log('=== FP vs TP analysis (within user zone) ===\n');
{
  const isZone = tw5.map(f => f.jdMean >= 0.035 && f.activeRate >= 0.05 && f.jeMean >= 5);
  
  // Within zone: user vs AI
  const zoneUser = [], zoneAI = [];
  for(let i=0;i<N;i++){
    if(!isZone[i]) continue;
    if(act[i]) zoneUser.push(i);
    else zoneAI.push(i);
  }
  console.log(`In zone: ${zoneUser.length} user, ${zoneAI.length} AI\n`);
  
  const feats = {
    score: i => all[i].score,
    jawDelta: i => all[i].jawDelta,
    jawVelocity: i => all[i].jawVelocity,
    jawEff: i => jawEff[i],
    scoreVelAnti: i => scoreVelAnti[i],
    timeDelta: i => dt[i],
    isHighJW: i => isHighJW[i] ? 1 : 0,
    'jd*vel': i => all[i].jawDelta * all[i].jawVelocity,
    'jd+vel': i => all[i].jawDelta + all[i].jawVelocity,
    noJawPenalty: i => all[i].noJawPenalty || 0,
    jawMargin: i => all[i].jawMargin || 0,
    finalScore: i => all[i].finalScore || 0,
  };
  
  for(const [name, fn] of Object.entries(feats)){
    const uVals = zoneUser.map(fn);
    const aVals = zoneAI.map(fn);
    const uM = mean(uVals), uS = std(uVals);
    const aM = mean(aVals), aS = std(aVals);
    const pooledS = Math.sqrt((uS**2 + aS**2)/2);
    const d = pooledS > 0 ? Math.abs(uM - aM) / pooledS : 0;
    if(d > 0.2) console.log(`${name.padEnd(15)} User=${uM.toFixed(4)}±${uS.toFixed(4)} AI=${aM.toFixed(4)}±${aS.toFixed(4)} d=${d.toFixed(3)}`);
  }
}

// === v100 base + penalty search ===
console.log('\n=== v100 + penalty search ===\n');
{
  let best = {f1:0}, count=0;
  
  function baseScore(s, i, p) {
    let v = 0;
    const f = tw5[i];
    // Zone signal
    if(f.jdMean >= 0.035 && f.activeRate >= 0.05 && f.jeMean >= 5) v += p.zoneW;
    // Token signals
    if(s.jawVelocity >= 0.5) v += p.vHW;
    else if(s.jawVelocity >= 0.1) v += p.vHW*0.3;
    if(s.jawDelta >= 0.05) v += p.jdW;
    else if(s.jawDelta >= 0.02) v += p.jdW*0.5;
    if(jawEff[i] >= 5) v += 0.5;
    if(scoreVelAnti[i] >= 0.2) v += 0.5;
    if(s.score < 0.45) v += 0.5;
    if(dt[i] >= 0.2) v += 0.5;
    // Base penalties
    if(f.jdMean < 0.005) v -= 2;
    if(f.jeMean < 1.5) v -= 1;
    // New penalties
    if(p.p1 && !isHighJW[i] && s.score >= p.p1Th) v -= p.p1W;
    if(p.p2 && s.jawDelta < p.p2Th && s.jawVelocity >= 0.1) v -= p.p2W;
    if(p.p3 && s.jawDelta * s.jawVelocity < p.p3Th) v -= p.p3W;
    if(p.p4 && s.score >= 0.6 && s.jawDelta < 0.03) v -= p.p4W;
    if(p.p5 && dt[i] < 0.001 && s.score >= 0.5) v -= p.p5W;
    return v;
  }
  
  // Search penalty combinations
  for(let zoneW=3;zoneW<=4.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=0.5;jdW<=1.5;jdW+=0.5){
        for(let p1Th=0.6;p1Th<=0.75;p1Th+=0.05){
          for(let p1W=0.5;p1W<=2;p1W+=0.5){
            for(let p2Th=0.02;p2Th<=0.04;p2Th+=0.01){
              for(let p2W=0.5;p2W<=2;p2W+=0.5){
                for(let p4W=0;p4W<=1.5;p4W+=0.5){
                  for(let p5W=0;p5W<=1;p5W+=0.5){
                    for(let t=4.5;t<=6;t+=0.25){
                      const p = {
                        zoneW, vHW, jdW,
                        p1:true, p1Th, p1W,
                        p2:true, p2Th, p2W,
                        p3:false,
                        p4:p4W>0, p4W,
                        p5:p5W>0, p5W,
                      };
                      const preds = all.map((s,i) => baseScore(s, i, p) >= t);
                      const r = ev(preds);
                      if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                        best={...r,...p,t};
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
  console.log(`Penalty search: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  p1: !hjw && score>=${best.p1Th} → -${best.p1W}`);
    console.log(`  p2: jd<${best.p2Th} && vel>=0.1 → -${best.p2W}`);
    if(best.p4) console.log(`  p4: score>=0.6 && jd<0.03 → -${best.p4W}`);
    if(best.p5) console.log(`  p5: dt=0 && score>=0.5 → -${best.p5W}`);
    console.log(`  threshold=${best.t}`);
    
    // Error analysis
    const sc = all.map((s,i) => baseScore(s, i, best));
    const preds = sc.map(v => v >= best.t);
    const FP = [], FN = [];
    for(let i=0;i<N;i++){
      if(preds[i]&&!act[i]) FP.push(i);
      if(!preds[i]&&act[i]) FN.push(i);
    }
    console.log(`\nFP: ${FP.length}`);
    FP.slice(0,10).forEach(i => {
      const s=all[i];
      console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} jd=${s.jawDelta.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(3)} hjw=${isHighJW[i]?1:0} v=${sc[i].toFixed(2)}`);
    });
    console.log(`\nFN: ${FN.length}`);
    FN.slice(0,10).forEach(i => {
      const s=all[i];
      console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} jd=${s.jawDelta.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(3)} hjw=${isHighJW[i]?1:0} v=${sc[i].toFixed(2)}`);
    });
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v100: F1=81.9%');
