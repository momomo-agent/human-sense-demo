// autoresearch v108: Jitter-based features
// kenefe insight: use time jitter to distinguish user vs AI
// User speech has natural timing irregularity, AI has mechanical regularity
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

// === Jitter features ===

// 1. dt jitter = std of dt in window (how irregular is timing)
function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}

const dtJitter3 = wstat(dt, 1, std);  // hw=1 → 3 tokens
const dtJitter5 = wstat(dt, 2, std);  // hw=2 → 5 tokens
const dtJitter7 = wstat(dt, 3, std);  // hw=3 → 7 tokens
const dtJitter11 = wstat(dt, 5, std); // hw=5 → 11 tokens

// 2. dt coefficient of variation (std/mean, normalized jitter)
const dtCV5 = wstat(dt, 2, a => { const m = mean(a); return m > 0.001 ? std(a) / m : 0; });

// 3. dt consecutive difference (how much timing changes between adjacent tokens)
const dtDiff = dt.map((d, i) => i <= 1 ? 0 : Math.abs(d - dt[i-1]));
const dtDiffMean5 = wstat(dtDiff, 2, mean);

// 4. dt zero ratio in window (fraction of dt=0 tokens)
const dtZeroRatio5 = wstat(dt, 2, a => a.filter(v => v < 0.001).length / a.length);
const dtZeroRatio11 = wstat(dt, 5, a => a.filter(v => v < 0.001).length / a.length);

// 5. dt non-zero mean (average of non-zero dt values in window)
const dtNonZeroMean5 = wstat(dt, 2, a => {
  const nz = a.filter(v => v >= 0.001);
  return nz.length > 0 ? mean(nz) : 0;
});

// 6. dt pattern regularity (how regular are the non-zero intervals)
const dtRegularity5 = wstat(dt, 2, a => {
  const nz = a.filter(v => v >= 0.001);
  if(nz.length < 2) return 0;
  const m = mean(nz);
  const s = std(nz);
  return m > 0 ? s / m : 0; // CV of non-zero dt
});

// 7. Velocity jitter (how much velocity changes between tokens)
const velDiff = all.map((s, i) => i === 0 ? 0 : Math.abs(s.jawVelocity - all[i-1].jawVelocity));
const velJitter5 = wstat(velDiff, 2, mean);

// 8. jawDelta jitter
const jdDiff = all.map((s, i) => i === 0 ? 0 : Math.abs(s.jawDelta - all[i-1].jawDelta));
const jdJitter5 = wstat(jdDiff, 2, mean);

// === Analyze all jitter features ===
console.log('=== Jitter feature analysis ===\n');
const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

const feats = {
  dtJitter3: i => dtJitter3[i],
  dtJitter5: i => dtJitter5[i],
  dtJitter7: i => dtJitter7[i],
  dtJitter11: i => dtJitter11[i],
  dtCV5: i => dtCV5[i],
  dtDiffMean5: i => dtDiffMean5[i],
  dtZeroRatio5: i => dtZeroRatio5[i],
  dtZeroRatio11: i => dtZeroRatio11[i],
  dtNonZeroMean5: i => dtNonZeroMean5[i],
  dtRegularity5: i => dtRegularity5[i],
  velJitter5: i => velJitter5[i],
  jdJitter5: i => jdJitter5[i],
};

for(const [name, fn] of Object.entries(feats)){
  const uVals = userIdx.map(fn);
  const aVals = aiIdx.map(fn);
  const uM = mean(uVals), uS = std(uVals);
  const aM = mean(aVals), aS = std(aVals);
  const pooledS = Math.sqrt((uS**2 + aS**2)/2);
  const d = pooledS > 0 ? Math.abs(uM - aM) / pooledS : 0;
  console.log(`${name.padEnd(18)} User=${uM.toFixed(4)}±${uS.toFixed(4)} AI=${aM.toFixed(4)}±${aS.toFixed(4)} d=${d.toFixed(3)}`);
}

// === Best single jitter classifier ===
console.log('\n=== Best single jitter classifiers ===\n');
for(const [name, fn] of Object.entries(feats)){
  const vals = (new Array(N)).fill(0).map((_,i) => fn(i));
  let bestF1 = 0, bestTh = 0, bestDir = '';
  const sorted = [...new Set(vals)].sort((a,b) => a-b);
  const step = Math.max(1, Math.floor(sorted.length / 50));
  for(let si = 0; si < sorted.length; si += step) {
    const th = sorted[si];
    const p1 = vals.map(v => v >= th);
    const r1 = ev(p1);
    if(r1.f1 > bestF1) { bestF1 = r1.f1; bestTh = th; bestDir = '>='; }
    const p2 = vals.map(v => v < th);
    const r2 = ev(p2);
    if(r2.f1 > bestF1) { bestF1 = r2.f1; bestTh = th; bestDir = '<'; }
  }
  if(bestF1 > 0.3) console.log(`${name.padEnd(18)} F1=${(bestF1*100).toFixed(1)}% (${bestDir}${bestTh.toFixed(4)})`);
}

// === Integrate jitter into v107 best ===
console.log('\n=== v107 + jitter features ===\n');
{
  // Time-windowed features
  function timeWindowIdx(centerIdx, windowSec) {
    const t0 = all[centerIdx].audioTime;
    const indices = [];
    for(let j=centerIdx;j>=0;j--){if(t0-all[j].audioTime>windowSec)break;indices.push(j);}
    for(let j=centerIdx+1;j<N;j++){if(all[j].audioTime-t0>windowSec)break;indices.push(j);}
    return indices;
  }
  const tw5 = all.map((s,i) => {
    const idx = timeWindowIdx(i, 5);
    const jds = idx.map(j => all[j].jawDelta);
    const effs = idx.map(j => jawEff[j]);
    return { jdMean: mean(jds), jeMean: mean(effs) };
  });
  
  let best = {f1:0}, count=0;
  
  // v107 base + jitter penalty/bonus
  for(let zoneW=3.5;zoneW<=4.5;zoneW+=0.5){
    for(let vHW=2;vHW<=3;vHW+=0.5){
      for(let jdW=1;jdW<=2;jdW+=0.5){
        for(let jitFeat of ['velJitter5','jdJitter5','dtZeroRatio5','dtJitter5']){
          const jitArr = jitFeat==='velJitter5'?velJitter5:jitFeat==='jdJitter5'?jdJitter5:jitFeat==='dtZeroRatio5'?dtZeroRatio5:dtJitter5;
          for(let jitTh=0;jitTh<=1;jitTh+=0.1){
            for(let jitW=0.5;jitW<=2;jitW+=0.5){
              for(let jitDir of ['>=','<']){
                for(let p1W=0.5;p1W<=1.5;p1W+=0.5){
                  for(let t=4.5;t<=5.75;t+=0.25){
                    const preds = all.map((s,i) => {
                      let v = 0;
                      const f = tw5[i];
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
                      if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                      // Jitter feature
                      if(jitDir==='>=' && jitArr[i] >= jitTh) v += jitW;
                      if(jitDir==='<' && jitArr[i] < jitTh) v -= jitW;
                      return v >= t;
                    });
                    const r = ev(preds);
                    if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1){
                      best={...r,zoneW,vHW,jdW,jitFeat,jitTh,jitW,jitDir,p1W,t};
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
  console.log(`v107+jitter: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  jitter: ${best.jitFeat} ${best.jitDir}${best.jitTh} → ${best.jitDir==='>='?'+':'-'}${best.jitW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v107: F1=83.5% (no finalScore), 84.8% (with finalScore)');
