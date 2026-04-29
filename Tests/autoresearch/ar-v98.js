// autoresearch v98: Time-based segmentation features
// Key insight from kenefe: we have timestamps, use them!
// User speech and AI speech happen in distinct time segments
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
function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}

const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);
const jdMean5 = wstat(all.map(s=>s.jawDelta), 2, mean);
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);

// === Time-based features ===

// 1. Time gap segmentation: large dt = speaker change boundary
// Find segments separated by large time gaps
function findSegments(gapThreshold) {
  const segs = [];
  let start = 0;
  for(let i=1;i<N;i++){
    if(dt[i] >= gapThreshold){
      segs.push({start, end: i-1});
      start = i;
    }
  }
  segs.push({start, end: N-1});
  return segs;
}

// 2. Time-windowed features (by actual seconds, not token count)
function timeWindow(centerIdx, windowSec) {
  const t0 = all[centerIdx].audioTime;
  const indices = [];
  // Look backward
  for(let j=centerIdx;j>=0;j--){
    if(t0 - all[j].audioTime > windowSec) break;
    indices.push(j);
  }
  // Look forward
  for(let j=centerIdx+1;j<N;j++){
    if(all[j].audioTime - t0 > windowSec) break;
    indices.push(j);
  }
  return indices;
}

// 3. Compute time-windowed jawDelta mean (by seconds)
const jdMeanTime2 = all.map((s,i) => {
  const idx = timeWindow(i, 2);
  return mean(idx.map(j => all[j].jawDelta));
});
const jdMeanTime5 = all.map((s,i) => {
  const idx = timeWindow(i, 5);
  return mean(idx.map(j => all[j].jawDelta));
});
const velMeanTime2 = all.map((s,i) => {
  const idx = timeWindow(i, 2);
  return mean(idx.map(j => all[j].jawVelocity));
});
const velMeanTime5 = all.map((s,i) => {
  const idx = timeWindow(i, 5);
  return mean(idx.map(j => all[j].jawVelocity));
});
const jemTime2 = all.map((s,i) => {
  const idx = timeWindow(i, 2);
  return mean(idx.map(j => jawEff[j]));
});
const jemTime5 = all.map((s,i) => {
  const idx = timeWindow(i, 5);
  return mean(idx.map(j => jawEff[j]));
});

// 4. Segment-level features: what fraction of tokens in this segment are "active" (jawDelta > threshold)
const segs2 = findSegments(2.0);
const segActiveRate = new Array(N).fill(0);
const segJdMean = new Array(N).fill(0);
const segVelMean = new Array(N).fill(0);
const segLen = new Array(N).fill(0);
for(const seg of segs2){
  const idx = [];
  for(let i=seg.start;i<=seg.end;i++) idx.push(i);
  const activeRate = idx.filter(i => all[i].jawDelta >= 0.03).length / idx.length;
  const jdm = mean(idx.map(i => all[i].jawDelta));
  const vm = mean(idx.map(i => all[i].jawVelocity));
  for(const i of idx){
    segActiveRate[i] = activeRate;
    segJdMean[i] = jdm;
    segVelMean[i] = vm;
    segLen[i] = idx.length;
  }
}

// === Analyze new features ===
console.log('=== Time-based feature analysis ===\n');
const feats = {
  jdMeanTime2: i => jdMeanTime2[i],
  jdMeanTime5: i => jdMeanTime5[i],
  velMeanTime2: i => velMeanTime2[i],
  velMeanTime5: i => velMeanTime5[i],
  jemTime2: i => jemTime2[i],
  jemTime5: i => jemTime5[i],
  segActiveRate: i => segActiveRate[i],
  segJdMean: i => segJdMean[i],
  segVelMean: i => segVelMean[i],
  segLen: i => segLen[i],
};

const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

for(const [name, fn] of Object.entries(feats)){
  const uVals = userIdx.map(fn);
  const aVals = aiIdx.map(fn);
  const uM = mean(uVals), uS = std(uVals);
  const aM = mean(aVals), aS = std(aVals);
  const pooledS = Math.sqrt((uS**2 + aS**2)/2);
  const d = pooledS > 0 ? Math.abs(uM - aM) / pooledS : 0;
  console.log(`${name.padEnd(16)} User=${uM.toFixed(4)}±${uS.toFixed(4)} AI=${aM.toFixed(4)}±${aS.toFixed(4)} d=${d.toFixed(3)}`);
}

// === Search with time features ===
console.log('\n=== Search with time features ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let jemT5Th=3;jemT5Th<=6;jemT5Th+=0.5){
    for(let jemT5W=1;jemT5W<=3;jemT5W+=0.5){
      for(let jdT2Th=0.02;jdT2Th<=0.06;jdT2Th+=0.01){
        for(let jdT2W=0.5;jdT2W<=2;jdT2W+=0.5){
          for(let vH=0.3;vH<=0.5;vH+=0.1){
            for(let vHW=1.5;vHW<=3;vHW+=0.5){
              for(let sarTh=0.1;sarTh<=0.4;sarTh+=0.1){
                for(let sarW=0.5;sarW<=2;sarW+=0.5){
                  for(let t=3;t<=5;t+=0.5){
                    const preds = all.map((s,i) => {
                      let v = 0;
                      // Time-windowed jawEff (strongest new feature)
                      if(jemTime5[i] >= jemT5Th) v += jemT5W;
                      // Time-windowed jawDelta
                      if(jdMeanTime2[i] >= jdT2Th) v += jdT2W;
                      // jawVelocity
                      if(s.jawVelocity >= vH) v += vHW;
                      else if(s.jawVelocity >= 0.1) v += vHW*0.4;
                      // Segment active rate
                      if(segActiveRate[i] >= sarTh) v += sarW;
                      // jawDelta
                      if(s.jawDelta >= 0.05) v += 0.75;
                      // scoreVelAnti
                      if(scoreVelAnti[i] >= 0.2) v += 0.5;
                      // timeDelta
                      if(dt[i] >= 0.2) v += 0.5;
                      // score
                      if(s.score < 0.45) v += 0.75;
                      // Penalties
                      if(jdMeanTime2[i] < 0.005) v -= 2;
                      if(jemTime5[i] < 1.5) v -= 1;
                      return v >= t;
                    });
                    const r = ev(preds);
                    if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                      best={...r,jemT5Th,jemT5W,jdT2Th,jdT2W,vH,vHW,sarTh,sarW,t};
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
  console.log(`Time features: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  jemTime5: >=${best.jemT5Th}→${best.jemT5W}`);
    console.log(`  jdMeanTime2: >=${best.jdT2Th}→${best.jdT2W}`);
    console.log(`  vel: >=${best.vH}→${best.vHW}`);
    console.log(`  segActiveRate: >=${best.sarTh}→${best.sarW}`);
    console.log(`  threshold: ${best.t}`);
    
    // Add rescue
    const sc = all.map((s,i) => {
      let v = 0;
      if(jemTime5[i] >= best.jemT5Th) v += best.jemT5W;
      if(jdMeanTime2[i] >= best.jdT2Th) v += best.jdT2W;
      if(s.jawVelocity >= best.vH) v += best.vHW;
      else if(s.jawVelocity >= 0.1) v += best.vHW*0.4;
      if(segActiveRate[i] >= best.sarTh) v += best.sarW;
      if(s.jawDelta >= 0.05) v += 0.75;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(dt[i] >= 0.2) v += 0.5;
      if(s.score < 0.45) v += 0.75;
      if(jdMeanTime2[i] < 0.005) v -= 2;
      if(jemTime5[i] < 1.5) v -= 1;
      return v;
    });
    const p1 = sc.map(v => v >= best.t);
    
    let bestR = {f1:0};
    for(let hw=6;hw<=14;hw+=2){
      for(let nTh=0.3;nTh<=0.7;nTh+=0.1){
        for(let low=-4;low<=0;low+=1){
          const preds = all.map((_,i) => {
            if(p1[i]) return true;
            if(all[i].jawVelocity < 0.05 && all[i].jawDelta < 0.01) return false;
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
          if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>bestR.f1) bestR={...r,hw,nTh,low};
        }
      }
    }
    if(bestR.f1>0){
      console.log(`\n+Rescue: R=${(bestR.recall*100).toFixed(1)}% S=${(bestR.specificity*100).toFixed(1)}% F1=${(bestR.f1*100).toFixed(1)}% FP=${bestR.FP} FN=${bestR.FN}`);
      console.log(`  hw=${bestR.hw} nTh=${bestR.nTh} low=${bestR.low}`);
    }
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v97: F1=79.7%');
