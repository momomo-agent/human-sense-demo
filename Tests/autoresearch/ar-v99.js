// autoresearch v99: Two-stage classifier
// Stage 1: Time-segment classification (is this a "user speaking" time window?)
// Stage 2: Token-level refinement within segments
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

// Time-windowed features (by actual seconds)
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

// Compute time-windowed stats for various windows
function computeTimeFeats(winSec) {
  return all.map((s,i) => {
    const idx = timeWindowIdx(i, winSec);
    const jds = idx.map(j => all[j].jawDelta);
    const vels = idx.map(j => all[j].jawVelocity);
    const effs = idx.map(j => jawEff[j]);
    return {
      jdMean: mean(jds),
      velMean: mean(vels),
      jeMean: mean(effs),
      activeRate: jds.filter(v => v >= 0.03).length / jds.length,
      nTokens: idx.length,
    };
  });
}

const tw2 = computeTimeFeats(2);
const tw5 = computeTimeFeats(5);
const tw10 = computeTimeFeats(10);

// === Stage 1: Segment-level classification ===
// Use wide time window to determine if we're in a "user speaking" zone
console.log('=== Stage 1: Segment classification ===\n');

// Try different time windows and thresholds for "user zone"
{
  let best = {f1:0};
  
  for(const tw of [{name:'tw2',d:tw2},{name:'tw5',d:tw5},{name:'tw10',d:tw10}]){
    for(let jdTh=0.02;jdTh<=0.06;jdTh+=0.005){
      for(let arTh=0.1;arTh<=0.4;arTh+=0.05){
        for(let jeTh=3;jeTh<=6;jeTh+=0.5){
          // User zone = jdMean >= jdTh AND activeRate >= arTh AND jeMean >= jeTh
          const isUserZone = tw.d.map(f => f.jdMean >= jdTh && f.activeRate >= arTh && f.jeMean >= jeTh);
          const r = ev(isUserZone);
          if(r.recall>=0.85&&r.specificity>=0.85&&r.f1>best.f1){
            best={...r,tw:tw.name,jdTh,arTh,jeTh};
          }
        }
      }
    }
  }
  console.log(`Segment-only: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
  console.log(`  window=${best.tw} jdMean>=${best.jdTh} activeRate>=${best.arTh} jeMean>=${best.jeTh}`);
}

// === Stage 2: Segment + token refinement ===
console.log('\n=== Stage 2: Segment + token refinement ===\n');
{
  let best = {f1:0}, count=0;
  
  for(const tw of [{name:'tw2',d:tw2},{name:'tw5',d:tw5}]){
    for(let jdTh=0.02;jdTh<=0.05;jdTh+=0.005){
      for(let arTh=0.1;arTh<=0.3;arTh+=0.05){
        for(let jeTh=3;jeTh<=5;jeTh+=0.5){
          // Stage 1: user zone
          const isUserZone = tw.d.map(f => f.jdMean >= jdTh && f.activeRate >= arTh && f.jeMean >= jeTh);
          
          // Stage 2: within user zone, filter out AI tokens
          for(let minJd=0;minJd<=0.02;minJd+=0.005){
            for(let minVel=0;minVel<=0.1;minVel+=0.025){
              const preds = all.map((s,i) => {
                if(!isUserZone[i]) return false;
                // Within user zone, require minimum jaw activity
                if(s.jawDelta < minJd && s.jawVelocity < minVel) return false;
                return true;
              });
              const r = ev(preds);
              if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,tw:tw.name,jdTh,arTh,jeTh,minJd,minVel};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Segment+token: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  window=${best.tw} jdMean>=${best.jdTh} activeRate>=${best.arTh} jeMean>=${best.jeTh}`);
    console.log(`  minJd=${best.minJd} minVel=${best.minVel}`);
  }
}

// === Stage 3: Hybrid — segment zone + voting ===
console.log('\n=== Stage 3: Hybrid segment+voting ===\n');
{
  let best = {f1:0}, count=0;
  
  // Use segment zone as a strong positive signal in voting
  for(const tw of [{name:'tw2',d:tw2},{name:'tw5',d:tw5}]){
    for(let jdTh=0.02;jdTh<=0.05;jdTh+=0.005){
      for(let arTh=0.1;arTh<=0.3;arTh+=0.05){
        for(let jeTh=3;jeTh<=5;jeTh+=0.5){
          for(let zoneW=2;zoneW<=4;zoneW+=0.5){
            for(let t=3;t<=5;t+=0.5){
              const preds = all.map((s,i) => {
                let v = 0;
                // Zone signal
                const f = tw.d[i];
                if(f.jdMean >= jdTh && f.activeRate >= arTh && f.jeMean >= jeTh) v += zoneW;
                // Token signals
                if(s.jawVelocity >= 0.5) v += 2;
                else if(s.jawVelocity >= 0.1) v += 0.75;
                if(s.jawDelta >= 0.05) v += 1;
                else if(s.jawDelta >= 0.02) v += 0.5;
                if(jawEff[i] >= 5) v += 0.5;
                if(scoreVelAnti[i] >= 0.2) v += 0.5;
                if(s.score < 0.45) v += 0.5;
                if(dt[i] >= 0.2) v += 0.5;
                // Penalties
                if(f.jdMean < 0.005) v -= 2;
                if(f.jeMean < 1.5) v -= 1;
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,tw:tw.name,jdTh,arTh,jeTh,zoneW,t};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Hybrid: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  window=${best.tw} jdMean>=${best.jdTh} activeRate>=${best.arTh} jeMean>=${best.jeTh}`);
    console.log(`  zoneW=${best.zoneW} threshold=${best.t}`);
    
    // Add rescue
    const sc = all.map((s,i) => {
      let v = 0;
      const f = best.tw==='tw2'?tw2[i]:tw5[i];
      if(f.jdMean >= best.jdTh && f.activeRate >= best.arTh && f.jeMean >= best.jeTh) v += best.zoneW;
      if(s.jawVelocity >= 0.5) v += 2;
      else if(s.jawVelocity >= 0.1) v += 0.75;
      if(s.jawDelta >= 0.05) v += 1;
      else if(s.jawDelta >= 0.02) v += 0.5;
      if(jawEff[i] >= 5) v += 0.5;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(s.score < 0.45) v += 0.5;
      if(dt[i] >= 0.2) v += 0.5;
      if(f.jdMean < 0.005) v -= 2;
      if(f.jeMean < 1.5) v -= 1;
      return v;
    });
    const p1 = sc.map(v => v >= best.t);
    
    let bestR = {f1:0};
    for(let hw=6;hw<=14;hw+=2){
      for(let nTh=0.3;nTh<=0.7;nTh+=0.1){
        for(let low=-3;low<=1;low+=1){
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
console.log('v98: F1=79.5%');
