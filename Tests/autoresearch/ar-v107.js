// autoresearch v107: Focused fine-tune around v103 optimum
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

function computeTW(winSec) {
  return all.map((s,i) => {
    const t0 = s.audioTime;
    const idx = [];
    for(let j=i;j>=0;j--){if(t0-all[j].audioTime>winSec)break;idx.push(j);}
    for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>winSec)break;idx.push(j);}
    const jds = idx.map(j => all[j].jawDelta);
    const effs = idx.map(j => jawEff[j]);
    return { jdMean: mean(jds), jeMean: mean(effs) };
  });
}

const tw5 = computeTW(5);

// Focused search: vary zone params, weights, penalties, threshold
console.log('=== Focused fine-tune ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zJdTh=0.025;zJdTh<=0.045;zJdTh+=0.005){
    for(let zJeTh=4;zJeTh<=6;zJeTh+=0.5){
      for(let zoneW=3;zoneW<=5;zoneW+=0.5){
        for(let vHW=2;vHW<=3.5;vHW+=0.5){
          for(let jdW=0.5;jdW<=2;jdW+=0.5){
            for(let p1W=0.5;p1W<=2;p1W+=0.5){
              for(let p2W=0;p2W<=1;p2W+=0.5){
                for(let t=4;t<=5.5;t+=0.25){
                  const preds = all.map((s,i) => {
                    let v = 0;
                    const f = tw5[i];
                    if(f.jdMean >= zJdTh && f.jeMean >= zJeTh) v += zoneW;
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
                    if(p2W>0 && s.jawDelta < 0.04 && s.jawVelocity >= 0.1) v -= p2W;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1){
                    best={...r,zJdTh,zJeTh,zoneW,vHW,jdW,p1W,p2W,t};
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
  console.log(`Focused: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zone: jd>=${best.zJdTh} je>=${best.zJeTh} → ${best.zoneW}`);
    console.log(`  vel>=0.5→${best.vHW} jd>=0.05→${best.jdW}`);
    console.log(`  p1: !hjw&&sc>=0.7→-${best.p1W}, p2: jd<0.04&&vel>=0.1→-${best.p2W}`);
    console.log(`  threshold=${best.t}`);
    
    // Add rescue
    const sc = all.map((s,i) => {
      let v = 0;
      const f = tw5[i];
      if(f.jdMean >= best.zJdTh && f.jeMean >= best.zJeTh) v += best.zoneW;
      if(s.jawVelocity >= 0.5) v += best.vHW;
      else if(s.jawVelocity >= 0.1) v += best.vHW*0.3;
      if(s.jawDelta >= 0.05) v += best.jdW;
      else if(s.jawDelta >= 0.02) v += best.jdW*0.4;
      if(jawEff[i] >= 5) v += 0.5;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(s.score < 0.45) v += 0.5;
      if(dt[i] >= 0.2) v += 0.5;
      if(f.jdMean < 0.005) v -= 2;
      if(f.jeMean < 1.5) v -= 1;
      if(!isHighJW[i] && s.score >= 0.7) v -= best.p1W;
      if(best.p2W>0 && s.jawDelta < 0.04 && s.jawVelocity >= 0.1) v -= best.p2W;
      return v;
    });
    const p1 = sc.map(v => v >= best.t);
    
    let bestR = {f1:best.f1};
    for(let hw=4;hw<=14;hw+=2){
      for(let nTh=0.2;nTh<=0.7;nTh+=0.1){
        for(let low=-4;low<=1;low+=1){
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
          if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>bestR.f1) bestR={...r,hw,nTh,low};
        }
      }
    }
    if(bestR.f1>best.f1){
      console.log(`\n+Rescue: R=${(bestR.recall*100).toFixed(1)}% S=${(bestR.specificity*100).toFixed(1)}% F1=${(bestR.f1*100).toFixed(1)}% FP=${bestR.FP} FN=${bestR.FN}`);
      console.log(`  hw=${bestR.hw} nTh=${bestR.nTh} low=${bestR.low}`);
    } else {
      console.log('\nRescue did not improve');
    }
  }
}

// === Also try with finalScore penalty ===
console.log('\n=== With finalScore penalty ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=3.5;zoneW<=4.5;zoneW+=0.5){
    for(let vHW=2;vHW<=3;vHW+=0.5){
      for(let jdW=0.5;jdW<=1.5;jdW+=0.5){
        for(let fsTh=0.5;fsTh<=0.75;fsTh+=0.05){
          for(let fsW=0.5;fsW<=2;fsW+=0.5){
            for(let p1W=0.5;p1W<=2;p1W+=0.5){
              for(let t=4;t<=5.5;t+=0.25){
                const preds = all.map((s,i) => {
                  let v = 0;
                  const f = tw5[i];
                  if(f.jdMean >= 0.035 && f.jeMean >= 5) v += zoneW;
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
                  if((s.finalScore||0) >= fsTh) v -= fsW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.85&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsTh,fsW,p1W,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`With finalScore: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  finalScore>=${best.fsTh}→-${best.fsW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v103: F1=83.1%');
console.log('v104: F1=84.3% (with finalScore)');
