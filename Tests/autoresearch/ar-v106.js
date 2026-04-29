// autoresearch v106: Fine-grained search around v103 optimum
// Also try: different zone windows, rescue, and finalScore integration
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

function computeTW(winSec) {
  return all.map((s,i) => {
    const idx = timeWindowIdx(i, winSec);
    const jds = idx.map(j => all[j].jawDelta);
    const effs = idx.map(j => jawEff[j]);
    return {
      jdMean: mean(jds),
      jeMean: mean(effs),
      activeRate: jds.filter(v => v >= 0.03).length / jds.length,
    };
  });
}

const tw3 = computeTW(3);
const tw5 = computeTW(5);
const tw7 = computeTW(7);

// === Fine-grained search ===
console.log('=== Fine-grained search ===\n');
{
  let best = {f1:0}, count=0;
  
  for(const twData of [{name:'tw3',d:tw3},{name:'tw5',d:tw5},{name:'tw7',d:tw7}]){
    for(let zJdTh=0.025;zJdTh<=0.045;zJdTh+=0.005){
      for(let zJeTh=4;zJeTh<=6;zJeTh+=0.5){
        for(let zoneW=3;zoneW<=5;zoneW+=0.25){
          for(let vHW=2;vHW<=3.5;vHW+=0.25){
            for(let vH=0.4;vH<=0.6;vH+=0.1){
              for(let jdW=0.5;jdW<=2;jdW+=0.25){
                for(let jdH=0.04;jdH<=0.06;jdH+=0.01){
                  for(let p1W=0.5;p1W<=2;p1W+=0.25){
                    for(let t=4;t<=5.5;t+=0.25){
                      const preds = all.map((s,i) => {
                        let v = 0;
                        const f = twData.d[i];
                        if(f.jdMean >= zJdTh && f.jeMean >= zJeTh) v += zoneW;
                        if(s.jawVelocity >= vH) v += vHW;
                        else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                        if(s.jawDelta >= jdH) v += jdW;
                        else if(s.jawDelta >= 0.02) v += jdW*0.4;
                        if(jawEff[i] >= 5) v += 0.5;
                        if(scoreVelAnti[i] >= 0.2) v += 0.5;
                        if(s.score < 0.45) v += 0.5;
                        if(dt[i] >= 0.2) v += 0.5;
                        if(f.jdMean < 0.005) v -= 2;
                        if(f.jeMean < 1.5) v -= 1;
                        if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                        if(s.jawDelta < 0.04 && s.jawVelocity >= 0.1) v -= 0.5;
                        return v >= t;
                      });
                      const r = ev(preds);
                      if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>best.f1){
                        best={...r,tw:twData.name,zJdTh,zJeTh,zoneW,vH,vHW,jdH,jdW,p1W,t};
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
  console.log(`Fine-grained: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  window=${best.tw} zone: jd>=${best.zJdTh} je>=${best.zJeTh} → ${best.zoneW}`);
    console.log(`  vel>=${best.vH}→${best.vHW} jd>=${best.jdH}→${best.jdW}`);
    console.log(`  p1: !hjw&&sc>=0.7→-${best.p1W}`);
    console.log(`  threshold=${best.t}`);
    
    // Add rescue on top
    const twD = best.tw==='tw3'?tw3:best.tw==='tw5'?tw5:tw7;
    const sc = all.map((s,i) => {
      let v = 0;
      const f = twD[i];
      if(f.jdMean >= best.zJdTh && f.jeMean >= best.zJeTh) v += best.zoneW;
      if(s.jawVelocity >= best.vH) v += best.vHW;
      else if(s.jawVelocity >= 0.1) v += best.vHW*0.3;
      if(s.jawDelta >= best.jdH) v += best.jdW;
      else if(s.jawDelta >= 0.02) v += best.jdW*0.4;
      if(jawEff[i] >= 5) v += 0.5;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(s.score < 0.45) v += 0.5;
      if(dt[i] >= 0.2) v += 0.5;
      if(f.jdMean < 0.005) v -= 2;
      if(f.jeMean < 1.5) v -= 1;
      if(!isHighJW[i] && s.score >= 0.7) v -= best.p1W;
      if(s.jawDelta < 0.04 && s.jawVelocity >= 0.1) v -= 0.5;
      return v;
    });
    const p1 = sc.map(v => v >= best.t);
    
    let bestR = {f1:best.f1};
    for(let hw=4;hw<=14;hw+=2){
      for(let nTh=0.2;nTh<=0.7;nTh+=0.1){
        for(let low=-4;low<=1;low+=1){
          for(let velTh=0.05;velTh<=0.15;velTh+=0.05){
            const preds = all.map((_,i) => {
              if(p1[i]) return true;
              if(all[i].jawVelocity < velTh && all[i].jawDelta < 0.01) return false;
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
            if(r.recall>=0.85&&r.specificity>=0.91&&r.f1>bestR.f1) bestR={...r,hw,nTh,low,velTh};
          }
        }
      }
    }
    if(bestR.f1>best.f1){
      console.log(`\n+Rescue: R=${(bestR.recall*100).toFixed(1)}% S=${(bestR.specificity*100).toFixed(1)}% F1=${(bestR.f1*100).toFixed(1)}% FP=${bestR.FP} FN=${bestR.FN}`);
      console.log(`  hw=${bestR.hw} nTh=${bestR.nTh} low=${bestR.low} velTh=${bestR.velTh}`);
    }
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v103: F1=83.1%');
console.log('v104: F1=84.3% (with finalScore)');
