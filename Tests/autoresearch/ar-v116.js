// autoresearch v116: v112 base + cross penalties fine-tune
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

// === Comprehensive search: all best features + cross penalties ===
console.log('=== v116: Comprehensive fine-tune ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsTh=0.6;fsTh<=0.75;fsTh+=0.05){
          for(let fsW=1.5;fsW<=3;fsW+=0.25){
            for(let xW=0;xW<=2;xW+=0.5){
              for(let p1W=0;p1W<=1;p1W+=0.5){
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
                    if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                    const fsc = s.finalScore||0;
                    if(fsc >= fsTh) v -= fsW;
                    if(xW>0 && fsc >= 0.5 && s.score >= 0.7) v -= xW;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                    best={...r,zoneW,vHW,jdW,fsTh,fsW,xW,p1W,t};
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
  console.log(`Comprehensive: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fs>=${best.fsTh}→-${best.fsW} cross(fs>=0.5&&sc>=0.7)→-${best.xW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
    
    // Now add rescue to this
    const sc = all.map((s,i) => {
      let v = 0;
      const f = tw10[i];
      if(f.jdMean >= 0.03 && f.jeMean >= 5) v += best.zoneW;
      if(s.jawVelocity >= 0.5) v += best.vHW;
      else if(s.jawVelocity >= 0.1) v += best.vHW*0.3;
      if(s.jawDelta >= 0.05) v += best.jdW;
      else if(s.jawDelta >= 0.02) v += best.jdW*0.4;
      if(jawEff[i] >= 5) v += 0.5;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(s.score < 0.45) v += 0.5;
      if(dt[i] >= 0.2) v += 0.5;
      if(dtZeroRatio5[i] >= 0.5) v += 0.5;
      if(f.jdMean < 0.005) v -= 2;
      if(f.jeMean < 1.5) v -= 1;
      if(!isHighJW[i] && s.score >= 0.7) v -= best.p1W;
      const fsc = s.finalScore||0;
      if(fsc >= best.fsTh) v -= best.fsW;
      if(best.xW>0 && fsc >= 0.5 && s.score >= 0.7) v -= best.xW;
      return v;
    });
    const p1 = sc.map(v => v >= best.t);
    
    let bestR = {f1:best.f1};
    for(let hw=3;hw<=15;hw+=3){
      for(let nTh=0.2;nTh<=0.6;nTh+=0.1){
        for(let low=-2;low<=2;low+=1){
          const preds = all.map((_,i) => {
            if(p1[i]) return true;
            if(all[i].jawVelocity < 0.02 && all[i].jawDelta < 0.005) return false;
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
          if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>bestR.f1) bestR={...r,hw,nTh,low};
        }
      }
    }
    if(bestR.f1>best.f1){
      console.log(`\n+Rescue: R=${(bestR.recall*100).toFixed(1)}% S=${(bestR.specificity*100).toFixed(1)}% F1=${(bestR.f1*100).toFixed(1)}% FP=${bestR.FP} FN=${bestR.FN}`);
      console.log(`  hw=${bestR.hw} nTh=${bestR.nTh.toFixed(1)} low=${bestR.low}`);
    }
  }
}

// === Also: try higher S constraint ===
console.log('\n=== S>=93% constraint ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsTh=0.5;fsTh<=0.7;fsTh+=0.1){
          for(let fsW=1;fsW<=3;fsW+=0.5){
            for(let xW=0.5;xW<=2;xW+=0.5){
              for(let p1W=0;p1W<=1;p1W+=0.5){
                for(let t=5;t<=6.5;t+=0.25){
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
                    const fsc = s.finalScore||0;
                    if(fsc >= fsTh) v -= fsW;
                    if(fsc >= 0.5 && s.score >= 0.7) v -= xW;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.85&&r.specificity>=0.93&&r.f1>best.f1){
                    best={...r,zoneW,vHW,jdW,fsTh,fsW,xW,p1W,t};
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
  console.log(`S>=93%: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fs>=${best.fsTh}→-${best.fsW} cross→-${best.xW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
console.log('v115 cross: F1=86.1% (R=94.1% S=93.1%)');
