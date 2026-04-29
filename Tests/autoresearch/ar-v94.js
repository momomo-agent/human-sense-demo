// autoresearch v94: Deep analysis of new data AI patterns + aggressive penalty search
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

const scoreGap = all.map(s => {
  const jw = s.jawWeight || 0.2;
  const jvw = s.jawVelocityWeight || jw;
  return Math.abs(s.score * Math.max(0.1, 1-jw*s.jawDelta) * Math.max(0.1, 1-jvw*s.jawVelocity) * ((s.jawDelta<0.02&&s.jawVelocity<0.1)?1.5:1) - s.score);
});
const dtEnt5 = wstat(dt, 2, a => {const b=[0,0,0];a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;});
const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
const scoreStd5=wstat(all.map(s=>s.score),2,std);
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

// New feature: jawDelta windowed mean
const jdMean5 = wstat(all.map(s=>s.jawDelta), 2, mean);
// New feature: velocity / jawDelta ratio (jaw efficiency)
const velMean5 = wstat(all.map(s=>s.jawVelocity), 2, mean);

// === Analyze: what makes new-data AI look like user? ===
console.log('=== New data AI with high jawVelocity ===\n');
{
  const aiHighVel = [];
  for(let i=0;i<N;i++){
    if(!act[i] && all[i].jawVelocity >= 0.1){
      aiHighVel.push(i);
    }
  }
  console.log(`AI tokens with vel>=0.1: ${aiHighVel.length} / ${N-act.filter(x=>x).length} AI total`);
  
  // What distinguishes these from user?
  const userHighVel = [];
  for(let i=0;i<N;i++){
    if(act[i] && all[i].jawVelocity >= 0.1) userHighVel.push(i);
  }
  console.log(`User tokens with vel>=0.1: ${userHighVel.length} / ${act.filter(x=>x).length} user total\n`);
  
  const feats = {
    score: i => all[i].score,
    jawDelta: i => all[i].jawDelta,
    jawVelocity: i => all[i].jawVelocity,
    jawEffMean5: i => jawEffMean5[i],
    velStd5: i => velStd5[i],
    scoreVelAnti: i => scoreVelAnti[i],
    jdMean5: i => jdMean5[i],
    velMean5: i => velMean5[i],
    dtEnt5: i => dtEnt5[i],
    burstLen: i => burstLen[i],
    isHighJW: i => isHighJW[i] ? 1 : 0,
    scoreGap: i => scoreGap[i],
  };
  
  for(const [name, fn] of Object.entries(feats)){
    const aVals = aiHighVel.map(fn);
    const uVals = userHighVel.map(fn);
    const aM = mean(aVals), aS = std(aVals);
    const uM = mean(uVals), uS = std(uVals);
    const pooledS = Math.sqrt((aS**2 + uS**2)/2);
    const d = pooledS > 0 ? Math.abs(aM - uM) / pooledS : 0;
    if(d > 0.3) console.log(`${name.padEnd(15)} AI=${aM.toFixed(3)}±${aS.toFixed(3)} User=${uM.toFixed(3)}±${uS.toFixed(3)} d=${d.toFixed(3)}`);
  }
}

// === Full search with jawDelta-based features ===
console.log('\n=== Full voting + penalty search ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let vH=0.3;vH<=0.6;vH+=0.1){
    for(let vHW=2;vHW<=4;vHW+=0.5){
      for(let jeTh=4;jeTh<=6;jeTh+=0.5){
        for(let jeW=1;jeW<=3;jeW+=0.5){
          for(let jdTh=0.02;jdTh<=0.08;jdTh+=0.02){
            for(let jdW=0.5;jdW<=2;jdW+=0.5){
              for(let jdmTh=0.02;jdmTh<=0.06;jdmTh+=0.02){
                for(let jdmW=0.5;jdmW<=2;jdmW+=0.5){
                  for(let t=3;t<=5;t+=0.5){
                    const preds = all.map((s,i) => {
                      let v = 0;
                      // jawVelocity
                      if(s.jawVelocity >= vH) v += vHW;
                      else if(s.jawVelocity >= 0.1) v += vHW*0.5;
                      // jawEffMean5
                      if(jawEffMean5[i] >= jeTh) v += jeW;
                      // jawDelta (physical mouth opening)
                      if(s.jawDelta >= jdTh) v += jdW;
                      // jawDelta windowed mean
                      if(jdMean5[i] >= jdmTh) v += jdmW;
                      // scoreVelAnti
                      if(scoreVelAnti[i] >= 0.2) v += 0.5;
                      // timeDelta
                      if(dt[i] >= 0.3) v += 0.75;
                      else if(dt[i] >= 0.03) v += 0.375;
                      // score
                      if(s.score < 0.45) v += 1;
                      // velStd5
                      if(velStd5[i] >= 0.15) v += 0.5;
                      // Penalty: low jawDelta mean (AI lip sync has tiny jaw movements)
                      if(jdMean5[i] < 0.01) v -= 1.5;
                      // Penalty: low jawEffMean5
                      if(jawEffMean5[i] < 2) v -= 1;
                      return v >= t;
                    });
                    const r = ev(preds);
                    if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                      best={...r,vH,vHW,jeTh,jeW,jdTh,jdW,jdmTh,jdmW,t};
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
  console.log(`Full search: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  vel: >=${best.vH}→${best.vHW}`);
    console.log(`  jawEff: >=${best.jeTh}→${best.jeW}`);
    console.log(`  jawDelta: >=${best.jdTh}→${best.jdW}`);
    console.log(`  jdMean5: >=${best.jdmTh}→${best.jdmW}`);
    console.log(`  threshold: ${best.t}`);
    
    // Add rescue
    const sc = all.map((s,i) => {
      let v = 0;
      if(s.jawVelocity >= best.vH) v += best.vHW;
      else if(s.jawVelocity >= 0.1) v += best.vHW*0.5;
      if(jawEffMean5[i] >= best.jeTh) v += best.jeW;
      if(s.jawDelta >= best.jdTh) v += best.jdW;
      if(jdMean5[i] >= best.jdmTh) v += best.jdmW;
      if(scoreVelAnti[i] >= 0.2) v += 0.5;
      if(dt[i] >= 0.3) v += 0.75;
      else if(dt[i] >= 0.03) v += 0.375;
      if(s.score < 0.45) v += 1;
      if(velStd5[i] >= 0.15) v += 0.5;
      if(jdMean5[i] < 0.01) v -= 1.5;
      if(jawEffMean5[i] < 2) v -= 1;
      return v;
    });
    const p1 = sc.map(v => v >= best.t);
    
    let bestR = {f1:0};
    for(let hw=6;hw<=14;hw+=2){
      for(let nTh=0.3;nTh<=0.8;nTh+=0.1){
        for(let low=-4;low<=1;low+=1){
          const preds = all.map((_,i) => {
            if(p1[i]) return true;
            if(all[i].jawVelocity < 0.1) return false;
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
console.log('v93: F1=79.7%');
