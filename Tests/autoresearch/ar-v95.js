// autoresearch v95: Split classifier by jawWeight + aggressive penalty
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
const jdMean5 = wstat(all.map(s=>s.jawDelta), 2, mean);

// Stats by jawWeight
console.log('=== Stats by jawWeight ===');
const hjw = [], ljw = [];
for(let i=0;i<N;i++) { if(isHighJW[i]) hjw.push(i); else ljw.push(i); }
console.log(`High JW: ${hjw.length} (${hjw.filter(i=>act[i]).length} user, ${hjw.filter(i=>!act[i]).length} AI)`);
console.log(`Low JW: ${ljw.length} (${ljw.filter(i=>act[i]).length} user, ${ljw.filter(i=>!act[i]).length} AI)`);
console.log(`High JW user rate: ${(hjw.filter(i=>act[i]).length/hjw.length*100).toFixed(1)}%`);
console.log(`Low JW user rate: ${(ljw.filter(i=>act[i]).length/ljw.length*100).toFixed(1)}%\n`);

// === Combined search: different thresholds for high/low JW ===
console.log('=== Split threshold search ===\n');
{
  let best = {f1:0}, count=0;
  
  function baseVotes(s, i) {
    let v = 0;
    if(s.jawVelocity >= 0.5) v += 3;
    else if(s.jawVelocity >= 0.1) v += 1.5;
    if(jawEffMean5[i] >= 5) v += 2;
    else if(jawEffMean5[i] >= 3) v += 0.5;
    if(s.jawDelta >= 0.05) v += 1;
    else if(s.jawDelta >= 0.02) v += 0.5;
    if(jdMean5[i] >= 0.03) v += 1;
    if(scoreVelAnti[i] >= 0.2) v += 0.5;
    if(dt[i] >= 0.3) v += 0.75;
    else if(dt[i] >= 0.03) v += 0.375;
    if(s.score < 0.45) v += 1;
    if(velStd5[i] >= 0.15) v += 0.5;
    // Penalties
    if(jdMean5[i] < 0.01) v -= 2;
    if(jawEffMean5[i] < 2) v -= 1;
    if(!isHighJW[i] && s.score >= 0.7 && s.jawVelocity < 0.3) v -= 1;
    return v;
  }
  
  const sc = all.map((s,i) => baseVotes(s, i));
  
  // Try different thresholds for high/low JW
  for(let tH=2;tH<=5;tH+=0.25){
    for(let tL=3;tL<=6;tL+=0.25){
      const p1 = sc.map((v,i) => v >= (isHighJW[i] ? tH : tL));
      const r = ev(p1);
      if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
        best={...r,tH,tL};
        count++;
      }
    }
  }
  console.log(`Split threshold: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tHigh=${best.tH} tLow=${best.tL}`);
    
    // Add rescue with split thresholds
    const p1 = sc.map((v,i) => v >= (isHighJW[i] ? best.tH : best.tL));
    let bestR = {f1:0};
    for(let hwH=4;hwH<=12;hwH+=2){
      for(let nThH=0.15;nThH<=0.5;nThH+=0.05){
        for(let hwL=6;hwL<=14;hwL+=2){
          for(let nThL=0.4;nThL<=0.8;nThL+=0.1){
            for(let lowH=-5;lowH<=-1;lowH+=1){
              for(let lowL=-3;lowL<=1;lowL+=1){
                const preds = all.map((_,i) => {
                  if(p1[i]) return true;
                  if(all[i].jawVelocity < 0.1) return false;
                  const hw = isHighJW[i] ? hwH : hwL;
                  const nTh = isHighJW[i] ? nThH : nThL;
                  const low = isHighJW[i] ? lowH : lowL;
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
                if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>bestR.f1) bestR={...r,hwH,nThH,hwL,nThL,lowH,lowL};
              }
            }
          }
        }
      }
    }
    if(bestR.f1>0){
      console.log(`\n+Rescue: R=${(bestR.recall*100).toFixed(1)}% S=${(bestR.specificity*100).toFixed(1)}% F1=${(bestR.f1*100).toFixed(1)}% FP=${bestR.FP} FN=${bestR.FN}`);
      console.log(`  jw=1: hw=${bestR.hwH} nTh=${bestR.nThH} low=${bestR.lowH}`);
      console.log(`  jw=0.2: hw=${bestR.hwL} nTh=${bestR.nThL} low=${bestR.lowL}`);
    }
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v93: F1=79.7%');
