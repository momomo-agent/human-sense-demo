// autoresearch v91: Full re-optimization on merged 2299-token dataset
// Step 1: Diagnose feature distribution shift
// Step 2: Re-tune all weights from scratch
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
  const jawF = Math.max(0.1, 1.0 - jw * s.jawDelta);
  const velF = Math.max(0.1, 1.0 - jvw * s.jawVelocity);
  const nmF = (s.jawDelta < 0.02 && s.jawVelocity < 0.1) ? 1.5 : 1.0;
  const fs2 = s.score * jawF * velF * nmF;
  return Math.abs(fs2 - s.score);
});
const dtEnt5 = wstat(dt, 2, a => {const b=[0,0,0];a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;});
const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
const scoreStd5=wstat(all.map(s=>s.score),2,std);
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {if(a.length<2)return 0;const n=a.length,mx=mean(a.map((_,i)=>i)),my=mean(a);let num=0,den=0;a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});return den>0?num/den:0;});
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

// === Step 1: Feature distribution comparison ===
console.log('=== Feature distributions: User vs AI ===\n');
const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

const features = {
  score: i => all[i].score,
  jawDelta: i => all[i].jawDelta,
  jawVelocity: i => all[i].jawVelocity,
  timeDelta: i => dt[i],
  dtEnt5: i => dtEnt5[i],
  burstLen: i => burstLen[i],
  velStd5: i => velStd5[i],
  scoreStd5: i => scoreStd5[i],
  scoreGap: i => scoreGap[i],
  scoreSlope5: i => scoreSlope5[i],
  scoreVelAnti: i => scoreVelAnti[i],
  scoreAccel: i => scoreAccel[i],
  jawEffMean5: i => jawEffMean5[i],
  isHighJW: i => isHighJW[i] ? 1 : 0,
};

for(const [name, fn] of Object.entries(features)) {
  const uVals = userIdx.map(fn);
  const aVals = aiIdx.map(fn);
  const uM = mean(uVals), uS = std(uVals);
  const aM = mean(aVals), aS = std(aVals);
  const pooledS = Math.sqrt((uS**2 + aS**2)/2);
  const d = pooledS > 0 ? Math.abs(uM - aM) / pooledS : 0;
  console.log(`${name.padEnd(15)} User=${uM.toFixed(3)}±${uS.toFixed(3)} AI=${aM.toFixed(3)}±${aS.toFixed(3)} d=${d.toFixed(3)}`);
}

// === Step 2: Simple threshold search on individual features ===
console.log('\n=== Step 2: Best single-feature classifiers ===\n');
for(const [name, fn] of Object.entries(features)) {
  const vals = (new Array(N)).fill(0).map((_,i) => fn(i));
  let bestF1 = 0, bestTh = 0, bestDir = '';
  // Try both directions
  const sorted = [...new Set(vals)].sort((a,b) => a-b);
  const step = Math.max(1, Math.floor(sorted.length / 50));
  for(let si = 0; si < sorted.length; si += step) {
    const th = sorted[si];
    // user if val >= th
    const p1 = vals.map(v => v >= th);
    const r1 = ev(p1);
    if(r1.f1 > bestF1) { bestF1 = r1.f1; bestTh = th; bestDir = '>='; }
    // user if val < th
    const p2 = vals.map(v => v < th);
    const r2 = ev(p2);
    if(r2.f1 > bestF1) { bestF1 = r2.f1; bestTh = th; bestDir = '<'; }
  }
  if(bestF1 > 0.3) console.log(`${name.padEnd(15)} F1=${(bestF1*100).toFixed(1)}% (${bestDir}${bestTh.toFixed(3)})`);
}

// === Step 3: Re-tune positive vote weights ===
console.log('\n=== Step 3: Re-tune base voting ===\n');
{
  // Start fresh: which features matter most?
  // Try different score thresholds
  let best = {f1:0};
  
  for(let sL=0.3;sL<=0.5;sL+=0.05){
    for(let sM=sL+0.05;sM<=0.6;sM+=0.05){
      for(let sH=sM+0.05;sH<=0.8;sH+=0.05){
        for(let vH=0.3;vH<=0.8;vH+=0.1){
          for(let vM=0.05;vM<=vH;vM+=0.05){
            for(let dtH=0.1;dtH<=0.5;dtH+=0.1){
              // Simple voting
              const preds = all.map((s,i) => {
                let v = 0;
                if(s.score < sL) v += 3;
                else if(s.score < sM) v += 0.75;
                else if(s.score < sH) v += 0.25;
                if(s.jawVelocity >= vH) v += 4;
                else if(s.jawVelocity >= vM) v += 2;
                if(dt[i] >= dtH) v += 1.5;
                else if(dt[i] >= 0.03) v += 0.75;
                return v >= 4;
              });
              const r = ev(preds);
              if(r.recall>=0.7&&r.specificity>=0.7&&r.f1>best.f1){
                best={...r,sL,sM,sH,vH,vM,dtH};
              }
            }
          }
        }
      }
    }
  }
  console.log(`Base voting: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
  console.log(`  score: <${best.sL}→3, <${best.sM}→0.75, <${best.sH}→0.25`);
  console.log(`  vel: >=${best.vH}→4, >=${best.vM}→2`);
  console.log(`  dt: >=${best.dtH}→1.5`);
}

console.log('\n=== PROGRESS ===');
console.log('v90 (merged): F1=68.4% (v88 params on new data)');
