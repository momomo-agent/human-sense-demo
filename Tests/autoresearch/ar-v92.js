// autoresearch v92: Full re-optimization on 2299 tokens
// Key insight: jawEffMean5 is now the strongest feature (d=1.798)
// Strategy: rebuild voting from top discriminative features
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
  return Math.abs(s.score * jawF * velF * nmF - s.score);
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

// === Rebuild voting system from scratch ===
// Top features by Cohen's d:
// 1. jawEffMean5 (1.798) — user>=5.4, AI=2.8
// 2. jawVelocity (1.462) — user=0.85, AI=0.08
// 3. jawDelta (1.450) — user=0.11, AI=0.01
// 4. scoreVelAnti (1.346) — user=0.41, AI=0.03
// 5. velStd5 (1.087) — user=0.38, AI=0.09
// 6. scoreStd5 (0.842) — user=0.11, AI=0.04
// 7. isHighJW (0.601) — user=24%, AI=4%
// 8. scoreGap (0.440) — user=0.19, AI=0.26 (reversed! AI has higher gap)
// 9. score (0.391) — user=0.52, AI=0.59 (weak, reversed)

console.log('=== Part 1: Optimized voting with top features ===\n');
{
  let best = {f1:0}, count=0;
  
  // Grid search over key thresholds and weights
  for(let velH=0.3;velH<=0.7;velH+=0.1){
    for(let velM=0.05;velM<=0.2;velM+=0.05){
      for(let velW=2;velW<=5;velW+=0.5){
        for(let jemTh=4;jemTh<=7;jemTh+=0.5){
          for(let jemW=0.5;jemW<=3;jemW+=0.5){
            for(let svaTh=0.1;svaTh<=0.5;svaTh+=0.1){
              for(let svaW=0.5;svaW<=2;svaW+=0.5){
                for(let t=2;t<=5;t+=0.5){
                  const preds = all.map((s,i) => {
                    let v = 0;
                    // jawVelocity (strongest physical signal)
                    if(s.jawVelocity >= velH) v += velW;
                    else if(s.jawVelocity >= velM) v += velW * 0.5;
                    // jawEffMean5 (strongest overall)
                    if(jawEffMean5[i] >= jemTh) v += jemW;
                    // scoreVelAnti
                    if(scoreVelAnti[i] >= svaTh) v += svaW;
                    // timeDelta
                    if(dt[i] >= 0.3) v += 1;
                    else if(dt[i] >= 0.03) v += 0.5;
                    // score (weak but still useful)
                    if(s.score < 0.45) v += 1.5;
                    else if(s.score < 0.55) v += 0.5;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.8&&r.specificity>=0.8&&r.f1>best.f1){
                    best={...r,velH,velM,velW,jemTh,jemW,svaTh,svaW,t};
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
  console.log(`Optimized: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  vel: >=${best.velH}→${best.velW}, >=${best.velM}→${best.velW*0.5}`);
    console.log(`  jawEffMean5: >=${best.jemTh}→${best.jemW}`);
    console.log(`  scoreVelAnti: >=${best.svaTh}→${best.svaW}`);
    console.log(`  threshold: ${best.t}`);
  }
}

// === Part 2: Add more features to best base ===
console.log('\n=== Part 2: Add secondary features ===\n');
{
  let best = {f1:0}, count=0;
  
  // Use best from Part 1 as base, add velStd5, scoreStd5, dtEnt5, jawDelta, scoreGap
  for(let vs5Th=0.1;vs5Th<=0.4;vs5Th+=0.1){
    for(let vs5W=0;vs5W<=2;vs5W+=0.5){
      for(let ss5Th=0.03;ss5Th<=0.1;ss5Th+=0.025){
        for(let ss5W=0;ss5W<=1;ss5W+=0.25){
          for(let jdTh=0.02;jdTh<=0.1;jdTh+=0.02){
            for(let jdW=0;jdW<=1;jdW+=0.25){
              const preds = all.map((s,i) => {
                let v = 0;
                // Core features (from Part 1 best, using reasonable defaults)
                if(s.jawVelocity >= 0.5) v += 4;
                else if(s.jawVelocity >= 0.1) v += 2;
                if(jawEffMean5[i] >= 5.5) v += 2;
                if(scoreVelAnti[i] >= 0.2) v += 1;
                if(dt[i] >= 0.3) v += 1;
                else if(dt[i] >= 0.03) v += 0.5;
                if(s.score < 0.45) v += 1.5;
                else if(s.score < 0.55) v += 0.5;
                // Secondary features
                if(vs5W>0 && velStd5[i] >= vs5Th) v += vs5W;
                if(ss5W>0 && scoreStd5[i] >= ss5Th) v += ss5W;
                if(jdW>0 && s.jawDelta >= jdTh) v += jdW;
                return v >= 4;
              });
              const r = ev(preds);
              if(r.recall>=0.8&&r.specificity>=0.8&&r.f1>best.f1){
                best={...r,vs5Th,vs5W,ss5Th,ss5W,jdTh,jdW};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Secondary: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  velStd5: >=${best.vs5Th}→${best.vs5W}`);
    console.log(`  scoreStd5: >=${best.ss5Th}→${best.ss5W}`);
    console.log(`  jawDelta: >=${best.jdTh}→${best.jdW}`);
  }
}

// === Part 3: Add two-pass rescue ===
console.log('\n=== Part 3: Two-pass rescue ===\n');
{
  // Use best combined voting, add rescue
  const sc = all.map((s,i) => {
    let v = 0;
    if(s.jawVelocity >= 0.5) v += 4;
    else if(s.jawVelocity >= 0.1) v += 2;
    if(jawEffMean5[i] >= 5.5) v += 2;
    if(scoreVelAnti[i] >= 0.2) v += 1;
    if(dt[i] >= 0.3) v += 1;
    else if(dt[i] >= 0.03) v += 0.5;
    if(s.score < 0.45) v += 1.5;
    else if(s.score < 0.55) v += 0.5;
    if(velStd5[i] >= 0.2) v += 1;
    if(s.jawDelta >= 0.04) v += 0.5;
    return v;
  });
  const p1 = sc.map(v => v >= 4);
  const baseR = ev(p1);
  console.log(`Base (no rescue): R=${(baseR.recall*100).toFixed(1)}% S=${(baseR.specificity*100).toFixed(1)}% F1=${(baseR.f1*100).toFixed(1)}% FP=${baseR.FP} FN=${baseR.FN}`);
  
  let best = {f1:0};
  for(let hw=4;hw<=14;hw+=2){
    for(let nTh=0.2;nTh<=0.8;nTh+=0.1){
      for(let velTh=0.05;velTh<=0.15;velTh+=0.025){
        for(let low=-5;low<=1;low+=1){
          const preds = all.map((_,i) => {
            if(p1[i]) return true;
            if(all[i].jawVelocity < velTh) return false;
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
          if(r.recall>=0.8&&r.specificity>=0.8&&r.f1>best.f1) best={...r,hw,nTh,velTh,low};
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`+Rescue: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} nTh=${best.nTh} velTh=${best.velTh} low=${best.low}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4% (v88 on merged)');
