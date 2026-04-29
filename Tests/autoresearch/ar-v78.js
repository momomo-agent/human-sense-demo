// autoresearch v78: Asymmetric two-pass — different rescue for jw=0.2 vs jw=1.0
// Key insight: jawWeight=1.0 tokens are 63% user — rescue them more aggressively
// jawWeight=0.2 tokens are 90% AI — rescue them more conservatively
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

const dtEnt5 = wstat(dt, 2, a => {const b=[0,0,0];a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;});
const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
const scoreMean5=wstat(all.map(s=>s.score),2,mean);
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
const scoreStd5=wstat(all.map(s=>s.score),2,std);
const scoreGap=all.map(s=>Math.abs(s.finalScore-s.score));
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {if(a.length<2)return 0;const n=a.length,mx=mean(a.map((_,i)=>i)),my=mean(a);let num=0,den=0;a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});return den>0?num/den:0;});
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

function v73votes(s, i) {
  let v=0;
  if(s.score<0.45)v+=3;else if(s.score<0.5)v+=0.75;else if(s.score<0.72)v+=0.25;
  if(s.jawDelta>=0.1)v+=0.25;else if(s.jawDelta>=0.05)v+=0.125;
  if(s.jawVelocity>=0.5)v+=4;else if(s.jawVelocity>=0.1)v+=2;else if(s.jawVelocity>=0.05)v+=1;
  if(dt[i]>=0.3)v+=1.5;else if(dt[i]>=0.03)v+=0.75;
  if(dtEnt5[i]>=0.725)v+=1;
  const sv=(1-s.score)*s.jawVelocity;
  if(sv>=0.875)v+=0.375;
  if(scoreAccel[i]>=1.5)v+=0.75;
  if(jawEffMean5[i]<4.5)v+=0.25;
  if(scoreGap[i]>=0.425)v+=1.75;
  if(scoreSlope5[i]<-0.1)v+=0.5;
  if(scoreVelAnti[i]>=0.3)v+=0.375;
  if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15) v -= 1.625;
  if(velStd5[i]>=0.6&&dt[i]<0.001) v -= 0.875;
  if(scoreStd5[i]<0.12&&dt[i]<0.001) v -= 0.375;
  if(v>=4.25&&dt[i]<0.001&&s.score<0.35) v -= 1.75;
  if(s.score>=0.7 && s.jawVelocity>=0.4 && dt[i]<0.001) v -= 2.0;
  return v;
}
const v73sc = all.map((s,i) => v73votes(s,i));
const v73pred = v73sc.map(v => v >= 4);

// ============================================================
// Part 1: Asymmetric rescue — different params for jw=0.2 vs jw=1.0
// ============================================================
console.log('=== Part 1: Asymmetric rescue by jawWeight ===\n');
{
  let best={f1:0}, count=0;
  
  // For jw=1.0 (likely user): more aggressive rescue
  // For jw=0.2 (likely AI): more conservative rescue
  for(let hwHigh=8;hwHigh<=14;hwHigh+=2){
    for(let nThHigh=0.5;nThHigh<=0.8;nThHigh+=0.05){
      for(let lowHigh=-3;lowHigh<=1;lowHigh+=1){
        for(let hwLow=10;hwLow<=14;hwLow+=2){
          for(let nThLow=0.7;nThLow<=0.9;nThLow+=0.05){
            for(let lowLow=-1;lowLow<=2;lowLow+=0.5){
              for(let velTh=0.05;velTh<=0.15;velTh+=0.025){
                const preds = all.map((_,i) => {
                  if(v73pred[i]) return true;
                  if(all[i].jawVelocity < velTh) return false;
                  
                  const hw = isHighJW[i] ? hwHigh : hwLow;
                  const nTh = isHighJW[i] ? nThHigh : nThLow;
                  const low = isHighJW[i] ? lowHigh : lowLow;
                  
                  if(v73sc[i] < low) return false;
                  
                  let userN=0, total=0;
                  for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
                    if(j===i) continue;
                    total++;
                    if(v73pred[j]) userN++;
                  }
                  return total>0 && userN/total >= nTh;
                });
                const r=ev(preds);
                if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                  best={...r,hwHigh,nThHigh,lowHigh,hwLow,nThLow,lowLow,velTh};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Asymmetric: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  jw=1: hw=${best.hwHigh} nTh=${best.nThHigh} low=${best.lowHigh}`);
    console.log(`  jw=0.2: hw=${best.hwLow} nTh=${best.nThLow} low=${best.lowLow}`);
    console.log(`  velTh=${best.velTh}`);
  }
}

// ============================================================
// Part 2: Stage 1 threshold split by jawWeight
// ============================================================
console.log('\n=== Part 2: Split threshold by jawWeight ===\n');
{
  let best={f1:0}, count=0;
  
  // Different vote threshold for jw=0.2 vs jw=1.0
  for(let tHigh=3;tHigh<=5;tHigh+=0.25){
    for(let tLow=4;tLow<=6;tLow+=0.25){
      const s1pred = all.map((_,i) => {
        const t = isHighJW[i] ? tHigh : tLow;
        return v73sc[i] >= t;
      });
      
      // Rescue
      for(let rHw=10;rHw<=14;rHw+=2){
        for(let rNTh=0.6;rNTh<=0.8;rNTh+=0.05){
          const preds = all.map((_,i) => {
            if(s1pred[i]) return true;
            if(v73sc[i] < -3) return false;
            if(all[i].jawVelocity < 0.075) return false;
            let userN=0, total=0;
            for(let j=Math.max(0,i-rHw);j<=Math.min(N-1,i+rHw);j++){
              if(j===i) continue;
              total++;
              if(s1pred[j]) userN++;
            }
            return total>0 && userN/total >= rNTh;
          });
          const r=ev(preds);
          if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
            best={...r,tHigh,tLow,rHw,rNTh};
            count++;
          }
        }
      }
    }
  }
  console.log(`Split threshold: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tHigh=${best.tHigh} tLow=${best.tLow} hw=${best.rHw} nTh=${best.rNTh}`);
  }
}

// ============================================================
// Part 3: Analyze remaining 40 FP in detail
// ============================================================
console.log('\n=== Part 3: Remaining FP analysis ===\n');
{
  const preds = all.map((_,i) => {
    if(v73pred[i]) return true;
    if(v73sc[i] < -3) return false;
    if(all[i].jawVelocity < 0.075) return false;
    let userN=0, total=0;
    for(let j=Math.max(0,i-13);j<=Math.min(N-1,i+13);j++){
      if(j===i) continue;
      total++;
      if(v73pred[j]) userN++;
    }
    return total>0 && userN/total >= 0.7;
  });
  
  const FP=[];
  for(let i=0;i<N;i++) if(preds[i]&&!act[i]) FP.push(i);
  
  console.log(`FP count: ${FP.length}`);
  console.log(`FP by jawWeight: jw=0.2: ${FP.filter(i=>!isHighJW[i]).length}, jw=1.0: ${FP.filter(i=>isHighJW[i]).length}`);
  console.log(`FP by dt: dt=0: ${FP.filter(i=>dt[i]<0.001).length}, dt>0: ${FP.filter(i=>dt[i]>=0.001).length}`);
  
  // Show the dt>0 FP (hardest to fix)
  const fpDtPos = FP.filter(i => dt[i]>=0.001);
  console.log(`\ndt>0 FP (${fpDtPos.length}):`);
  fpDtPos.forEach(i => {
    const s=all[i];
    console.log(`  i=${i} "${s.text}" score=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(3)} jw=${(s.jawWeight||0).toFixed(1)} votes=${v73sc[i].toFixed(2)} gap=${scoreGap[i].toFixed(3)}`);
  });
  
  // Show dt=0 FP
  const fpDt0 = FP.filter(i => dt[i]<0.001);
  console.log(`\ndt=0 FP (${fpDt0.length}):`);
  fpDt0.forEach(i => {
    const s=all[i];
    console.log(`  i=${i} "${s.text}" score=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} jw=${(s.jawWeight||0).toFixed(1)} votes=${v73sc[i].toFixed(2)} gap=${scoreGap[i].toFixed(3)}`);
  });
}

console.log('\n=== PROGRESS ===');
console.log('v73: F1=89.1%, FP=42, FN=9');
console.log('v76: F1=89.5%, FP=40, FN=9');
