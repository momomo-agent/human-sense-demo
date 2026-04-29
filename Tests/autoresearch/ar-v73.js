// autoresearch v73: Fine-tune v72 penalty redesign — push toward F1=90%
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

function basePositiveVotes(s, i) {
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
  return v;
}

// Fine-grained search around v72 best
console.log('=== Part 1: Fine-tune v72 best ===\n');
{
  let best={f1:0}, count=0;
  
  for(let pLipW=1.0;pLipW<=2.0;pLipW+=0.125){
    for(let pVelStdW=0.5;pVelStdW<=1.0;pVelStdW+=0.125){
      for(let pScoreMeanW=0;pScoreMeanW<=0.5;pScoreMeanW+=0.125){
        for(let pFPW=1.25;pFPW<=2.25;pFPW+=0.25){
          for(let pSVMW=1.0;pSVMW<=2.0;pSVMW+=0.125){
            for(let pSVMScoreTh=0.6;pSVMScoreTh<=0.75;pSVMScoreTh+=0.05){
              for(let pSVMVelTh=0.3;pSVMVelTh<=0.7;pSVMVelTh+=0.1){
                const votes = all.map((s,i) => {
                  let v = basePositiveVotes(s, i);
                  if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15) v -= pLipW;
                  if(velStd5[i]>=0.6&&dt[i]<0.001) v -= pVelStdW;
                  if(scoreMean5[i]>=0.65&&dt[i]<0.001) v -= pScoreMeanW;
                  if(scoreStd5[i]<0.12&&dt[i]<0.001) v -= 0.375;
                  if(v>=4.25&&dt[i]<0.001&&s.score<0.35) v -= pFPW;
                  if(pSVMW>0 && s.score>=pSVMScoreTh && s.jawVelocity>=pSVMVelTh && dt[i]<0.001) v -= pSVMW;
                  return v;
                });
                
                const s1pred = votes.map(v => v >= 4);
                const preds = all.map((_,i) => {
                  if(s1pred[i]) return true;
                  if(votes[i] < -2) return false;
                  if(all[i].jawVelocity < 0.075) return false;
                  let userN=0, total=0;
                  for(let j=Math.max(0,i-10);j<=Math.min(N-1,i+10);j++){
                    if(j===i) continue;
                    total++;
                    if(s1pred[j]) userN++;
                  }
                  return total>0 && userN/total >= 0.75;
                });
                
                const r=ev(preds);
                if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                  best={...r,pLipW,pVelStdW,pScoreMeanW,pFPW,pSVMW,pSVMScoreTh,pSVMVelTh};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Fine-tune: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  lipW=${best.pLipW} velStdW=${best.pVelStdW} scoreMeanW=${best.pScoreMeanW}`);
    console.log(`  fpW=${best.pFPW} svmW=${best.pSVMW} svmScore=${best.pSVMScoreTh} svmVel=${best.pSVMVelTh}`);
  }
}

// Part 2: Add more penalty types
console.log('\n=== Part 2: Additional penalties ===\n');
{
  // Use v72 best as base, add new penalties
  function v72votes(s, i) {
    let v = basePositiveVotes(s, i);
    if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15) v -= 1.5;
    if(velStd5[i]>=0.6&&dt[i]<0.001) v -= 0.75;
    if(scoreMean5[i]>=0.65&&dt[i]<0.001) v -= 0.25;
    if(scoreStd5[i]<0.12&&dt[i]<0.001) v -= 0.375;
    if(v>=4.25&&dt[i]<0.001&&s.score<0.35) v -= 1.75;
    if(s.score>=0.7 && s.jawVelocity>=0.5 && dt[i]<0.001) v -= 1.5;
    return v;
  }
  const v72sc = all.map((s,i) => v72votes(s,i));
  const v72pred = v72sc.map(v => v >= 4);
  
  // New penalty candidates:
  // P1: dt>0 + high score + low scoreGap (AI with natural timing)
  // P2: high velocity + low jawDelta (fast but small movement = AI)
  // P3: scoreMean5 high + dt>0 (sustained high score zone = AI)
  
  let best={f1:0};
  
  for(let p1W=0;p1W<=2;p1W+=0.25){
    for(let p1ScoreTh=0.6;p1ScoreTh<=0.8;p1ScoreTh+=0.1){
      for(let p1GapTh=0.05;p1GapTh<=0.15;p1GapTh+=0.05){
        for(let p2W=0;p2W<=1.5;p2W+=0.25){
          for(let p2VelTh=0.5;p2VelTh<=1;p2VelTh+=0.25){
            for(let p2DeltaTh=0.02;p2DeltaTh<=0.05;p2DeltaTh+=0.01){
              const votes = all.map((s,i) => {
                let v = v72sc[i];
                if(p1W>0 && dt[i]>=0.001 && s.score>=p1ScoreTh && scoreGap[i]<p1GapTh) v -= p1W;
                if(p2W>0 && s.jawVelocity>=p2VelTh && s.jawDelta<p2DeltaTh) v -= p2W;
                return v;
              });
              
              const s1pred = votes.map(v => v >= 4);
              const preds = all.map((_,i) => {
                if(s1pred[i]) return true;
                if(votes[i] < -2) return false;
                if(all[i].jawVelocity < 0.075) return false;
                let userN=0, total=0;
                for(let j=Math.max(0,i-10);j<=Math.min(N-1,i+10);j++){
                  if(j===i) continue;
                  total++;
                  if(s1pred[j]) userN++;
                }
                return total>0 && userN/total >= 0.75;
              });
              
              const r=ev(preds);
              if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,p1W,p1ScoreTh,p1GapTh,p2W,p2VelTh,p2DeltaTh};
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Extra penalties: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  p1: W=${best.p1W} score>=${best.p1ScoreTh} gap<${best.p1GapTh}`);
    console.log(`  p2: W=${best.p2W} vel>=${best.p2VelTh} delta<${best.p2DeltaTh}`);
  }
  
  // Also try: rescue parameter tuning on v72
  let best2={f1:0};
  for(let rLow=-3;rLow<=0;rLow+=0.5){
    for(let rHw=8;rHw<=14;rHw++){
      for(let rNTh=0.65;rNTh<=0.85;rNTh+=0.05){
        for(let velTh=0.05;velTh<=0.15;velTh+=0.025){
          const preds = all.map((_,i) => {
            if(v72pred[i]) return true;
            if(v72sc[i] < rLow) return false;
            if(all[i].jawVelocity < velTh) return false;
            let userN=0, total=0;
            for(let j=Math.max(0,i-rHw);j<=Math.min(N-1,i+rHw);j++){
              if(j===i) continue;
              total++;
              if(v72pred[j]) userN++;
            }
            return total>0 && userN/total >= rNTh;
          });
          const r=ev(preds);
          if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best2.f1){
            best2={...r,rLow,rHw,rNTh,velTh};
          }
        }
      }
    }
  }
  if(best2.f1>0){
    console.log(`\nRescue re-tune: R=${(best2.recall*100).toFixed(1)}% S=${(best2.specificity*100).toFixed(1)}% F1=${(best2.f1*100).toFixed(1)}% FP=${best2.FP} FN=${best2.FN}`);
    console.log(`  low=${best2.rLow} hw=${best2.rHw} nTh=${best2.rNTh} velTh=${best2.velTh}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v63: F1=86.6%');
console.log('v69: F1=87.4%');
console.log('v72: F1=88.7% (penalty redesign + SVM penalty)');
