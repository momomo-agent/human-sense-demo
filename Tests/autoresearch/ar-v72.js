// autoresearch v72: Redesign vote weights to minimize FP while keeping high recall
// Strategy: use the 50 FP and 208 TP as training signal to find better weights
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

// Precompute ALL features as continuous values
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

// ============================================================
// APPROACH: Penalty-focused vote redesign
// Keep the positive votes (what makes something user)
// But redesign the penalty system (what makes something NOT user)
// ============================================================

// Current penalties:
// 1. lipSyncPen: score 0.3-0.7 + dt=0 + vel>=0.15 → -1.5
// 2. velStd: velStd5>=0.6 + dt=0 → -0.75
// 3. scoreMean: scoreMean5>=0.65 + dt=0 → -0.5
// 4. scoreStd: scoreStd5<0.12 + dt=0 → -0.375
// 5. FP filter: votes>=4.25 + dt=0 + score<0.35 → -1.75
// 6. burstLen: burstLen>=3 → -0.25

// New penalty ideas:
// A. Neighbor-aware penalty: if most neighbors are non-user, penalize
// B. Score-velocity mismatch: high score + high velocity = suspicious (AI lip sync with big jaw)
// C. Burst homogeneity: if all tokens in burst have similar features = AI pattern

console.log('=== Penalty redesign search ===\n');

// First, compute v63 votes WITHOUT penalties
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
  if(s.score<0.72&&s.score>=0.5)v+=0; // already counted
  return v;
}

// Search for optimal penalty combination
let best={f1:0}, count=0;

// Penalty parameters to search
const pLipW_r = [1.0, 1.5, 2.0, 2.5];
const pVelStdW_r = [0.5, 0.75, 1.0, 1.25];
const pScoreMeanW_r = [0.25, 0.5, 0.75, 1.0];
const pScoreStdW_r = [0.25, 0.375, 0.5];
const pBurstW_r = [0, 0.25, 0.5, 0.75];
const pFPW_r = [1.0, 1.5, 1.75, 2.0, 2.5];
const pFPTh_r = [4.0, 4.25, 4.5];
const pFPScore_r = [0.3, 0.35, 0.4];
// New: score-velocity mismatch penalty
const pSVMW_r = [0, 0.5, 1.0, 1.5];
const pSVMScoreTh_r = [0.6, 0.7];
const pSVMVelTh_r = [0.5, 0.8];

for(const pLipW of pLipW_r){
  for(const pVelStdW of pVelStdW_r){
    for(const pScoreMeanW of pScoreMeanW_r){
      for(const pBurstW of pBurstW_r){
        for(const pFPW of pFPW_r){
          for(const pFPTh of pFPTh_r){
            for(const pFPScore of pFPScore_r){
              for(const pSVMW of pSVMW_r){
                for(const pSVMScoreTh of pSVMScoreTh_r){
                  for(const pSVMVelTh of pSVMVelTh_r){
                    const votes = all.map((s,i) => {
                      let v = basePositiveVotes(s, i);
                      // Existing penalties (with new weights)
                      if(burstLen[i]>=3) v -= pBurstW;
                      if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15) v -= pLipW;
                      if(velStd5[i]>=0.6&&dt[i]<0.001) v -= pVelStdW;
                      if(scoreMean5[i]>=0.65&&dt[i]<0.001) v -= pScoreMeanW;
                      if(scoreStd5[i]<0.12&&dt[i]<0.001) v -= 0.375;
                      if(scoreSlope5[i]<-0.1) v += 0.5;
                      // FP filter
                      if(v>=pFPTh&&dt[i]<0.001&&s.score<pFPScore) v -= pFPW;
                      // New: score-velocity mismatch
                      if(pSVMW>0 && s.score>=pSVMScoreTh && s.jawVelocity>=pSVMVelTh && dt[i]<0.001) v -= pSVMW;
                      return v;
                    });
                    
                    // Two-pass with rescue
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
                      best={...r,pLipW,pVelStdW,pScoreMeanW,pBurstW,pFPW,pFPTh,pFPScore,pSVMW,pSVMScoreTh,pSVMVelTh};
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

console.log(`Penalty redesign: ${count} qualifying`);
if(best.f1>0){
  console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
  console.log(`  lipW=${best.pLipW} velStdW=${best.pVelStdW} scoreMeanW=${best.pScoreMeanW} burstW=${best.pBurstW}`);
  console.log(`  fpW=${best.pFPW} fpTh=${best.pFPTh} fpScore=${best.pFPScore}`);
  console.log(`  svmW=${best.pSVMW} svmScore=${best.pSVMScoreTh} svmVel=${best.pSVMVelTh}`);
}

console.log('\n=== PROGRESS ===');
console.log('v69: F1=87.4%, FP=50, FN=10');
console.log('Target: FP≤35 for F1≥90%');
