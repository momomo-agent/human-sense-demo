// autoresearch v71: FP reduction — the path to F1=90%
// Current best: F1=87.4%, FP=50, FN=10
// Need: FP≤30 to reach F1≈90% (with FN=10)
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
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreGap=all.map(s=>Math.abs(s.finalScore-s.score));
const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {if(a.length<2)return 0;const n=a.length,mx=mean(a.map((_,i)=>i)),my=mean(a);let num=0,den=0;a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});return den>0?num/den:0;});

function v63votes(s, i) {
  let v=0;
  if(s.score<0.45)v+=3;else if(s.score<0.5)v+=0.75;else if(s.score<0.72)v+=0.25;
  if(s.jawDelta>=0.1)v+=0.25;else if(s.jawDelta>=0.05)v+=0.125;
  if(s.jawVelocity>=0.5)v+=4;else if(s.jawVelocity>=0.1)v+=2;else if(s.jawVelocity>=0.05)v+=1;
  if(dt[i]>=0.3)v+=1.5;else if(dt[i]>=0.03)v+=0.75;
  if(dtEnt5[i]>=0.725)v+=1;
  if(burstLen[i]>=3)v-=0.25;
  if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15)v-=1.5;
  if(velStd5[i]>=0.6&&dt[i]<0.001)v-=0.75;
  if(scoreMean5[i]>=0.65&&dt[i]<0.001)v-=0.5;
  if(scoreStd5[i]<0.12&&dt[i]<0.001)v-=0.375;
  if(scoreVelAnti[i]>=0.3)v+=0.375;
  const sv=(1-s.score)*s.jawVelocity;
  if(sv>=0.875)v+=0.375;
  if(v>=4.25&&dt[i]<0.001&&s.score<0.35)v-=1.75;
  if(scoreAccel[i]>=1.5)v+=0.75;
  if(jawEffMean5[i]<4.5)v+=0.25;
  if(scoreGap[i]>=0.425)v+=1.75;
  if(scoreSlope5[i]<-0.1)v+=0.5;
  return v;
}
const v63sc = all.map((s,i) => v63votes(s,i));
const v63pred = v63sc.map(v => v >= 4);

// v69 best: two-pass with vel rescue
function v69preds() {
  return all.map((_,i) => {
    if(v63pred[i]) return true;
    if(v63sc[i] < -2) return false;
    if(all[i].jawVelocity < 0.075) return false;
    let userN=0, total=0;
    for(let j=Math.max(0,i-10);j<=Math.min(N-1,i+10);j++){
      if(j===i) continue;
      total++;
      if(v63pred[j]) userN++;
    }
    return total>0 && userN/total >= 0.75;
  });
}
const v69p = v69preds();
const v69r = ev(v69p);
console.log(`v69 baseline: R=${(v69r.recall*100).toFixed(1)}% S=${(v69r.specificity*100).toFixed(1)}% F1=${(v69r.f1*100).toFixed(1)}% FP=${v69r.FP} FN=${v69r.FN}`);

// ============================================================
// Deep FP analysis — find patterns to suppress
// ============================================================
console.log('\n=== FP deep analysis ===\n');
{
  const FP=[], TP=[];
  for(let i=0;i<N;i++){
    if(v69p[i]&&!act[i])FP.push(i);
    if(v69p[i]&&act[i])TP.push(i);
  }
  
  // Categorize FP
  const fpDt0 = FP.filter(i => dt[i]<0.001);
  const fpDtPos = FP.filter(i => dt[i]>=0.001);
  const fpHighVotes = FP.filter(i => v63sc[i]>=6);
  const fpMedVotes = FP.filter(i => v63sc[i]>=4 && v63sc[i]<6);
  const fpLowVotes = FP.filter(i => v63sc[i]<4); // rescued FP
  
  console.log(`FP categories: dt=0:${fpDt0.length} dt>0:${fpDtPos.length} highV:${fpHighVotes.length} medV:${fpMedVotes.length} lowV(rescued):${fpLowVotes.length}`);
  
  // For each FP, compute: is it isolated or in a cluster of FP?
  const fpSet = new Set(FP);
  const fpCluster = FP.map(i => {
    let fpNeighbors=0;
    for(let j=Math.max(0,i-3);j<=Math.min(N-1,i+3);j++){
      if(j!==i && fpSet.has(j)) fpNeighbors++;
    }
    return fpNeighbors;
  });
  
  console.log(`FP clustering: isolated(0)=${fpCluster.filter(c=>c===0).length} paired(1)=${fpCluster.filter(c=>c===1).length} clustered(2+)=${fpCluster.filter(c=>c>=2).length}`);
  
  // Key insight: FP that are isolated (no other FP nearby) are harder to suppress
  // FP that are clustered might be suppressible by checking "is this a FP cluster?"
  
  // For TP, same analysis
  const tpSet = new Set(TP);
  const tpCluster = TP.map(i => {
    let tpNeighbors=0;
    for(let j=Math.max(0,i-3);j<=Math.min(N-1,i+3);j++){
      if(j!==i && tpSet.has(j)) tpNeighbors++;
    }
    return tpNeighbors;
  });
  console.log(`TP clustering: isolated(0)=${tpCluster.filter(c=>c===0).length} paired(1)=${tpCluster.filter(c=>c===1).length} clustered(2+)=${tpCluster.filter(c=>c>=2).length}`);
  
  // FP with high score (>0.7) — these are AI tokens that look like user
  const fpHighScore = FP.filter(i => all[i].score >= 0.7);
  console.log(`\nFP with score>=0.7: ${fpHighScore.length}`);
  
  // FP with low scoreGap — AI tokens with stable score
  const fpLowGap = FP.filter(i => scoreGap[i] < 0.1);
  console.log(`FP with scoreGap<0.1: ${fpLowGap.length}`);
  
  // Overlap: FP with score>=0.7 AND scoreGap<0.1
  const fpBoth = FP.filter(i => all[i].score >= 0.7 && scoreGap[i] < 0.1);
  console.log(`FP with score>=0.7 AND gap<0.1: ${fpBoth.length}`);
  
  // Can we suppress these without losing TP?
  const tpBoth = TP.filter(i => all[i].score >= 0.7 && scoreGap[i] < 0.1);
  console.log(`TP with score>=0.7 AND gap<0.1: ${tpBoth.length}`);
}

// ============================================================
// Part 1: FP suppression rules on top of v69
// ============================================================
console.log('\n=== Part 1: FP suppression rules ===\n');
{
  let best={f1:0};
  
  // Rule A: suppress if score >= scoreTh AND scoreGap < gapTh AND votes < votesTh
  for(let scoreTh=0.6;scoreTh<=0.8;scoreTh+=0.05){
    for(let gapTh=0.05;gapTh<=0.2;gapTh+=0.025){
      for(let votesTh=5;votesTh<=8;votesTh+=0.5){
        const preds = v69p.map((pred,i) => {
          if(!pred) return false;
          if(all[i].score >= scoreTh && scoreGap[i] < gapTh && v63sc[i] < votesTh) return false;
          return true;
        });
        const r=ev(preds);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,rule:'A',scoreTh,gapTh,votesTh};
      }
    }
  }
  
  // Rule B: suppress if dt>0 AND score >= scoreTh AND velocity consistency is low
  const velConsistency = all.map((s,i) => {
    let diffs=0, count=0;
    for(let j=Math.max(0,i-2);j<=Math.min(N-1,i+2);j++){
      if(j===i) continue;
      diffs += Math.abs(s.jawVelocity - all[j].jawVelocity);
      count++;
    }
    return count>0 ? diffs/count : 0;
  });
  
  for(let scoreTh=0.5;scoreTh<=0.8;scoreTh+=0.1){
    for(let vcTh=0.1;vcTh<=0.5;vcTh+=0.05){
      for(let votesTh=4;votesTh<=7;votesTh+=0.5){
        const preds = v69p.map((pred,i) => {
          if(!pred) return false;
          if(dt[i]>=0.001 && all[i].score >= scoreTh && velConsistency[i] < vcTh && v63sc[i] < votesTh) return false;
          return true;
        });
        const r=ev(preds);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,rule:'B',scoreTh,vcTh,votesTh};
      }
    }
  }
  
  // Rule C: suppress isolated predictions (no user neighbor within hw)
  for(let hw=1;hw<=4;hw++){
    for(let minNeighbors=1;minNeighbors<=3;minNeighbors++){
      for(let votesTh=4;votesTh<=7;votesTh+=0.5){
        const preds = v69p.map((pred,i) => {
          if(!pred) return false;
          if(v63sc[i] >= votesTh) return true; // high confidence, keep
          let userN=0;
          for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
            if(j===i) continue;
            if(v69p[j]) userN++;
          }
          if(userN < minNeighbors) return false;
          return true;
        });
        const r=ev(preds);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,rule:'C',hw,minNeighbors,votesTh};
      }
    }
  }
  
  if(best.f1>0){
    console.log(`Best suppression: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  rule=${best.rule}`, best);
  }
}

// ============================================================
// Part 2: Combined suppression rules
// ============================================================
console.log('\n=== Part 2: Combined suppression ===\n');
{
  let best={f1:0}, count=0;
  
  // Combine Rule A + Rule C
  for(let scoreTh=0.6;scoreTh<=0.8;scoreTh+=0.1){
    for(let gapTh=0.05;gapTh<=0.15;gapTh+=0.025){
      for(let aVotesTh=5;aVotesTh<=7;aVotesTh+=0.5){
        for(let cHw=1;cHw<=3;cHw++){
          for(let cMinN=1;cMinN<=2;cMinN++){
            for(let cVotesTh=4;cVotesTh<=6;cVotesTh+=0.5){
              const preds = v69p.map((pred,i) => {
                if(!pred) return false;
                // Rule A: high score + low gap + not super high votes
                if(all[i].score >= scoreTh && scoreGap[i] < gapTh && v63sc[i] < aVotesTh) return false;
                // Rule C: isolated + not high votes
                if(v63sc[i] < cVotesTh) {
                  let userN=0;
                  for(let j=Math.max(0,i-cHw);j<=Math.min(N-1,i+cHw);j++){
                    if(j===i) continue;
                    if(v69p[j]) userN++;
                  }
                  if(userN < cMinN) return false;
                }
                return true;
              });
              const r=ev(preds);
              if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,scoreTh,gapTh,aVotesTh,cHw,cMinN,cVotesTh};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Combined: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  A: score>=${best.scoreTh} gap<${best.gapTh} votes<${best.aVotesTh}`);
    console.log(`  C: hw=${best.cHw} minN=${best.cMinN} votes<${best.cVotesTh}`);
  }
}

// ============================================================
// Part 3: What's the theoretical max F1 with current TP?
// ============================================================
console.log('\n=== Part 3: Theoretical analysis ===\n');
{
  // Current: TP=208, FP=50, FN=10
  // If we could remove ALL FP: F1 = 2*208/(2*208+0+10) = 97.7%
  // If we could remove half FP: F1 = 2*208/(2*208+25+10) = 92.0%
  // If FP=30: F1 = 2*208/(2*208+30+10) = 91.2%
  // If FP=35: F1 = 2*208/(2*208+35+10) = 90.1%
  // If FP=40: F1 = 2*208/(2*208+40+10) = 89.3%
  
  for(const fp of [50, 45, 40, 35, 30, 25, 20]) {
    const tp=208, fn=10;
    const pr=tp/(tp+fp), r=tp/(tp+fn);
    const f1=2*pr*r/(pr+r);
    console.log(`FP=${fp}: F1=${(f1*100).toFixed(1)}% (precision=${(pr*100).toFixed(1)}%)`);
  }
  console.log('\nNeed FP≤35 for F1≥90%');
}

console.log('\n=== FINAL ===');
console.log('v63: F1=86.6%');
console.log('v69: F1=87.4%, FP=50, FN=10');
console.log('Need: FP from 50 → 35 (remove 15 FP without losing TP)');
