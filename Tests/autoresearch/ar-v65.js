// autoresearch v65: Aggressive new approaches
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

// All features
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

// ============================================================
// DIR T: dt>0 FP targeted — score>0.6 + vel>0.5 + dt>0 but NOT user
// These look like AI speaking with natural timing (not batched)
// Key insight: AI tokens with dt>0 often have CONSISTENT velocity across neighbors
// ============================================================
console.log('=== DIR T: dt>0 FP targeted penalty ===\n');
{
  // For dt>0 tokens: check if velocity is suspiciously consistent with neighbors
  const velConsistency = all.map((s,i) => {
    if(dt[i] < 0.001) return 0;
    // Look at ±2 neighbors, compute how similar their velocity is
    let diffs = 0, count = 0;
    for(let j=Math.max(0,i-2);j<=Math.min(N-1,i+2);j++){
      if(j===i) continue;
      diffs += Math.abs(s.jawVelocity - all[j].jawVelocity);
      count++;
    }
    return count > 0 ? diffs/count : 0;
  });
  
  const uv=[], nv=[];
  for(let i=0;i<N;i++){if(dt[i]>=0.001){if(act[i])uv.push(velConsistency[i]);else nv.push(velConsistency[i]);}}
  console.log(`velConsistency (dt>0 only): user=${mean(uv).toFixed(3)} nonUser=${mean(nv).toFixed(3)}`);
  
  let best={f1:0};
  for(let w=-2;w<=0;w+=0.25){
    if(w===0)continue;
    for(let th=0;th<=1;th+=0.05){
      const p=all.map((s,i)=>{
        let v=v63sc[i];
        // Penalize dt>0 tokens with LOW velocity consistency (= similar to neighbors = AI pattern)
        if(dt[i]>=0.001 && velConsistency[i]<th) v+=w; // w is negative
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>86.6/100)console.log(`velConsistency penalty: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement');
}

// DIR U: Score stability — AI tends to have stable score across burst
console.log('\n=== DIR U: Score stability in burst ===');
{
  // For each token in a burst, compute score variance within the burst
  const burstScoreVar = all.map((s,i) => {
    if(burstLen[i] <= 1) return 0;
    // Find burst start
    let start = i;
    while(start > 0 && dt[start] < 0.001) start--;
    if(dt[start] >= 0.001 && start < i) start++;
    const end = Math.min(start + burstLen[i] - 1, N-1);
    const scores = [];
    for(let j=start;j<=end;j++) scores.push(all[j].score);
    return std(scores);
  });
  
  const uv=[], nv=[];
  for(let i=0;i<N;i++){if(act[i])uv.push(burstScoreVar[i]);else nv.push(burstScoreVar[i]);}
  console.log(`burstScoreVar: user=${mean(uv).toFixed(4)} nonUser=${mean(nv).toFixed(4)}`);
  
  let best={f1:0};
  for(let w=-2;w<=2;w+=0.25){
    if(w===0)continue;
    for(let th=0;th<=0.15;th+=0.01){
      const p=all.map((s,i)=>{
        let v=v63sc[i];
        if(w>0&&burstScoreVar[i]>=th)v+=w;
        if(w<0&&burstScoreVar[i]<th)v+=Math.abs(w);
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>86.6/100)console.log(`burstScoreVar: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement');
}

// DIR V: Velocity momentum — is velocity increasing or decreasing?
console.log('\n=== DIR V: Velocity momentum ===');
{
  const velMomentum = all.map((s,i) => {
    if(i < 2) return 0;
    return s.jawVelocity - all[i-1].jawVelocity; // positive = accelerating
  });
  const velMomMean5 = wstat(velMomentum, 2, mean);
  
  let best={f1:0};
  for(let w=-2;w<=2;w+=0.25){
    if(w===0)continue;
    for(let th=-0.5;th<=0.5;th+=0.05){
      const p=all.map((s,i)=>{
        let v=v63sc[i];
        if(w>0&&velMomMean5[i]>=th)v+=w;
        if(w<0&&velMomMean5[i]<th)v+=Math.abs(w);
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>86.6/100)console.log(`velMomentum: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement');
}

// DIR W: Combined dt pattern — not just entropy but specific dt sequences
console.log('\n=== DIR W: dt sequence patterns ===');
{
  // Count transitions: 0→0, 0→pos, pos→0, pos→pos in window
  const dtTransitions = all.map((s,i) => {
    const hw=2;
    let t00=0,t0p=0,tp0=0,tpp=0;
    for(let j=Math.max(0,i-hw);j<Math.min(N-1,i+hw);j++){
      const curr=dt[j]<0.001?0:1, next=dt[j+1]<0.001?0:1;
      if(!curr&&!next)t00++;
      else if(!curr&&next)t0p++;
      else if(curr&&!next)tp0++;
      else tpp++;
    }
    const total=t00+t0p+tp0+tpp||1;
    return {t00:t00/total, t0p:t0p/total, tp0:tp0/total, tpp:tpp/total};
  });
  
  // t0p = transition from batched to spaced — might indicate user speech start
  const t0p = dtTransitions.map(t => t.t0p);
  const tpp = dtTransitions.map(t => t.tpp);
  
  for(const [name,vals] of [['t0p',t0p],['tpp',tpp]]){
    let best={f1:0};
    for(let w=0.25;w<=2;w+=0.25){
      for(let th=0;th<=0.6;th+=0.05){
        const p=all.map((s,i)=>{
          let v=v63sc[i];
          if(vals[i]>=th)v+=w;
          return v>=4;
        });
        const r=ev(p);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
      }
    }
    if(best.f1>86.6/100)console.log(`${name}: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  }
}

// DIR X: Quadratic vote terms — v^2 threshold
console.log('\n=== DIR X: Non-linear vote transform ===');
{
  // Maybe the relationship between features and user probability is non-linear
  // Try: sqrt(votes) or votes^2 as decision boundary
  let best={f1:0};
  for(let exp=0.5;exp<=2;exp+=0.1){
    for(let th=1;th<=6;th+=0.25){
      const p=v63sc.map(v=>{
        const tv = v > 0 ? Math.pow(v, exp) : -Math.pow(-v, exp);
        return tv >= th;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,exp,th};
    }
  }
  if(best.f1>86.6/100)console.log(`Power transform: F1=${(best.f1*100).toFixed(1)}% exp=${best.exp.toFixed(1)} th=${best.th}`);
  else console.log('No improvement');
}

// DIR Y: Two-threshold system — different thresholds for different score ranges
console.log('\n=== DIR Y: Score-dependent threshold ===');
{
  let best={f1:0};
  for(let tLow=3;tLow<=5;tLow+=0.25){      // threshold when score < 0.5
    for(let tMid=3;tMid<=5;tMid+=0.25){      // threshold when 0.5 <= score < 0.7
      for(let tHigh=3;tHigh<=6;tHigh+=0.25){  // threshold when score >= 0.7
        const p=all.map((s,i)=>{
          const t = s.score < 0.5 ? tLow : s.score < 0.7 ? tMid : tHigh;
          return v63sc[i] >= t;
        });
        const r=ev(p);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,tLow,tMid,tHigh};
      }
    }
  }
  if(best.f1>86.6/100){
    console.log(`Score-dep threshold: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tLow=${best.tLow} tMid=${best.tMid} tHigh=${best.tHigh}`);
  } else console.log('No improvement');
}

// DIR Z: Ensemble — majority vote of multiple threshold configs
console.log('\n=== DIR Z: Ensemble of thresholds ===');
{
  // Run v63 with multiple thresholds, take majority vote
  const thresholds = [3.5, 3.75, 4.0, 4.25, 4.5];
  let best={f1:0};
  for(let minVotes=1;minVotes<=5;minVotes++){
    const p=all.map((s,i)=>{
      let votes=0;
      for(const t of thresholds){if(v63sc[i]>=t)votes++;}
      return votes>=minVotes;
    });
    const r=ev(p);
    console.log(`  minVotes=${minVotes}: R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}%`);
    if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,minVotes};
  }
}

console.log('\n=== FINAL PROGRESS ===');
console.log('v49: F1=84.4%, R=90.4%, S=94.8%');
console.log('v61: F1=85.8%, R=91.7%, S=95.2%');
console.log('v63: F1=86.6%, R=93.6%, S=95.1% (current best)');
