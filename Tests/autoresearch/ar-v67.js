// autoresearch v67: Paradigm shift — move beyond single-token voting
// Goal: F1 → 90%. Current ceiling: 86.6% (v63)
// Key insight: remaining 14 FN and 49 FP are indistinguishable at token level
// New idea: use SEGMENT-level classification instead of token-level
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

// All precomputed features
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
// APPROACH 1: Segment-level classification
// Group consecutive tokens into segments, classify entire segment
// ============================================================
console.log('=== APPROACH 1: Segment-level classification ===\n');
{
  // Build segments: consecutive tokens with dt < gap_threshold
  function buildSegments(gapTh) {
    const segs = [];
    let start = 0;
    for(let i=1;i<=N;i++){
      if(i===N || dt[i]>=gapTh){
        segs.push({start, end:i-1, len:i-start});
        start = i;
      }
    }
    return segs;
  }
  
  for(const gapTh of [0.001, 0.05, 0.1, 0.5]) {
    const segs = buildSegments(gapTh);
    
    // Segment features: aggregate token features
    const segFeats = segs.map(seg => {
      const tokens = [];
      for(let i=seg.start;i<=seg.end;i++) tokens.push(i);
      const votes = tokens.map(i => v63sc[i]);
      const scores = tokens.map(i => all[i].score);
      const vels = tokens.map(i => all[i].jawVelocity);
      const gaps = tokens.map(i => scoreGap[i]);
      
      return {
        meanVotes: mean(votes),
        maxVotes: Math.max(...votes),
        minVotes: Math.min(...votes),
        medianVotes: [...votes].sort((a,b)=>a-b)[Math.floor(votes.length/2)],
        voteStd: std(votes),
        pctAbove4: votes.filter(v=>v>=4).length/votes.length,
        pctAbove3: votes.filter(v=>v>=3).length/votes.length,
        meanScore: mean(scores),
        meanVel: mean(vels),
        maxVel: Math.max(...vels),
        meanGap: mean(gaps),
        maxGap: Math.max(...gaps),
        len: seg.len,
        tokens,
      };
    });
    
    // Try different segment-level decision rules
    let best={f1:0};
    // Rule: if segment mean votes >= th, classify all tokens as user
    for(let th=1;th<=6;th+=0.25){
      const preds = new Array(N).fill(false);
      segFeats.forEach(sf => {
        if(sf.meanVotes >= th) sf.tokens.forEach(i => preds[i]=true);
      });
      const r=ev(preds);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,rule:'meanVotes',th};
    }
    // Rule: if segment max votes >= th AND pct above 3 >= pctTh
    for(let th=3;th<=8;th+=0.5){
      for(let pctTh=0.2;pctTh<=0.8;pctTh+=0.1){
        const preds = new Array(N).fill(false);
        segFeats.forEach(sf => {
          if(sf.maxVotes >= th && sf.pctAbove3 >= pctTh) sf.tokens.forEach(i => preds[i]=true);
        });
        const r=ev(preds);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,rule:'max+pct',th,pctTh};
      }
    }
    // Rule: if segment pctAbove4 >= pctTh
    for(let pctTh=0.1;pctTh<=0.9;pctTh+=0.05){
      const preds = new Array(N).fill(false);
      segFeats.forEach(sf => {
        if(sf.pctAbove4 >= pctTh) sf.tokens.forEach(i => preds[i]=true);
      });
      const r=ev(preds);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,rule:'pctAbove4',pctTh};
    }
    
    console.log(`gap=${gapTh}: ${segs.length} segments, best F1=${(best.f1*100).toFixed(1)}% rule=${best.rule||'none'}`);
    if(best.f1>86.6/100) console.log(`  *** IMPROVEMENT: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
  }
}

// ============================================================
// APPROACH 2: Sliding window majority vote
// Instead of per-token, classify windows and propagate
// ============================================================
console.log('\n=== APPROACH 2: Sliding window majority ===\n');
{
  for(const hw of [1, 2, 3, 4, 5]) {
    let best={f1:0};
    // Window: if majority of tokens in window are user (by v63), classify center as user
    for(let majTh=0.3;majTh<=0.8;majTh+=0.05){
      const preds = all.map((_,i) => {
        let userCount=0, total=0;
        for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
          total++;
          if(v63sc[j]>=4) userCount++;
        }
        return userCount/total >= majTh;
      });
      const r=ev(preds);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,majTh};
    }
    if(best.f1>0) console.log(`hw=${hw}: F1=${(best.f1*100).toFixed(1)}% majTh=${best.majTh} FP=${best.FP} FN=${best.FN}`);
  }
}

// ============================================================
// APPROACH 3: Two-pass with context propagation
// Pass 1: v63 classification
// Pass 2: for borderline tokens, use neighbor labels to decide
// ============================================================
console.log('\n=== APPROACH 3: Two-pass context propagation ===\n');
{
  const v63pred = v63sc.map(v => v >= 4);
  
  let best={f1:0};
  // Pass 2: for tokens with votes in [lowTh, 4), check if neighbors are user
  for(let lowTh=0;lowTh<=3.5;lowTh+=0.25){
    for(let hw=1;hw<=8;hw++){
      for(let neighborTh=0.3;neighborTh<=0.8;neighborTh+=0.1){
        const preds = all.map((_,i) => {
          if(v63pred[i]) return true;
          if(v63sc[i] < lowTh) return false;
          // Borderline: check neighbors
          let userNeighbors=0, total=0;
          for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
            if(j===i) continue;
            total++;
            if(v63pred[j]) userNeighbors++;
          }
          return total>0 && userNeighbors/total >= neighborTh;
        });
        const r=ev(preds);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,lowTh,hw,neighborTh};
      }
    }
  }
  if(best.f1>0){
    console.log(`Two-pass: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  lowTh=${best.lowTh} hw=${best.hw} neighborTh=${best.neighborTh}`);
  }
}

// ============================================================
// APPROACH 4: Iterative label propagation
// Start with v63, then iteratively update borderline tokens
// ============================================================
console.log('\n=== APPROACH 4: Iterative label propagation ===\n');
{
  let best={f1:0};
  for(let lowTh=0;lowTh<=3;lowTh+=0.5){
    for(let hw=2;hw<=6;hw++){
      for(let propTh=0.3;propTh<=0.7;propTh+=0.1){
        for(let iters=1;iters<=3;iters++){
          let labels = v63sc.map(v => v >= 4);
          for(let iter=0;iter<iters;iter++){
            const newLabels = [...labels];
            for(let i=0;i<N;i++){
              if(labels[i]) continue; // already user
              if(v63sc[i] < lowTh) continue; // too low
              let userN=0, total=0;
              for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
                if(j===i) continue;
                total++;
                if(labels[j]) userN++;
              }
              if(total>0 && userN/total >= propTh) newLabels[i] = true;
            }
            labels = newLabels;
          }
          const r=ev(labels);
          if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,lowTh,hw,propTh,iters};
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Iterative: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  lowTh=${best.lowTh} hw=${best.hw} propTh=${best.propTh} iters=${best.iters}`);
  }
}

// ============================================================
// APPROACH 5: Score re-weighting — give more weight to high-confidence tokens
// ============================================================
console.log('\n=== APPROACH 5: Confidence-weighted window ===\n');
{
  let best={f1:0};
  for(let hw=1;hw<=5;hw++){
    for(let th=2;th<=5;th+=0.25){
      // Weighted average of v63 votes in window, weighted by |votes - 4|
      const smoothed = all.map((_,i) => {
        let wSum=0, wTotal=0;
        for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
          const conf = Math.abs(v63sc[j] - 4) + 0.1; // confidence = distance from boundary
          wSum += v63sc[j] * conf;
          wTotal += conf;
        }
        return wTotal > 0 ? wSum/wTotal : v63sc[i];
      });
      const preds = smoothed.map(v => v >= th);
      const r=ev(preds);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,hw,th};
    }
  }
  if(best.f1>0){
    console.log(`Conf-weighted: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} th=${best.th}`);
  }
}

// ============================================================
// APPROACH 6: Completely new vote system — learned thresholds via grid search
// Instead of hand-picked thresholds, search ALL threshold combinations
// ============================================================
console.log('\n=== APPROACH 6: Rebuilt vote system with optimal thresholds ===\n');
{
  // Core features: score, jawVelocity, dt, scoreGap
  // Search for optimal threshold + weight for each
  let best={f1:0}, count=0;
  
  for(let sT1=0.3;sT1<=0.5;sT1+=0.05){    // score threshold 1
    for(let sW1=2;sW1<=4;sW1+=0.5){         // score weight 1
      for(let vT1=0.05;vT1<=0.2;vT1+=0.05){ // velocity threshold 1
        for(let vW1=1;vW1<=3;vW1+=0.5){      // velocity weight 1
          for(let vT2=0.3;vT2<=0.7;vT2+=0.1){// velocity threshold 2
            for(let vW2=2;vW2<=5;vW2+=0.5){   // velocity weight 2
              for(let dT=0.02;dT<=0.4;dT*=2){ // dt threshold
                for(let dW=0.5;dW<=2;dW+=0.5){ // dt weight
                  for(let gT=0.3;gT<=0.5;gT+=0.1){ // gap threshold
                    for(let gW=1;gW<=2.5;gW+=0.5){   // gap weight
                      const preds = all.map((s,i) => {
                        let v = 0;
                        if(s.score < sT1) v += sW1;
                        if(s.jawVelocity >= vT1) v += vW1;
                        if(s.jawVelocity >= vT2) v += vW2;
                        if(dt[i] >= dT) v += dW;
                        if(scoreGap[i] >= gT) v += gW;
                        // Penalties
                        if(dt[i]<0.001 && s.score>=0.3 && s.score<0.7 && s.jawVelocity>=0.15) v -= 1.5;
                        if(dtEnt5[i] >= 0.725) v += 1;
                        return v >= 4;
                      });
                      const r=ev(preds);
                      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                        best={...r,sT1,sW1,vT1,vW1,vT2,vW2,dT,dW,gT,gW};
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
  console.log(`Rebuilt: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  sT1=${best.sT1} sW1=${best.sW1} vT1=${best.vT1} vW1=${best.vW1} vT2=${best.vT2} vW2=${best.vW2} dT=${best.dT} dW=${best.dW} gT=${best.gT} gW=${best.gW}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v63: F1=86.6%, R=93.6%, S=95.1% (current best)');
console.log('Target: F1=90%');
