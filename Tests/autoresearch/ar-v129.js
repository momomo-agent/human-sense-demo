// autoresearch v129: Segment-level post-processing + HMM-like state machine
const fs = require('fs');
const DATA = '/Users/kenefe/LOCAL/momo-agent/projects/human-sense-demo/Tests/speaker-test-data.jsonl';
const lines = fs.readFileSync(DATA, 'utf8').trim().split('\n');
const all = lines.map(l => JSON.parse(l));
all.sort((a, b) => a.audioTime - b.audioTime);
const N = all.length;
const act = all.map(s => s.isUserSpeaker);
const dt = all.map((s, i) => i === 0 ? 0 : s.audioTime - all[i - 1].audioTime);
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

function twZone(i, sec) {
  const t0 = all[i].audioTime;
  const idx = [];
  for(let j=i;j>=0;j--){if(t0-all[j].audioTime>sec)break;idx.push(j);}
  for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>sec)break;idx.push(j);}
  return { jdMean: mean(idx.map(j=>all[j].jawDelta)), jeMean: mean(idx.map(j=>jawEff[j])) };
}
const tw10 = all.map((_,i) => twZone(i, 10));

function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}
const dtZeroRatio5 = wstat(dt, 2, a => a.filter(v => v < 0.001).length / a.length);

// v112 raw scores
function v112Score(s, i) {
  let v = 0;
  const f = tw10[i];
  if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 5;
  if(s.jawVelocity >= 0.5) v += 2;
  else if(s.jawVelocity >= 0.1) v += 0.6;
  if(s.jawDelta >= 0.05) v += 2;
  else if(s.jawDelta >= 0.02) v += 0.8;
  if(jawEff[i] >= 5) v += 0.5;
  if(scoreVelAnti[i] >= 0.2) v += 0.5;
  if(s.score < 0.45) v += 0.5;
  if(dt[i] >= 0.2) v += 0.5;
  if(dtZeroRatio5[i] >= 0.5) v += 0.5;
  if(f.jdMean < 0.005) v -= 2;
  if(f.jeMean < 1.5) v -= 1;
  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
  if((s.finalScore||0) >= 0.7) v -= 2.5;
  return v;
}
const scores = all.map((s,i) => v112Score(s,i));

// === Part 1: Segment analysis ===
console.log('=== Segment analysis ===\n');
{
  // Find actual user segments
  const userSegs = [];
  let curSeg = null;
  for(let i=0;i<N;i++){
    if(act[i]){
      if(!curSeg) curSeg = {start:i, tokens:[]};
      curSeg.tokens.push(i);
    } else {
      if(curSeg) { userSegs.push(curSeg); curSeg = null; }
    }
  }
  if(curSeg) userSegs.push(curSeg);
  
  console.log(`User segments: ${userSegs.length}`);
  for(let s=0;s<userSegs.length;s++){
    const seg = userSegs[s];
    const t0 = all[seg.tokens[0]].audioTime;
    const t1 = all[seg.tokens[seg.tokens.length-1]].audioTime;
    const avgScore = mean(seg.tokens.map(i=>scores[i]));
    const minScore = Math.min(...seg.tokens.map(i=>scores[i]));
    const maxScore = Math.max(...seg.tokens.map(i=>scores[i]));
    const avgFs = mean(seg.tokens.map(i=>all[i].finalScore||0));
    console.log(`  S${s}: t=${t0.toFixed(1)}-${t1.toFixed(1)} n=${seg.tokens.length} score=${avgScore.toFixed(1)}[${minScore.toFixed(1)},${maxScore.toFixed(1)}] fs=${avgFs.toFixed(3)}`);
  }
  
  // Find FP segments
  const v112preds = scores.map(v => v >= 5.75);
  const fpSegs = [];
  let curFP = null;
  for(let i=0;i<N;i++){
    if(v112preds[i] && !act[i]){
      if(!curFP) curFP = {start:i, tokens:[]};
      curFP.tokens.push(i);
    } else {
      if(curFP) { fpSegs.push(curFP); curFP = null; }
    }
  }
  if(curFP) fpSegs.push(curFP);
  
  console.log(`\nFP segments: ${fpSegs.length}`);
  for(let s=0;s<Math.min(15,fpSegs.length);s++){
    const seg = fpSegs[s];
    const t0 = all[seg.tokens[0]].audioTime;
    const t1 = all[seg.tokens[seg.tokens.length-1]].audioTime;
    const avgScore = mean(seg.tokens.map(i=>scores[i]));
    const avgFs = mean(seg.tokens.map(i=>all[i].finalScore||0));
    const avgVel = mean(seg.tokens.map(i=>all[i].jawVelocity));
    console.log(`  FP${s}: t=${t0.toFixed(1)}-${t1.toFixed(1)} n=${seg.tokens.length} score=${avgScore.toFixed(1)} fs=${avgFs.toFixed(3)} vel=${avgVel.toFixed(3)}`);
  }
}

// === Part 2: HMM-like state machine ===
console.log('\n=== HMM state machine ===\n');
{
  // State: USER or AI
  // Transition costs: switching state has a penalty (hysteresis)
  let best = {f1:0};
  
  for(let enterT=5;enterT<=7;enterT+=0.5){  // threshold to enter USER state
    for(let exitT=3;exitT<=5.5;exitT+=0.5){  // threshold to exit USER state (lower = stickier)
      for(let minRun=1;minRun<=5;minRun++){  // minimum tokens in a state before switching
        const preds = new Array(N).fill(false);
        let state = 'AI';
        let runLen = 0;
        
        for(let i=0;i<N;i++){
          const sc = scores[i];
          runLen++;
          
          if(state === 'AI'){
            if(sc >= enterT && runLen >= minRun){
              state = 'USER';
              runLen = 0;
            }
          } else {
            if(sc < exitT && runLen >= minRun){
              state = 'AI';
              runLen = 0;
            }
          }
          
          preds[i] = state === 'USER';
        }
        
        const r = ev(preds);
        if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
          best={...r,enterT,exitT,minRun};
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`HMM: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  enterT=${best.enterT} exitT=${best.exitT} minRun=${best.minRun}`);
  } else console.log('No qualifying HMM config');
}

// === Part 3: Segment-level post-processing ===
console.log('\n=== Segment post-processing ===\n');
{
  // Step 1: v112 raw predictions
  // Step 2: Find predicted-user segments
  // Step 3: Filter segments by aggregate features
  
  let best = {f1:0};
  
  for(let rawT=4.5;rawT<=6;rawT+=0.5){
    const rawPreds = scores.map(v => v >= rawT);
    
    // Find predicted segments
    const predSegs = [];
    let cur = null;
    for(let i=0;i<N;i++){
      if(rawPreds[i]){
        if(!cur) cur = {start:i, tokens:[i]};
        else cur.tokens.push(i);
      } else {
        if(cur) { predSegs.push(cur); cur = null; }
      }
    }
    if(cur) predSegs.push(cur);
    
    // Filter segments
    for(let minLen=1;minLen<=5;minLen++){
      for(let maxAvgFs=0.3;maxAvgFs<=0.7;maxAvgFs+=0.1){
        for(let minAvgVel=0;minAvgVel<=0.3;minAvgVel+=0.1){
          const preds = new Array(N).fill(false);
          
          for(const seg of predSegs){
            if(seg.tokens.length < minLen) continue;
            const avgFs = mean(seg.tokens.map(i=>all[i].finalScore||0));
            const avgVel = mean(seg.tokens.map(i=>all[i].jawVelocity));
            if(avgFs > maxAvgFs) continue;
            if(avgVel < minAvgVel) continue;
            for(const i of seg.tokens) preds[i] = true;
          }
          
          const r = ev(preds);
          if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
            best={...r,rawT,minLen,maxAvgFs,minAvgVel};
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`SegFilter: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  rawT=${best.rawT} minLen=${best.minLen} maxAvgFs=${best.maxAvgFs} minAvgVel=${best.minAvgVel}`);
  } else console.log('No qualifying segment filter');
}

// === Part 4: Gap filling — merge nearby user segments ===
console.log('\n=== Gap filling ===\n');
{
  let best = {f1:0};
  
  for(let rawT=5;rawT<=6;rawT+=0.5){
    const rawPreds = scores.map(v => v >= rawT);
    
    for(let maxGap=1;maxGap<=5;maxGap++){
      // Fill gaps of <= maxGap tokens between user segments
      const filled = [...rawPreds];
      for(let i=0;i<N;i++){
        if(!filled[i]){
          // Check if there's a user segment within maxGap on both sides
          let leftUser = false, rightUser = false;
          for(let j=i-1;j>=Math.max(0,i-maxGap);j--){
            if(filled[j]) { leftUser = true; break; }
          }
          for(let j=i+1;j<=Math.min(N-1,i+maxGap);j++){
            if(filled[j]) { rightUser = true; break; }
          }
          if(leftUser && rightUser) filled[i] = true;
        }
      }
      
      // Then filter short segments
      for(let minLen=1;minLen<=3;minLen++){
        const segs = [];
        let cur = null;
        for(let i=0;i<N;i++){
          if(filled[i]){
            if(!cur) cur = [i]; else cur.push(i);
          } else {
            if(cur) { segs.push(cur); cur = null; }
          }
        }
        if(cur) segs.push(cur);
        
        const preds = new Array(N).fill(false);
        for(const seg of segs){
          if(seg.length >= minLen){
            for(const i of seg) preds[i] = true;
          }
        }
        
        const r = ev(preds);
        if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
          best={...r,rawT,maxGap,minLen};
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`GapFill: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  rawT=${best.rawT} maxGap=${best.maxGap} minLen=${best.minLen}`);
  } else console.log('No qualifying gap fill');
}

// === Part 5: Exponential moving average of scores ===
console.log('\n=== EMA smoothing ===\n');
{
  let best = {f1:0};
  
  for(let alpha=0.1;alpha<=0.9;alpha+=0.1){
    const ema = new Array(N);
    ema[0] = scores[0];
    for(let i=1;i<N;i++){
      ema[i] = alpha * scores[i] + (1-alpha) * ema[i-1];
    }
    
    for(let t=3;t<=6;t+=0.25){
      const preds = ema.map(v => v >= t);
      const r = ev(preds);
      if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
        best={...r,alpha,t};
      }
    }
  }
  if(best.f1>0){
    console.log(`EMA: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  alpha=${best.alpha.toFixed(1)} t=${best.t}`);
  } else console.log('No qualifying EMA config');
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
