// autoresearch v101: Analyze time interleaving + try different approach
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
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}

// === Analyze interleaving ===
console.log('=== Time interleaving analysis ===\n');
{
  // Find speaker change points
  let changes = 0;
  for(let i=1;i<N;i++){
    if(act[i] !== act[i-1]) changes++;
  }
  console.log(`Speaker changes: ${changes} in ${N} tokens`);
  console.log(`Average run length: ${(N/changes).toFixed(1)} tokens`);
  
  // Run length distribution
  const runs = [];
  let runStart = 0;
  for(let i=1;i<=N;i++){
    if(i===N || act[i] !== act[i-1]){
      runs.push({
        speaker: act[runStart] ? 'user' : 'AI',
        len: i - runStart,
        startTime: all[runStart].audioTime,
        endTime: all[i-1].audioTime,
        duration: all[i-1].audioTime - all[runStart].audioTime,
      });
      runStart = i;
    }
  }
  
  const userRuns = runs.filter(r => r.speaker === 'user');
  const aiRuns = runs.filter(r => r.speaker === 'AI');
  console.log(`\nUser runs: ${userRuns.length}, mean len=${mean(userRuns.map(r=>r.len)).toFixed(1)}, mean dur=${mean(userRuns.map(r=>r.duration)).toFixed(2)}s`);
  console.log(`AI runs: ${aiRuns.length}, mean len=${mean(aiRuns.map(r=>r.len)).toFixed(1)}, mean dur=${mean(aiRuns.map(r=>r.duration)).toFixed(2)}s`);
  
  // Short runs (likely noise/interleaving)
  const shortUser = userRuns.filter(r => r.len <= 3);
  const shortAI = aiRuns.filter(r => r.len <= 3);
  console.log(`\nShort runs (<=3 tokens): User=${shortUser.length}/${userRuns.length}, AI=${shortAI.length}/${aiRuns.length}`);
  
  // Show first 30 runs
  console.log('\nFirst 30 runs:');
  runs.slice(0, 30).forEach((r,i) => {
    console.log(`  ${r.speaker.padEnd(4)} len=${String(r.len).padStart(3)} t=${r.startTime.toFixed(1)}-${r.endTime.toFixed(1)} (${r.duration.toFixed(1)}s)`);
  });
}

// === New approach: Run-level smoothing ===
// Instead of classifying individual tokens, classify RUNS
// Then assign all tokens in a run the same label
console.log('\n=== Run-level classification ===\n');
{
  const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
  const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
  
  // Find runs (consecutive tokens with same dt pattern)
  // Actually, let's use a simpler approach: classify each token, then smooth
  
  // Token-level scores
  const tokenScore = all.map((s,i) => {
    let v = 0;
    // Physical jaw movement
    v += s.jawDelta * 10;  // 0-1 range, user ~0.1, AI ~0.01
    v += s.jawVelocity * 1;  // user ~0.85, AI ~0.08
    v += jawEff[i] * 0.1;  // user ~8, AI ~3
    v += scoreVelAnti[i] * 1;  // user ~0.4, AI ~0.03
    // Negative signals
    v -= s.score * 0.5;  // higher score = more likely AI
    return v;
  });
  
  // Smooth with different window sizes
  for(let hw=3;hw<=15;hw+=3){
    const smoothed = tokenScore.map((v,i) => {
      let sum=0, cnt=0;
      for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
        sum += tokenScore[j];
        cnt++;
      }
      return sum/cnt;
    });
    
    // Find best threshold
    let bestF1=0, bestTh=0;
    for(let th=0.2;th<=1.5;th+=0.05){
      const preds = smoothed.map(v => v >= th);
      const r = ev(preds);
      if(r.f1>bestF1){bestF1=r.f1;bestTh=th;}
    }
    const r = ev(smoothed.map(v => v >= bestTh));
    console.log(`hw=${String(hw).padStart(2)}: F1=${(r.f1*100).toFixed(1)}% R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% FP=${r.FP} FN=${r.FN} th=${bestTh.toFixed(2)}`);
  }
}

// === Weighted smoothing with time decay ===
console.log('\n=== Time-decay smoothing ===\n');
{
  const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
  const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
  
  const tokenScore = all.map((s,i) => {
    let v = 0;
    v += s.jawDelta * 10;
    v += s.jawVelocity * 1;
    v += jawEff[i] * 0.1;
    v += scoreVelAnti[i] * 1;
    v -= s.score * 0.5;
    return v;
  });
  
  // Time-decay smoothing: weight by exp(-|dt|/tau)
  for(let tau=1;tau<=10;tau+=1){
    const smoothed = tokenScore.map((v,i) => {
      let sum=0, wSum=0;
      const t0 = all[i].audioTime;
      for(let j=Math.max(0,i-50);j<=Math.min(N-1,i+50);j++){
        const timeDiff = Math.abs(all[j].audioTime - t0);
        if(timeDiff > tau*3) continue;
        const w = Math.exp(-timeDiff/tau);
        sum += tokenScore[j] * w;
        wSum += w;
      }
      return wSum > 0 ? sum/wSum : v;
    });
    
    let bestF1=0, bestTh=0;
    for(let th=0.2;th<=1.5;th+=0.05){
      const preds = smoothed.map(v => v >= th);
      const r = ev(preds);
      if(r.f1>bestF1){bestF1=r.f1;bestTh=th;}
    }
    const r = ev(smoothed.map(v => v >= bestTh));
    console.log(`tau=${String(tau).padStart(2)}s: F1=${(r.f1*100).toFixed(1)}% R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% FP=${r.FP} FN=${r.FN} th=${bestTh.toFixed(2)}`);
  }
}

// === Optimize token score weights + smoothing ===
console.log('\n=== Optimize weights + smoothing ===\n');
{
  const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
  const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
  
  let best = {f1:0};
  
  for(let wJd=5;wJd<=15;wJd+=2.5){
    for(let wVel=0.5;wVel<=2;wVel+=0.5){
      for(let wJe=0.05;wJe<=0.2;wJe+=0.05){
        for(let wSva=0.5;wSva<=2;wSva+=0.5){
          for(let wSc=0;wSc<=1;wSc+=0.25){
            const tokenScore = all.map((s,i) => {
              return s.jawDelta*wJd + s.jawVelocity*wVel + jawEff[i]*wJe + scoreVelAnti[i]*wSva - s.score*wSc;
            });
            
            for(let tau=2;tau<=8;tau+=2){
              const smoothed = tokenScore.map((v,i) => {
                let sum=0, wSum=0;
                const t0 = all[i].audioTime;
                for(let j=Math.max(0,i-80);j<=Math.min(N-1,i+80);j++){
                  const timeDiff = Math.abs(all[j].audioTime - t0);
                  if(timeDiff > tau*3) continue;
                  const w = Math.exp(-timeDiff/tau);
                  sum += tokenScore[j] * w;
                  wSum += w;
                }
                return wSum > 0 ? sum/wSum : v;
              });
              
              for(let th=0.3;th<=1.5;th+=0.1){
                const preds = smoothed.map(v => v >= th);
                const r = ev(preds);
                if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
                  best={...r,wJd,wVel,wJe,wSva,wSc,tau,th};
                }
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  wJd=${best.wJd} wVel=${best.wVel} wJe=${best.wJe} wSva=${best.wSva} wSc=${best.wSc}`);
    console.log(`  tau=${best.tau}s th=${best.th}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v100: F1=81.9%');
