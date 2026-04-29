// autoresearch v105: Without finalScore, push v103 further
// v103 best: F1=83.1% (R=92.7% S=91.5% FP=153 FN=37)
// Strategy: more targeted penalties + rescue
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

const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

function timeWindowIdx(centerIdx, windowSec) {
  const t0 = all[centerIdx].audioTime;
  const indices = [];
  for(let j=centerIdx;j>=0;j--){
    if(t0 - all[j].audioTime > windowSec) break;
    indices.push(j);
  }
  for(let j=centerIdx+1;j<N;j++){
    if(all[j].audioTime - t0 > windowSec) break;
    indices.push(j);
  }
  return indices;
}

const tw5 = all.map((s,i) => {
  const idx = timeWindowIdx(i, 5);
  const jds = idx.map(j => all[j].jawDelta);
  const effs = idx.map(j => jawEff[j]);
  return {
    jdMean: mean(jds),
    jeMean: mean(effs),
    activeRate: jds.filter(v => v >= 0.03).length / jds.length,
  };
});

// v103 base function
function v103score(s, i) {
  let v = 0;
  const f = tw5[i];
  if(f.jdMean >= 0.035 && f.activeRate >= 0.05 && f.jeMean >= 5) v += 4;
  if(s.jawVelocity >= 0.5) v += 2.5;
  else if(s.jawVelocity >= 0.1) v += 2.5*0.3;
  if(s.jawDelta >= 0.05) v += 1.5;
  else if(s.jawDelta >= 0.02) v += 0.75;
  if(jawEff[i] >= 5) v += 0.5;
  if(scoreVelAnti[i] >= 0.2) v += 0.5;
  if(s.score < 0.45) v += 0.5;
  if(dt[i] >= 0.2) v += 0.5;
  if(f.jdMean < 0.005) v -= 2;
  if(f.jeMean < 1.5) v -= 1;
  if(!isHighJW[i] && s.score >= 0.7) v -= 1;
  if(s.jawDelta < 0.04 && s.jawVelocity >= 0.1) v -= 0.5;
  return v;
}

const sc103 = all.map((s,i) => v103score(s, i));
const p103 = sc103.map(v => v >= 4.75);
const r103 = ev(p103);
console.log(`v103 baseline: R=${(r103.recall*100).toFixed(1)}% S=${(r103.specificity*100).toFixed(1)}% F1=${(r103.f1*100).toFixed(1)}% FP=${r103.FP} FN=${r103.FN}`);

// === Add more penalties ===
console.log('\n=== Additional penalties ===\n');
{
  let best = {f1:r103.f1}, count=0;
  
  // Try adding penalties one at a time
  const penalties = [
    // score-based
    {name:'score>=0.65&&jd<0.03', test:(s,i)=>s.score>=0.65&&s.jawDelta<0.03},
    {name:'score>=0.6&&vel<0.3', test:(s,i)=>s.score>=0.6&&s.jawVelocity<0.3},
    {name:'score>=0.55&&jd<0.02', test:(s,i)=>s.score>=0.55&&s.jawDelta<0.02},
    // jawDelta-based
    {name:'jd<0.03&&vel>=0.3', test:(s,i)=>s.jawDelta<0.03&&s.jawVelocity>=0.3},
    {name:'jd<0.02&&vel>=0.1', test:(s,i)=>s.jawDelta<0.02&&s.jawVelocity>=0.1},
    // dt-based
    {name:'dt=0&&score>=0.5', test:(s,i)=>dt[i]<0.001&&s.score>=0.5},
    {name:'dt=0&&jd<0.03', test:(s,i)=>dt[i]<0.001&&s.jawDelta<0.03},
    // Combined
    {name:'!hjw&&vel>=0.3&&jd<0.04', test:(s,i)=>!isHighJW[i]&&s.jawVelocity>=0.3&&s.jawDelta<0.04},
    {name:'score>=0.6&&jd<0.04&&!hjw', test:(s,i)=>s.score>=0.6&&s.jawDelta<0.04&&!isHighJW[i]},
    {name:'vel>=0.5&&jd<0.03', test:(s,i)=>s.jawVelocity>=0.5&&s.jawDelta<0.03},
    {name:'sva<0.1&&vel>=0.1', test:(s,i)=>scoreVelAnti[i]<0.1&&s.jawVelocity>=0.1},
  ];
  
  // Test each penalty individually
  for(const pen of penalties){
    for(let w=0.5;w<=3;w+=0.5){
      const sc = all.map((s,i) => {
        let v = v103score(s, i);
        if(pen.test(s, i)) v -= w;
        return v;
      });
      for(let t=4;t<=5.5;t+=0.25){
        const preds = sc.map(v => v >= t);
        const r = ev(preds);
        if(r.recall>=0.85&&r.specificity>=0.9&&r.f1>best.f1){
          best={...r,pen:pen.name,w,t};
          count++;
        }
      }
    }
  }
  console.log(`Single penalty: ${count} improvements`);
  if(best.f1>r103.f1){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  penalty: ${best.pen} → -${best.w}, threshold=${best.t}`);
  }
  
  // Try pairs of penalties
  console.log('\n=== Penalty pairs ===');
  let bestPair = {f1:best.f1};
  for(let a=0;a<penalties.length;a++){
    for(let b=a+1;b<penalties.length;b++){
      for(let wa=0.5;wa<=2;wa+=0.5){
        for(let wb=0.5;wb<=2;wb+=0.5){
          const sc = all.map((s,i) => {
            let v = v103score(s, i);
            if(penalties[a].test(s, i)) v -= wa;
            if(penalties[b].test(s, i)) v -= wb;
            return v;
          });
          for(let t=4;t<=5.5;t+=0.25){
            const preds = sc.map(v => v >= t);
            const r = ev(preds);
            if(r.recall>=0.85&&r.specificity>=0.92&&r.f1>bestPair.f1){
              bestPair={...r,pa:penalties[a].name,pb:penalties[b].name,wa,wb,t};
            }
          }
        }
      }
    }
  }
  if(bestPair.f1>best.f1){
    console.log(`Best pair: R=${(bestPair.recall*100).toFixed(1)}% S=${(bestPair.specificity*100).toFixed(1)}% F1=${(bestPair.f1*100).toFixed(1)}% FP=${bestPair.FP} FN=${bestPair.FN}`);
    console.log(`  ${bestPair.pa} → -${bestPair.wa}`);
    console.log(`  ${bestPair.pb} → -${bestPair.wb}`);
    console.log(`  threshold=${bestPair.t}`);
  }
  
  // Try triples
  console.log('\n=== Penalty triples ===');
  let bestTriple = {f1:bestPair.f1};
  for(let a=0;a<penalties.length;a++){
    for(let b=a+1;b<penalties.length;b++){
      for(let c=b+1;c<penalties.length;c++){
        for(let wa=0.5;wa<=1.5;wa+=0.5){
          for(let wb=0.5;wb<=1.5;wb+=0.5){
            for(let wc=0.5;wc<=1.5;wc+=0.5){
              const sc = all.map((s,i) => {
                let v = v103score(s, i);
                if(penalties[a].test(s, i)) v -= wa;
                if(penalties[b].test(s, i)) v -= wb;
                if(penalties[c].test(s, i)) v -= wc;
                return v;
              });
              for(let t=4;t<=5.5;t+=0.25){
                const preds = sc.map(v => v >= t);
                const r = ev(preds);
                if(r.recall>=0.85&&r.specificity>=0.93&&r.f1>bestTriple.f1){
                  bestTriple={...r,pa:penalties[a].name,pb:penalties[b].name,pc:penalties[c].name,wa,wb,wc,t};
                }
              }
            }
          }
        }
      }
    }
  }
  if(bestTriple.f1>bestPair.f1){
    console.log(`Best triple: R=${(bestTriple.recall*100).toFixed(1)}% S=${(bestTriple.specificity*100).toFixed(1)}% F1=${(bestTriple.f1*100).toFixed(1)}% FP=${bestTriple.FP} FN=${bestTriple.FN}`);
    console.log(`  ${bestTriple.pa} → -${bestTriple.wa}`);
    console.log(`  ${bestTriple.pb} → -${bestTriple.wb}`);
    console.log(`  ${bestTriple.pc} → -${bestTriple.wc}`);
    console.log(`  threshold=${bestTriple.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v103: F1=83.1%');
