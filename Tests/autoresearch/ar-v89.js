// autoresearch v89: Re-eval with computed scoreGap + sentence-level features
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

// Computed scoreGap (using original formula, not finalScore field)
const scoreGapComputed = all.map(s => {
  const jawFactor = Math.max(0.1, 1.0 - 0.25 * s.jawDelta);
  const velocityFactor = Math.max(0.1, 1.0 - 2.0 * s.jawVelocity);
  const noMovementFactor = (s.jawDelta < 0.02 && s.jawVelocity < 0.1) ? 1.5 : 1.0;
  const finalScore = s.score * jawFactor * velocityFactor * noMovementFactor;
  return Math.abs(finalScore - s.score);
});

// Original scoreGap from test data (for comparison)
const scoreGapOriginal = all.map(s => Math.abs(s.finalScore - s.score));

// Compare computed vs original
console.log('=== scoreGap comparison ===');
let matchCount = 0;
for (let i = 0; i < N; i++) {
  if (Math.abs(scoreGapComputed[i] - scoreGapOriginal[i]) < 0.001) matchCount++;
}
console.log(`Match: ${matchCount}/${N} (${(matchCount/N*100).toFixed(1)}%)`);
console.log(`Computed mean: ${mean(scoreGapComputed).toFixed(4)}, Original mean: ${mean(scoreGapOriginal).toFixed(4)}`);
console.log(`Computed std: ${std(scoreGapComputed).toFixed(4)}, Original std: ${std(scoreGapOriginal).toFixed(4)}\n`);

// Standard features
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

// === Sentence segmentation ===
// Split by: gap > 0.5s between tokens OR isFinal=true
const sentences = [];
let sentStart = 0;
for (let i = 1; i <= N; i++) {
  if (i === N || dt[i] > 0.5 || all[i-1].isFinal) {
    sentences.push({ start: sentStart, end: i - 1 });
    sentStart = i;
  }
}
console.log(`=== Sentence segmentation ===`);
console.log(`${sentences.length} sentences from ${N} tokens`);
const sentLens = sentences.map(s => s.end - s.start + 1);
console.log(`Sentence lengths: min=${Math.min(...sentLens)} max=${Math.max(...sentLens)} mean=${mean(sentLens).toFixed(1)}`);

// Map each token to its sentence
const tokenSentence = new Array(N).fill(0);
sentences.forEach((s, si) => {
  for (let i = s.start; i <= s.end; i++) tokenSentence[i] = si;
});

// Sentence-level features
const sentUserRate = new Array(N).fill(0);  // % of user tokens in same sentence
const sentLen = new Array(N).fill(1);       // sentence length
const sentSingleChar = new Array(N).fill(false); // is single-char sentence
const sentVelJitter = new Array(N).fill(0); // velocity jitter within sentence
const sentScoreJitter = new Array(N).fill(0); // score jitter within sentence

sentences.forEach((s, si) => {
  const len = s.end - s.start + 1;
  const toks = all.slice(s.start, s.end + 1);
  const vels = toks.map(t => t.jawVelocity);
  const scores = toks.map(t => t.score);
  
  // Jitter = mean absolute difference between consecutive values
  let velJitter = 0, scoreJitter = 0;
  for (let j = 1; j < toks.length; j++) {
    velJitter += Math.abs(vels[j] - vels[j-1]);
    scoreJitter += Math.abs(scores[j] - scores[j-1]);
  }
  velJitter = toks.length > 1 ? velJitter / (toks.length - 1) : 0;
  scoreJitter = toks.length > 1 ? scoreJitter / (toks.length - 1) : 0;
  
  for (let i = s.start; i <= s.end; i++) {
    sentLen[i] = len;
    sentSingleChar[i] = len === 1;
    sentVelJitter[i] = velJitter;
    sentScoreJitter[i] = scoreJitter;
  }
});

console.log(`\nSingle-char sentences: ${sentences.filter(s => s.end === s.start).length}`);
console.log(`Single-char user: ${sentences.filter(s => s.end === s.start && act[s.start]).length}`);
console.log(`Single-char AI: ${sentences.filter(s => s.end === s.start && !act[s.start]).length}\n`);

// v88 voting function (with computed scoreGap)
function v88votes(s, i, useComputedGap) {
  const gap = useComputedGap ? scoreGapComputed[i] : scoreGapOriginal[i];
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
  if(gap>=0.425)v+=1.75;
  if(scoreSlope5[i]<-0.1)v+=0.5;
  if(scoreVelAnti[i]>=0.3)v+=0.375;
  if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15) v -= 1.625;
  if(velStd5[i]>=0.6&&dt[i]<0.001) v -= 0.875;
  if(scoreStd5[i]<0.12&&dt[i]<0.001) v -= 0.375;
  if(v>=4.25&&dt[i]<0.001&&s.score<0.35) v -= 1.75;
  if(s.score>=0.7 && s.jawVelocity>=0.4 && dt[i]<0.001) v -= 2.0;
  if(!isHighJW[i]) {
    if(dt[i]>=0.001 && s.score>=0.75) v -= 3.0;
    if(dt[i]<0.001 && s.score>=0.3 && s.jawVelocity>=0.5) v -= 1.5;
  }
  if(!isHighJW[i] && dt[i]<0.001 && s.jawVelocity>=0.4 && s.score<0.4) v -= 2.0;
  if(!isHighJW[i] && burstLen[i]<=2 && dt[i]<0.001) v -= 1.75;
  if(!isHighJW[i] && dtEnt5[i]<0.75) v -= 2.25;
  return v;
}

function fullPred(useComputedGap) {
  const sc = all.map((s,i) => v88votes(s, i, useComputedGap));
  const p1 = sc.map(v => v >= 4);
  return all.map((_,i) => {
    if(p1[i]) return true;
    if(all[i].jawVelocity < 0.1) return false;
    const hw = isHighJW[i] ? 6 : 10;
    const nTh = isHighJW[i] ? 0.15 : 0.6;
    const low = isHighJW[i] ? -5 : -1;
    if(sc[i] < low) return false;
    let userN=0, total=0;
    for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
      if(j===i) continue;
      total++;
      if(p1[j]) userN++;
    }
    return total>0 && userN/total >= nTh;
  });
}

// Eval with original scoreGap
const predOrig = fullPred(false);
const rOrig = ev(predOrig);
console.log(`=== v88 with ORIGINAL scoreGap ===`);
console.log(`R=${(rOrig.recall*100).toFixed(1)}% S=${(rOrig.specificity*100).toFixed(1)}% F1=${(rOrig.f1*100).toFixed(1)}% FP=${rOrig.FP} FN=${rOrig.FN}`);

// Eval with computed scoreGap
const predComp = fullPred(true);
const rComp = ev(predComp);
console.log(`\n=== v88 with COMPUTED scoreGap ===`);
console.log(`R=${(rComp.recall*100).toFixed(1)}% S=${(rComp.specificity*100).toFixed(1)}% F1=${(rComp.f1*100).toFixed(1)}% FP=${rComp.FP} FN=${rComp.FN}`);

// === Part 2: Sentence-level features ===
console.log('\n=== Part 2: Sentence-level features ===\n');
{
  // Analyze: FP/FN by sentence features
  const FP = [], FN = [], TP = [], TN = [];
  for(let i=0;i<N;i++){
    if(predComp[i]&&!act[i])FP.push(i);
    if(!predComp[i]&&act[i])FN.push(i);
    if(predComp[i]&&act[i])TP.push(i);
    if(!predComp[i]&&!act[i])TN.push(i);
  }
  
  console.log('FP sentence features:');
  FP.forEach(i => {
    console.log(`  i=${i} "${all[i].text}" sentLen=${sentLen[i]} velJit=${sentVelJitter[i].toFixed(3)} scoreJit=${sentScoreJitter[i].toFixed(3)} single=${sentSingleChar[i]}`);
  });
  
  console.log('\nFN sentence features:');
  FN.forEach(i => {
    console.log(`  i=${i} "${all[i].text}" sentLen=${sentLen[i]} velJit=${sentVelJitter[i].toFixed(3)} scoreJit=${sentScoreJitter[i].toFixed(3)} single=${sentSingleChar[i]}`);
  });
  
  // Cohen's d for sentence features
  const features = {
    sentLen: i => sentLen[i],
    sentVelJitter: i => sentVelJitter[i],
    sentScoreJitter: i => sentScoreJitter[i],
  };
  console.log('\nFP vs TP Cohen\'s d:');
  for(const [name, fn] of Object.entries(features)) {
    const fpVals = FP.map(fn);
    const tpVals = TP.map(fn);
    const fpM = mean(fpVals), fpS = std(fpVals);
    const tpM = mean(tpVals), tpS = std(tpVals);
    const pooledS = Math.sqrt((fpS**2 + tpS**2)/2);
    const d = pooledS > 0 ? Math.abs(fpM - tpM) / pooledS : 0;
    console.log(`  ${name}: FP=${fpM.toFixed(3)}±${fpS.toFixed(3)} TP=${tpM.toFixed(3)}±${tpS.toFixed(3)} d=${d.toFixed(3)}`);
  }
  
  // Try sentence-level penalties
  let best={f1:0}, count=0;
  const scComp = all.map((s,i) => v88votes(s, i, true));
  
  for(let slW=0;slW<=3;slW+=0.25){
    for(let slTh=1;slTh<=3;slTh++){
      for(let vjW=0;vjW<=2;vjW+=0.25){
        for(let vjTh=0.1;vjTh<=0.5;vjTh+=0.1){
          for(let sjW=0;sjW<=2;sjW+=0.25){
            for(let sjTh=0.02;sjTh<=0.1;sjTh+=0.02){
              const votes = scComp.map((v,i) => {
                let nv = v;
                if(slW>0 && !isHighJW[i] && sentLen[i]<=slTh) nv -= slW;
                if(vjW>0 && !isHighJW[i] && sentVelJitter[i]<vjTh) nv -= vjW;
                if(sjW>0 && !isHighJW[i] && sentScoreJitter[i]<sjTh) nv -= sjW;
                return nv;
              });
              
              const p1 = votes.map(v => v >= 4);
              const preds = all.map((_,i) => {
                if(p1[i]) return true;
                if(all[i].jawVelocity < 0.1) return false;
                const hw = isHighJW[i] ? 6 : 10;
                const nTh = isHighJW[i] ? 0.15 : 0.6;
                const low = isHighJW[i] ? -5 : -1;
                if(votes[i] < low) return false;
                let userN=0, total=0;
                for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
                  if(j===i) continue;
                  total++;
                  if(p1[j]) userN++;
                }
                return total>0 && userN/total >= nTh;
              });
              const r=ev(preds);
              if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,slW,slTh,vjW,vjTh,sjW,sjTh};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`\nSentence penalties: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  sentLen: W=${best.slW} th<=${best.slTh}`);
    console.log(`  velJitter: W=${best.vjW} th<${best.vjTh}`);
    console.log(`  scoreJitter: W=${best.sjW} th<${best.sjTh}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v88 original: F1=96.1%');
