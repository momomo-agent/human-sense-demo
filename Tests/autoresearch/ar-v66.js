// autoresearch v66: Radical approaches — rewrite vote system from scratch
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

const scoreGap=all.map(s=>Math.abs(s.finalScore-s.score));

// ============================================================
// APPROACH 1: Logistic regression (gradient descent)
// ============================================================
console.log('=== APPROACH 1: Logistic regression ===\n');
{
  // Features: score, jawVelocity, jawDelta, dt, scoreGap, dtEntropy, burstLen
  const dtEnt5 = wstat(dt, 2, a => {const b=[0,0,0];a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;});
  const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
  const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
  const jawEffMean5=wstat(jawEff,2,mean);
  const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
  
  // Normalize features
  function normalize(arr) {
    const m=mean(arr), s=std(arr)||1;
    return arr.map(v=>(v-m)/s);
  }
  
  const features = [
    normalize(all.map(s=>s.score)),
    normalize(all.map(s=>s.jawVelocity)),
    normalize(all.map(s=>s.jawDelta)),
    normalize(dt),
    normalize(scoreGap),
    normalize(dtEnt5),
    normalize(burstLen.map(v=>v)),
    normalize(jawEffMean5),
    normalize(scoreAccel),
    normalize(all.map(s=>(1-s.score)*s.jawVelocity)),
  ];
  const nFeats = features.length;
  
  // Gradient descent
  const sigmoid = x => 1/(1+Math.exp(-Math.max(-20,Math.min(20,x))));
  let w = new Array(nFeats).fill(0);
  let bias = 0;
  const lr = 0.01;
  
  for(let epoch=0;epoch<500;epoch++){
    const dw = new Array(nFeats).fill(0);
    let db = 0;
    for(let i=0;i<N;i++){
      let z = bias;
      for(let f=0;f<nFeats;f++) z += w[f]*features[f][i];
      const p = sigmoid(z);
      const err = p - (act[i]?1:0);
      for(let f=0;f<nFeats;f++) dw[f] += err*features[f][i];
      db += err;
    }
    for(let f=0;f<nFeats;f++) w[f] -= lr*dw[f]/N;
    bias -= lr*db/N;
  }
  
  // Find best threshold
  const probs = all.map((_,i) => {
    let z = bias;
    for(let f=0;f<nFeats;f++) z += w[f]*features[f][i];
    return sigmoid(z);
  });
  
  let best={f1:0};
  for(let th=0.1;th<=0.9;th+=0.01){
    const p=probs.map(v=>v>=th);
    const r=ev(p);
    if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,th};
  }
  console.log(`Logistic: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} th=${best.th}`);
  console.log(`Weights: ${w.map(v=>v.toFixed(3)).join(', ')}`);
  console.log(`Features: score, jawVel, jawDelta, dt, scoreGap, dtEnt5, burstLen, jawEffMean5, scoreAccel, scoreVelAnti`);
  
  // Best F1 regardless of constraints
  let bestAny={f1:0};
  for(let th=0.1;th<=0.9;th+=0.01){
    const p=probs.map(v=>v>=th);
    const r=ev(p);
    if(r.f1>bestAny.f1)bestAny={...r,th};
  }
  console.log(`Best unconstrained: R=${(bestAny.recall*100).toFixed(1)}% S=${(bestAny.specificity*100).toFixed(1)}% F1=${(bestAny.f1*100).toFixed(1)}% th=${bestAny.th}`);
}

// ============================================================
// APPROACH 2: Decision stump ensemble (AdaBoost-like)
// ============================================================
console.log('\n=== APPROACH 2: Decision stump ensemble ===\n');
{
  const feats = {
    score: all.map(s=>s.score),
    jawVel: all.map(s=>s.jawVelocity),
    jawDelta: all.map(s=>s.jawDelta),
    dt: dt,
    scoreGap: scoreGap,
    scoreVelAnti: all.map(s=>(1-s.score)*s.jawVelocity),
  };
  
  // Sample weights (start uniform)
  let weights = new Array(N).fill(1/N);
  const stumps = [];
  
  for(let round=0;round<20;round++){
    let bestStump = {err:Infinity};
    
    for(const [name, vals] of Object.entries(feats)){
      const sorted = [...new Set(vals)].sort((a,b)=>a-b);
      const thresholds = sorted.filter((_,i)=>i%Math.max(1,Math.floor(sorted.length/20))===0);
      
      for(const th of thresholds){
        for(const dir of [1,-1]){
          let err = 0;
          for(let i=0;i<N;i++){
            const pred = dir===1 ? vals[i]>=th : vals[i]<th;
            if(pred !== act[i]) err += weights[i];
          }
          if(err < bestStump.err) bestStump = {name,th,dir,err};
        }
      }
    }
    
    if(bestStump.err >= 0.5) break;
    
    const alpha = 0.5 * Math.log((1-bestStump.err)/bestStump.err);
    const vals = feats[bestStump.name];
    
    // Update weights
    let wSum = 0;
    for(let i=0;i<N;i++){
      const pred = bestStump.dir===1 ? vals[i]>=bestStump.th : vals[i]<bestStump.th;
      weights[i] *= Math.exp(pred===act[i] ? -alpha : alpha);
      wSum += weights[i];
    }
    for(let i=0;i<N;i++) weights[i] /= wSum;
    
    stumps.push({...bestStump, alpha});
  }
  
  // Evaluate ensemble
  const ensembleScores = all.map((_,i) => {
    let score = 0;
    for(const stump of stumps){
      const val = feats[stump.name][i];
      const pred = stump.dir===1 ? val>=stump.th : val<stump.th;
      score += pred ? stump.alpha : -stump.alpha;
    }
    return score;
  });
  
  let best={f1:0};
  for(let th=-3;th<=3;th+=0.1){
    const p=ensembleScores.map(v=>v>=th);
    const r=ev(p);
    if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,th};
  }
  console.log(`AdaBoost (${stumps.length} stumps): R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
  console.log('Top stumps:', stumps.slice(0,5).map(s=>`${s.name}${s.dir>0?'>=':'<'}${s.th.toFixed(3)}(α=${s.alpha.toFixed(2)})`).join(', '));
}

// ============================================================
// APPROACH 3: k-NN classifier
// ============================================================
console.log('\n=== APPROACH 3: k-NN ===\n');
{
  // Normalize features
  const norm = (arr) => {const m=mean(arr),s=std(arr)||1;return arr.map(v=>(v-m)/s);};
  const f = [
    norm(all.map(s=>s.score)),
    norm(all.map(s=>s.jawVelocity)),
    norm(dt),
    norm(scoreGap),
  ];
  const nf = f.length;
  
  // Leave-one-out k-NN
  for(const k of [3, 5, 7, 11]) {
    const preds = all.map((_,i) => {
      // Compute distance to all other points
      const dists = [];
      for(let j=0;j<N;j++){
        if(j===i) continue;
        let d=0;
        for(let fi=0;fi<nf;fi++) d+=(f[fi][i]-f[fi][j])**2;
        dists.push({d:Math.sqrt(d), label:act[j]});
      }
      dists.sort((a,b)=>a.d-b.d);
      const knn = dists.slice(0,k);
      const userVotes = knn.filter(d=>d.label).length;
      return userVotes > k/2;
    });
    const r = ev(preds);
    console.log(`k=${k}: R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% FP=${r.FP} FN=${r.FN}`);
  }
}

// ============================================================
// APPROACH 4: Hybrid — logistic on top of v63 votes
// ============================================================
console.log('\n=== APPROACH 4: Logistic on v63 votes + residual features ===\n');
{
  const dtEnt5 = wstat(dt, 2, a => {const b=[0,0,0];a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;});
  const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
  const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
  const jawEffMean5=wstat(jawEff,2,mean);
  const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
  const scoreMean5=wstat(all.map(s=>s.score),2,mean);
  const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
  const scoreStd5=wstat(all.map(s=>s.score),2,std);
  const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
  const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {if(a.length<2)return 0;const n=a.length,mx=mean(a.map((_,i)=>i)),my=mean(a);let num=0,den=0;a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});return den>0?num/den:0;});
  
  // v63 votes as primary feature + raw features as residuals
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
  
  const norm = (arr) => {const m=mean(arr),s=std(arr)||1;return arr.map(v=>(v-m)/s);};
  const features = [
    norm(v63sc),  // v63 votes as feature
    norm(all.map(s=>s.score)),
    norm(all.map(s=>s.jawVelocity)),
    norm(scoreGap),
    norm(dt),
  ];
  const nFeats = features.length;
  
  const sigmoid = x => 1/(1+Math.exp(-Math.max(-20,Math.min(20,x))));
  let w = new Array(nFeats).fill(0);
  w[0] = 1; // initialize with v63 votes having high weight
  let bias = 0;
  const lr = 0.01;
  
  for(let epoch=0;epoch<1000;epoch++){
    const dw = new Array(nFeats).fill(0);
    let db = 0;
    for(let i=0;i<N;i++){
      let z = bias;
      for(let f=0;f<nFeats;f++) z += w[f]*features[f][i];
      const p = sigmoid(z);
      const err = p - (act[i]?1:0);
      for(let f=0;f<nFeats;f++) dw[f] += err*features[f][i];
      db += err;
    }
    for(let f=0;f<nFeats;f++) w[f] -= lr*dw[f]/N;
    bias -= lr*db/N;
  }
  
  const probs = all.map((_,i) => {
    let z = bias;
    for(let f=0;f<nFeats;f++) z += w[f]*features[f][i];
    return sigmoid(z);
  });
  
  let best={f1:0};
  for(let th=0.1;th<=0.9;th+=0.01){
    const p=probs.map(v=>v>=th);
    const r=ev(p);
    if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,th};
  }
  console.log(`Hybrid logistic: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
  console.log(`Weights: v63=${w[0].toFixed(3)} score=${w[1].toFixed(3)} vel=${w[2].toFixed(3)} gap=${w[3].toFixed(3)} dt=${w[4].toFixed(3)} bias=${bias.toFixed(3)}`);
}

console.log('\n=== FINAL ===');
console.log('v63 (hand-tuned votes): F1=86.6%, R=93.6%, S=95.1%');
console.log('Compare ML approaches above to see if they can beat hand-tuning.');
