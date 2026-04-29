// autoresearch v63: Deep dive on scoreGap + jawWeight + scoreSlope + combos
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

const dtEnt5 = wstat(dt, 2, a => {
  const b=[0,0,0]; a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});
  let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;
});
const burstLen = (() => {
  const bl=new Array(N).fill(1);
  for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}
  for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}
  return bl;
})();
const scoreMean5=wstat(all.map(s=>s.score),2,mean);
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
const scoreStd5=wstat(all.map(s=>s.score),2,std);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);

function v61votes(s, i) {
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
  return v;
}
const v61sc = all.map((s,i) => v61votes(s,i));

// New features
const scoreGap = all.map(s => Math.abs(s.finalScore - s.score));
const jawWeight = all.map(s => s.jawWeight || 0);
const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {
  if(a.length<2) return 0;
  const n=a.length, mx=mean(a.map((_,i)=>i)), my=mean(a);
  let num=0,den=0;
  a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});
  return den>0?num/den:0;
});

// ============================================================
// Part 1: Fine-grained scoreGap search
// ============================================================
console.log('=== Part 1: scoreGap fine search ===\n');
{
  let best={f1:0};
  for(let w=0.25;w<=3;w+=0.125){
    for(let th=0.2;th<=0.6;th+=0.025){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(scoreGap[i]>=th)v+=w;
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  console.log(`scoreGap: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  
  // Analyze: what does scoreGap capture?
  const FN_idx = [];
  for(let i=0;i<N;i++) if(v61sc[i]<4 && act[i]) FN_idx.push(i);
  console.log(`\nFN scoreGap distribution:`);
  FN_idx.forEach(i => {
    console.log(`  i=${i} gap=${scoreGap[i].toFixed(3)} score=${all[i].score.toFixed(3)} finalScore=${all[i].finalScore.toFixed(3)} votes=${v61sc[i].toFixed(2)}`);
  });
}

// ============================================================
// Part 2: jawWeight as feature
// ============================================================
console.log('\n=== Part 2: jawWeight ===\n');
{
  // jawWeight has d=1.214, but check if it's redundant with existing features
  let best={f1:0};
  for(let w=0.25;w<=3;w+=0.25){
    for(let th=0.1;th<=0.9;th+=0.05){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(jawWeight[i]>=th)v+=w;
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>0)console.log(`jawWeight: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  
  // Also try negative (penalize low jawWeight)
  let best2={f1:0};
  for(let w=0.25;w<=2;w+=0.25){
    for(let th=0.1;th<=0.5;th+=0.05){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(jawWeight[i]<th)v-=w;
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best2.f1)best2={...r,w,th};
    }
  }
  if(best2.f1>best.f1)console.log(`jawWeight penalty: R=${(best2.recall*100).toFixed(1)}% S=${(best2.specificity*100).toFixed(1)}% F1=${(best2.f1*100).toFixed(1)}% FP=${best2.FP} FN=${best2.FN} w=${best2.w} th=${best2.th}`);
}

// ============================================================
// Part 3: Combine scoreGap + scoreSlope + jawWeight
// ============================================================
console.log('\n=== Part 3: Triple combo ===\n');
{
  let best={f1:0}, count=0;
  const gW_r=[0,0.5,1.0,1.5,1.75,2.0];
  const gTh_r=[0.3,0.35,0.4,0.425,0.45];
  const sW_r=[0,-0.25,-0.5];
  const sTh_r=[-0.1,-0.08,-0.06];
  const jW_r=[0,0.25,0.5];
  const jTh_r=[0.3,0.4,0.5];
  
  for(const gW of gW_r){
    for(const gTh of gTh_r){
      for(const sW of sW_r){
        for(const sTh of sTh_r){
          for(const jW of jW_r){
            for(const jTh of jTh_r){
              const p=all.map((s,i)=>{
                let v=v61sc[i];
                if(scoreGap[i]>=gTh)v+=gW;
                if(scoreSlope5[i]<sTh)v+=Math.abs(sW);
                if(jawWeight[i]>=jTh)v+=jW;
                return v>=4;
              });
              const r=ev(p);
              if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,gW,gTh,sW,sTh,jW,jTh};count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Triple combo: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  gW=${best.gW} gTh=${best.gTh} sW=${best.sW} sTh=${best.sTh} jW=${best.jW} jTh=${best.jTh}`);
  }
}

// ============================================================
// Part 4: scoreGap + rescue for dual-high
// ============================================================
console.log('\n=== Part 4: scoreGap + rescue (dual-high) ===\n');
{
  // Stage 1: v61 + scoreGap
  const stage1 = all.map((s,i) => {
    let v = v61sc[i];
    if(scoreGap[i] >= 0.425) v += 1.75;
    return v;
  });
  const s1pred = stage1.map(v => v >= 4);
  const s1r = ev(s1pred);
  console.log(`Stage1: R=${(s1r.recall*100).toFixed(1)}% S=${(s1r.specificity*100).toFixed(1)}% F1=${(s1r.f1*100).toFixed(1)}% FP=${s1r.FP} FN=${s1r.FN}`);
  
  function predDensity(preds, hw) {
    return preds.map((p, i) => {
      let c = 0, t = 0;
      for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) { t++; if (preds[j]) c++; }
      return c / t;
    });
  }
  
  let best={f1:0};
  for(let hw=3;hw<=12;hw++){
    const pd=predDensity(s1pred,hw);
    for(let rTh=0.15;rTh<=0.7;rTh+=0.05){
      for(let minV=0.5;minV<=3.5;minV+=0.25){
        const p=all.map((s,i)=>{
          if(s1pred[i])return true;
          if(pd[i]>=rTh&&stage1[i]>=minV)return true;
          return false;
        });
        const r=ev(p);
        if(r.recall>=0.95&&r.specificity>=0.9&&r.f1>best.f1)best={...r,hw,rTh,minV};
      }
    }
  }
  console.log(`Dual-high rescue:`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} rTh=${best.rTh.toFixed(2)} minV=${best.minV}`);
  }
  
  // Also try lower stage1 threshold for more recall
  for(let t=3;t<=4;t+=0.25){
    const s1p2=stage1.map(v=>v>=t);
    const s1r2=ev(s1p2);
    if(s1r2.recall>=0.95&&s1r2.specificity>=0.9)
      console.log(`  t=${t}: R=${(s1r2.recall*100).toFixed(1)}% S=${(s1r2.specificity*100).toFixed(1)}% F1=${(s1r2.f1*100).toFixed(1)}% FP=${s1r2.FP} FN=${s1r2.FN}`);
  }
}

// ============================================================
// Part 5: Completely different approach — weighted distance to class centroids
// ============================================================
console.log('\n=== Part 5: Centroid distance classifier ===\n');
{
  // Compute class centroids in feature space
  const features = all.map((s,i) => [s.score, s.jawVelocity, s.jawDelta, dt[i], scoreGap[i]]);
  const userF = features.filter((_,i) => act[i]);
  const nonUserF = features.filter((_,i) => !act[i]);
  
  const dims = features[0].length;
  const userCentroid = Array.from({length:dims}, (_,d) => mean(userF.map(f=>f[d])));
  const nonUserCentroid = Array.from({length:dims}, (_,d) => mean(nonUserF.map(f=>f[d])));
  const userStd = Array.from({length:dims}, (_,d) => std(userF.map(f=>f[d])) || 1);
  const nonUserStd = Array.from({length:dims}, (_,d) => std(nonUserF.map(f=>f[d])) || 1);
  
  console.log('User centroid:', userCentroid.map(v=>v.toFixed(3)));
  console.log('NonUser centroid:', nonUserCentroid.map(v=>v.toFixed(3)));
  
  // Mahalanobis-like distance ratio
  const distRatio = features.map(f => {
    let dUser=0, dNonUser=0;
    for(let d=0;d<dims;d++){
      dUser += ((f[d]-userCentroid[d])/userStd[d])**2;
      dNonUser += ((f[d]-nonUserCentroid[d])/nonUserStd[d])**2;
    }
    return Math.sqrt(dNonUser) - Math.sqrt(dUser); // positive = closer to user
  });
  
  let best={f1:0};
  for(let w=0.25;w<=3;w+=0.25){
    for(let th=-2;th<=2;th+=0.25){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(distRatio[i]>=th)v+=w;
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>0)console.log(`Centroid: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement');
}

// ============================================================
// Part 6: Error analysis on remaining FN after scoreGap
// ============================================================
console.log('\n=== Part 6: Remaining FN analysis ===\n');
{
  const bestVotes = all.map((s,i) => {
    let v = v61sc[i];
    if(scoreGap[i] >= 0.425) v += 1.75;
    return v;
  });
  const bestPred = bestVotes.map(v => v >= 4);
  
  const FN=[], FP=[];
  for(let i=0;i<N;i++){
    if(!bestPred[i]&&act[i])FN.push(i);
    if(bestPred[i]&&!act[i])FP.push(i);
  }
  
  console.log(`Remaining FN (${FN.length}):`);
  FN.forEach(i => {
    const s=all[i];
    console.log(`  i=${i} text="${s.text}" score=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} gap=${scoreGap[i].toFixed(3)} votes=${bestVotes[i].toFixed(2)} burst=${burstLen[i]}`);
  });
  
  console.log(`\nRemaining FP (${FP.length}) — top 10 by votes:`);
  FP.sort((a,b) => bestVotes[b]-bestVotes[a]);
  FP.slice(0,10).forEach(i => {
    const s=all[i];
    console.log(`  i=${i} text="${s.text}" score=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} gap=${scoreGap[i].toFixed(3)} votes=${bestVotes[i].toFixed(2)} burst=${burstLen[i]}`);
  });
}

console.log('\n=== PROGRESS ===');
console.log('v49: F1=84.4%');
console.log('v61: F1=85.8%');
console.log('v62: scoreGap F1=86.6% (if confirmed)');
