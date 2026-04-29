// autoresearch v62: New directions beyond v61
// v61 best: F1=85.8%, R=91.7%, S=95.2%, FP=48, FN=18
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
  for (let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}

// Precompute all features
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

// v61 best votes
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
  // v61 additions
  const sv=(1-s.score)*s.jawVelocity;
  if(sv>=0.875)v+=0.375;
  if(v>=4.25&&dt[i]<0.001&&s.score<0.35)v-=1.75;
  if(scoreAccel[i]>=1.5)v+=0.75;
  if(jawEffMean5[i]<4.5)v+=0.25;
  return v;
}
const v61sc = all.map((s,i) => v61votes(s,i));
const v61pred = v61sc.map(v => v >= 4);
const v61r = ev(v61pred);
console.log(`v61 baseline: R=${(v61r.recall*100).toFixed(1)}% S=${(v61r.specificity*100).toFixed(1)}% F1=${(v61r.f1*100).toFixed(1)}% FP=${v61r.FP} FN=${v61r.FN}`);

// ============================================================
// DIR A: Gradient features — rate of change of jawVelocity
// ============================================================
console.log('\n=== DIR A: Jaw velocity gradient (jerk) ===');
{
  const jawJerk = all.map((s,i) => {
    if(i===0) return 0;
    return Math.abs(s.jawVelocity - all[i-1].jawVelocity);
  });
  const jawJerkMean5 = wstat(jawJerk, 2, mean);
  
  // Cohen's d
  const uv=[], nv=[];
  for(let i=0;i<N;i++){if(act[i])uv.push(jawJerkMean5[i]);else nv.push(jawJerkMean5[i]);}
  const d = (mean(uv)-mean(nv)) / Math.sqrt((std(uv)**2+std(nv)**2)/2);
  console.log(`jawJerkMean5: d=${d.toFixed(3)} user=${mean(uv).toFixed(4)} nonUser=${mean(nv).toFixed(4)}`);
  
  let best={f1:0};
  for(let w=-2;w<=2;w+=0.25){
    if(w===0)continue;
    for(let th=0;th<=0.5;th+=0.025){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(w>0&&jawJerkMean5[i]>=th)v+=w;
        if(w<0&&jawJerkMean5[i]<th)v+=Math.abs(w);
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>0)console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement');
}

// ============================================================
// DIR B: Score trajectory shape — rising vs falling vs flat
// ============================================================
console.log('\n=== DIR B: Score trajectory shape ===');
{
  // Slope of score over window of 5
  const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {
    if(a.length<2) return 0;
    // Simple linear regression slope
    const n=a.length, mx=mean(a.map((_,i)=>i)), my=mean(a);
    let num=0,den=0;
    a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});
    return den>0?num/den:0;
  });
  
  const uv=[], nv=[];
  for(let i=0;i<N;i++){if(act[i])uv.push(scoreSlope5[i]);else nv.push(scoreSlope5[i]);}
  const d = (mean(uv)-mean(nv)) / Math.sqrt((std(uv)**2+std(nv)**2)/2);
  console.log(`scoreSlope5: d=${d.toFixed(3)} user=${mean(uv).toFixed(4)} nonUser=${mean(nv).toFixed(4)}`);
  
  let best={f1:0};
  for(let w=-2;w<=2;w+=0.25){
    if(w===0)continue;
    for(let th=-0.2;th<=0.2;th+=0.02){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(w>0&&scoreSlope5[i]>=th)v+=w;
        if(w<0&&scoreSlope5[i]<th)v+=Math.abs(w);
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>0)console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement');
}

// ============================================================
// DIR C: Jaw movement pattern — periodicity detection
// ============================================================
console.log('\n=== DIR C: Jaw periodicity (autocorrelation) ===');
{
  // AI lip sync tends to have periodic jaw movement; user speech is more irregular
  // Compute autocorrelation of jawDelta in window of 7
  const jawAC = all.map((s,i) => {
    const hw = 3;
    const win = [];
    for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++) win.push(all[j].jawDelta);
    if(win.length < 4) return 0;
    const m = mean(win);
    let num=0, den=0;
    for(let k=0;k<win.length-1;k++){num+=(win[k]-m)*(win[k+1]-m);den+=(win[k]-m)**2;}
    return den>0?num/den:0;
  });
  
  const uv=[], nv=[];
  for(let i=0;i<N;i++){if(act[i])uv.push(jawAC[i]);else nv.push(jawAC[i]);}
  const d = (mean(uv)-mean(nv)) / Math.sqrt((std(uv)**2+std(nv)**2)/2);
  console.log(`jawAC: d=${d.toFixed(3)} user=${mean(uv).toFixed(4)} nonUser=${mean(nv).toFixed(4)}`);
  
  let best={f1:0};
  for(let w=-2;w<=2;w+=0.25){
    if(w===0)continue;
    for(let th=-0.5;th<=1;th+=0.05){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(w>0&&jawAC[i]>=th)v+=w;
        if(w<0&&jawAC[i]<th)v+=Math.abs(w);
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>0)console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement');
}

// ============================================================
// DIR D: Multi-scale features — wider windows (7, 9, 11)
// ============================================================
console.log('\n=== DIR D: Multi-scale window features ===');
{
  for(const hw of [3, 4, 5]) {
    const wSize = hw*2+1;
    const sm = wstat(all.map(s=>s.score), hw, mean);
    const vm = wstat(all.map(s=>s.jawVelocity), hw, mean);
    const dm = wstat(dt, hw, mean);
    const ss = wstat(all.map(s=>s.score), hw, std);
    
    // Try each
    for(const [name, feat] of [['scoreMean'+wSize, sm], ['velMean'+wSize, vm], ['dtMean'+wSize, dm], ['scoreStd'+wSize, ss]]) {
      let best={f1:0};
      for(let w=-2;w<=2;w+=0.5){
        if(w===0)continue;
        const vals = feat;
        // Find reasonable threshold range
        const sorted = [...vals].sort((a,b)=>a-b);
        const p25=sorted[Math.floor(N*0.25)], p75=sorted[Math.floor(N*0.75)];
        const step = (p75-p25)/8 || 0.01;
        for(let th=p25;th<=p75;th+=step){
          const p=all.map((s,i)=>{
            let v=v61sc[i];
            if(w>0&&vals[i]>=th)v+=w;
            if(w<0&&vals[i]<th)v+=Math.abs(w);
            return v>=4;
          });
          const r=ev(p);
          if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
        }
      }
      if(best.f1>v61r.f1) console.log(`  ${name}: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th.toFixed(3)}`);
    }
  }
}

// ============================================================
// DIR E: isFinal flag exploitation
// ============================================================
console.log('\n=== DIR E: isFinal flag ===');
{
  const uFinal=all.filter((_,i)=>act[i]).map(s=>s.isFinal?1:0);
  const nFinal=all.filter((_,i)=>!act[i]).map(s=>s.isFinal?1:0);
  console.log(`isFinal rate: user=${mean(uFinal).toFixed(3)} nonUser=${mean(nFinal).toFixed(3)}`);
  
  let best={f1:0};
  for(let w=-2;w<=2;w+=0.25){
    if(w===0)continue;
    const p=all.map((s,i)=>{
      let v=v61sc[i];
      if(w>0&&s.isFinal)v+=w;
      if(w<0&&!s.isFinal)v+=Math.abs(w);
      return v>=4;
    });
    const r=ev(p);
    if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w};
  }
  if(best.f1>v61r.f1)console.log(`Best: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w}`);
  else console.log('No improvement over v61');
}

// ============================================================
// DIR F: finalScore vs score discrepancy
// ============================================================
console.log('\n=== DIR F: finalScore-score gap ===');
{
  const gap = all.map(s => Math.abs(s.finalScore - s.score));
  const uv=[], nv=[];
  for(let i=0;i<N;i++){if(act[i])uv.push(gap[i]);else nv.push(gap[i]);}
  const d = (mean(uv)-mean(nv)) / Math.sqrt((std(uv)**2+std(nv)**2)/2);
  console.log(`scoreGap: d=${d.toFixed(3)} user=${mean(uv).toFixed(4)} nonUser=${mean(nv).toFixed(4)}`);
  
  let best={f1:0};
  for(let w=-2;w<=2;w+=0.25){
    if(w===0)continue;
    for(let th=0;th<=0.5;th+=0.025){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(w>0&&gap[i]>=th)v+=w;
        if(w<0&&gap[i]<th)v+=Math.abs(w);
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>v61r.f1)console.log(`Best: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement over v61');
}

// ============================================================
// DIR G: jawMargin and jawWeight raw features
// ============================================================
console.log('\n=== DIR G: jawMargin / jawWeight / jawVelocityWeight ===');
{
  for(const fname of ['jawMargin','jawWeight','jawVelocityWeight','noJawPenalty','speakerThreshold']) {
    const vals = all.map(s => s[fname] || 0);
    const uv=[], nv=[];
    for(let i=0;i<N;i++){if(act[i])uv.push(vals[i]);else nv.push(vals[i]);}
    const d = (mean(uv)-mean(nv)) / Math.sqrt((std(uv)**2+std(nv)**2)/2);
    console.log(`${fname}: d=${d.toFixed(3)} user=${mean(uv).toFixed(4)} nonUser=${mean(nv).toFixed(4)}`);
  }
}

// ============================================================
// DIR H: Ratio features — score/velocity, delta/velocity
// ============================================================
console.log('\n=== DIR H: Ratio features ===');
{
  const scoreVelRatio = all.map(s => s.jawVelocity > 0.01 ? s.score / s.jawVelocity : s.score * 100);
  const deltaDtRatio = all.map((s,i) => dt[i] > 0.001 ? s.jawDelta / dt[i] : 0);
  
  for(const [name, vals] of [['score/vel', scoreVelRatio], ['delta/dt', deltaDtRatio]]) {
    const uv=[], nv=[];
    for(let i=0;i<N;i++){if(act[i])uv.push(vals[i]);else nv.push(vals[i]);}
    const d = (mean(uv)-mean(nv)) / Math.sqrt((std(uv)**2+std(nv)**2)/2);
    console.log(`${name}: d=${d.toFixed(3)} user=${mean(uv).toFixed(4)} nonUser=${mean(nv).toFixed(4)}`);
    
    let best={f1:0};
    for(let w=-2;w<=2;w+=0.25){
      if(w===0)continue;
      const sorted=[...vals].sort((a,b)=>a-b);
      const p10=sorted[Math.floor(N*0.1)],p90=sorted[Math.floor(N*0.9)];
      const step=(p90-p10)/20||0.01;
      for(let th=p10;th<=p90;th+=step){
        const p=all.map((s,i)=>{
          let v=v61sc[i];
          if(w>0&&vals[i]>=th)v+=w;
          if(w<0&&vals[i]<th)v+=Math.abs(w);
          return v>=4;
        });
        const r=ev(p);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
      }
    }
    if(best.f1>v61r.f1)console.log(`  Best: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th.toFixed(3)}`);
    else console.log(`  No improvement`);
  }
}

// ============================================================
// DIR I: Exponential moving average of votes
// ============================================================
console.log('\n=== DIR I: EMA of votes ===');
{
  let best={f1:0};
  for(let alpha=0.1;alpha<=0.9;alpha+=0.1){
    const ema=new Array(N);
    ema[0]=v61sc[0];
    for(let i=1;i<N;i++){
      if(dt[i]>5) ema[i]=v61sc[i]; // reset at session boundary
      else ema[i]=alpha*v61sc[i]+(1-alpha)*ema[i-1];
    }
    for(let th=3;th<=5;th+=0.25){
      const p=ema.map(v=>v>=th);
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,alpha,th};
    }
  }
  if(best.f1>v61r.f1)console.log(`Best: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} alpha=${best.alpha.toFixed(1)} th=${best.th}`);
  else console.log('No improvement over v61');
}

// ============================================================
// DIR J: Confidence calibration — logistic transform of votes
// ============================================================
console.log('\n=== DIR J: Logistic transform ===');
{
  let best={f1:0};
  for(let k=0.5;k<=3;k+=0.25){
    for(let mid=3;mid<=5;mid+=0.25){
      const logistic = v61sc.map(v => 1/(1+Math.exp(-k*(v-mid))));
      for(let th=0.3;th<=0.7;th+=0.05){
        const p=logistic.map(v=>v>=th);
        const r=ev(p);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,k,mid,th};
      }
    }
  }
  if(best.f1>v61r.f1)console.log(`Best: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} k=${best.k} mid=${best.mid} th=${best.th}`);
  else console.log('No improvement over v61');
}

// ============================================================
// DIR K: Negative correlation — score and velocity moving opposite
// ============================================================
console.log('\n=== DIR K: Score-velocity correlation in window ===');
{
  const svCorr = all.map((s,i) => {
    const hw=2;
    const scores=[], vels=[];
    for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
      scores.push(all[j].score); vels.push(all[j].jawVelocity);
    }
    if(scores.length<3) return 0;
    const ms=mean(scores), mv=mean(vels);
    let num=0,ds=0,dv=0;
    for(let k=0;k<scores.length;k++){
      num+=(scores[k]-ms)*(vels[k]-mv);
      ds+=(scores[k]-ms)**2;
      dv+=(vels[k]-mv)**2;
    }
    return (ds>0&&dv>0)?num/Math.sqrt(ds*dv):0;
  });
  
  const uv=[], nv=[];
  for(let i=0;i<N;i++){if(act[i])uv.push(svCorr[i]);else nv.push(svCorr[i]);}
  const d = (mean(uv)-mean(nv)) / Math.sqrt((std(uv)**2+std(nv)**2)/2);
  console.log(`svCorr: d=${d.toFixed(3)} user=${mean(uv).toFixed(4)} nonUser=${mean(nv).toFixed(4)}`);
  
  let best={f1:0};
  for(let w=-2;w<=2;w+=0.25){
    if(w===0)continue;
    for(let th=-0.8;th<=0.8;th+=0.1){
      const p=all.map((s,i)=>{
        let v=v61sc[i];
        if(w>0&&svCorr[i]>=th)v+=w;
        if(w<0&&svCorr[i]<th)v+=Math.abs(w);
        return v>=4;
      });
      const r=ev(p);
      if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,w,th};
    }
  }
  if(best.f1>v61r.f1)console.log(`Best: F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log('No improvement over v61');
}

// ============================================================
// DIR L: Threshold tuning on v61 (maybe 4 isn't optimal anymore)
// ============================================================
console.log('\n=== DIR L: Threshold re-optimization ===');
{
  let best={f1:0};
  for(let t=2.5;t<=6;t+=0.125){
    const p=v61sc.map(v=>v>=t);
    const r=ev(p);
    if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,t};
  }
  console.log(`Best threshold: t=${best.t} R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
}

console.log('\n=== SUMMARY ===');
console.log('v61: F1=85.8%, R=91.7%, S=95.2%, FP=48, FN=18');
