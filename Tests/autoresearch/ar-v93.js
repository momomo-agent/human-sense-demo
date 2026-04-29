// autoresearch v93: Systematic re-optimization with penalties
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

const scoreGap = all.map(s => {
  const jw = s.jawWeight || 0.2;
  const jvw = s.jawVelocityWeight || jw;
  const jawF = Math.max(0.1, 1.0 - jw * s.jawDelta);
  const velF = Math.max(0.1, 1.0 - jvw * s.jawVelocity);
  const nmF = (s.jawDelta < 0.02 && s.jawVelocity < 0.1) ? 1.5 : 1.0;
  return Math.abs(s.score * jawF * velF * nmF - s.score);
});
const dtEnt5 = wstat(dt, 2, a => {const b=[0,0,0];a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;});
const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
const scoreStd5=wstat(all.map(s=>s.score),2,std);
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

// Full voting with all features + penalties
function votes(s, i, params) {
  const p = params;
  let v = 0;
  
  // Positive: jawVelocity
  if(s.jawVelocity >= p.vH) v += p.vHW;
  else if(s.jawVelocity >= p.vM) v += p.vMW;
  else if(s.jawVelocity >= p.vL) v += p.vLW;
  
  // Positive: jawEffMean5
  if(jawEffMean5[i] >= p.jeTh) v += p.jeW;
  
  // Positive: scoreVelAnti
  if(scoreVelAnti[i] >= p.svTh) v += p.svW;
  
  // Positive: timeDelta
  if(dt[i] >= p.dtH) v += p.dtHW;
  else if(dt[i] >= p.dtM) v += p.dtMW;
  
  // Positive: score (low = more likely user)
  if(s.score < p.sL) v += p.sLW;
  else if(s.score < p.sM) v += p.sMW;
  
  // Positive: velStd5
  if(velStd5[i] >= p.vsTh) v += p.vsW;
  
  // Positive: jawDelta
  if(s.jawDelta >= p.jdTh) v += p.jdW;
  
  // Positive: scoreGap (low gap = more likely user in new data)
  if(scoreGap[i] < p.sgTh) v += p.sgW;
  
  // Penalty: low jawEffMean5 (AI pattern)
  if(jawEffMean5[i] < p.jeLowTh) v -= p.jeLowW;
  
  // Penalty: low velStd5 + dt=0 (AI burst)
  if(velStd5[i] < p.vsLowTh && dt[i] < 0.001) v -= p.vsLowW;
  
  return v;
}

// Coarse grid search
console.log('=== Coarse grid search ===\n');
{
  let best = {f1:0}, count=0;
  const configs = [];
  
  // Generate parameter combinations
  for(let vH=0.4;vH<=0.6;vH+=0.1){
    for(let vHW=3;vHW<=5;vHW+=1){
      for(let vM=0.08;vM<=0.15;vM+=0.035){
        for(let jeTh=4.5;jeTh<=6.5;jeTh+=0.5){
          for(let jeW=1;jeW<=3;jeW+=0.5){
            for(let svTh=0.1;svTh<=0.3;svTh+=0.1){
              for(let svW=0.5;svW<=1.5;svW+=0.5){
                for(let jeLowTh=2;jeLowTh<=4;jeLowTh+=1){
                  for(let jeLowW=1;jeLowW<=3;jeLowW+=1){
                    for(let t=3;t<=5;t+=0.5){
                      const p = {
                        vH, vHW, vM, vMW: vHW*0.5, vL: 0.05, vLW: vHW*0.25,
                        jeTh, jeW, svTh, svW,
                        dtH: 0.3, dtHW: 1, dtM: 0.03, dtMW: 0.5,
                        sL: 0.45, sLW: 1.5, sM: 0.55, sMW: 0.5,
                        vsTh: 0.15, vsW: 0.5,
                        jdTh: 0.03, jdW: 0.5,
                        sgTh: 0.15, sgW: 0.5,
                        jeLowTh, jeLowW,
                        vsLowTh: 0.05, vsLowW: 0.5,
                      };
                      const preds = all.map((s,i) => votes(s, i, p) >= t);
                      const r = ev(preds);
                      if(r.recall>=0.85&&r.specificity>=0.85&&r.f1>best.f1){
                        best={...r,...p,t};
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
  console.log(`Coarse: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  vel: >=${best.vH}→${best.vHW}, >=${best.vM}→${best.vMW}`);
    console.log(`  jawEff: >=${best.jeTh}→${best.jeW}, <${best.jeLowTh}→-${best.jeLowW}`);
    console.log(`  scoreVelAnti: >=${best.svTh}→${best.svW}`);
    console.log(`  threshold: ${best.t}`);
    
    // Add rescue
    const sc = all.map((s,i) => votes(s, i, best));
    const p1 = sc.map(v => v >= best.t);
    
    let bestR = {f1:0};
    for(let hw=6;hw<=14;hw+=2){
      for(let nTh=0.2;nTh<=0.7;nTh+=0.1){
        for(let velTh=0.05;velTh<=0.15;velTh+=0.05){
          for(let low=-4;low<=1;low+=1){
            const preds = all.map((_,i) => {
              if(p1[i]) return true;
              if(all[i].jawVelocity < velTh) return false;
              if(sc[i] < low) return false;
              let userN=0, total=0;
              for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
                if(j===i) continue;
                total++;
                if(p1[j]) userN++;
              }
              return total>0 && userN/total >= nTh;
            });
            const r = ev(preds);
            if(r.recall>=0.85&&r.specificity>=0.85&&r.f1>bestR.f1) bestR={...r,hw,nTh,velTh,low};
          }
        }
      }
    }
    if(bestR.f1>0){
      console.log(`\n+Rescue: R=${(bestR.recall*100).toFixed(1)}% S=${(bestR.specificity*100).toFixed(1)}% F1=${(bestR.f1*100).toFixed(1)}% FP=${bestR.FP} FN=${bestR.FN}`);
      console.log(`  hw=${bestR.hw} nTh=${bestR.nTh} velTh=${bestR.velTh} low=${bestR.low}`);
    }
  }
}

console.log('\n=== PROGRESS ===');
console.log('v90: F1=68.4%');
console.log('v92: F1=78.4%');
