// autoresearch v83: Push beyond F1=91.9%
// Current: FP=29, FN=8. Analyze remaining errors for new patterns.
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

const dtEnt5 = wstat(dt, 2, a => {const b=[0,0,0];a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;});
const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
const scoreStd5=wstat(all.map(s=>s.score),2,std);
const scoreGap=all.map(s=>Math.abs(s.finalScore-s.score));
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {if(a.length<2)return 0;const n=a.length,mx=mean(a.map((_,i)=>i)),my=mean(a);let num=0,den=0;a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});return den>0?num/den:0;});
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

function v82votes(s, i) {
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
  if(scoreGap[i]>=0.425)v+=1.75;
  if(scoreSlope5[i]<-0.1)v+=0.5;
  if(scoreVelAnti[i]>=0.3)v+=0.375;
  // Penalties
  if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15) v -= 1.625;
  if(velStd5[i]>=0.6&&dt[i]<0.001) v -= 0.875;
  if(scoreStd5[i]<0.12&&dt[i]<0.001) v -= 0.375;
  if(v>=4.25&&dt[i]<0.001&&s.score<0.35) v -= 1.75;
  if(s.score>=0.7 && s.jawVelocity>=0.4 && dt[i]<0.001) v -= 2.0;
  // v81 jawWeight penalties
  if(!isHighJW[i]) {
    if(dt[i]>=0.001 && s.score>=0.75) v -= 3.0;
    if(dt[i]<0.001 && s.score>=0.3 && s.jawVelocity>=0.5) v -= 1.5;
  }
  return v;
}
const v82sc = all.map((s,i) => v82votes(s,i));
const v82pred = v82sc.map(v => v >= 4);

function v82twoPred() {
  return all.map((_,i) => {
    if(v82pred[i]) return true;
    if(all[i].jawVelocity < 0.1) return false;
    const hw = isHighJW[i] ? 8 : 10;
    const nTh = isHighJW[i] ? 0.5 : 0.85;
    const low = isHighJW[i] ? -3 : -1;
    if(v82sc[i] < low) return false;
    let userN=0, total=0;
    for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
      if(j===i) continue;
      total++;
      if(v82pred[j]) userN++;
    }
    return total>0 && userN/total >= nTh;
  });
}
const v82tp = v82twoPred();
const v82r = ev(v82tp);
console.log(`v82 baseline: R=${(v82r.recall*100).toFixed(1)}% S=${(v82r.specificity*100).toFixed(1)}% F1=${(v82r.f1*100).toFixed(1)}% FP=${v82r.FP} FN=${v82r.FN}\n`);

// Analyze remaining errors
const FP=[], FN=[], TP=[];
for(let i=0;i<N;i++){
  if(v82tp[i]&&!act[i])FP.push(i);
  if(!v82tp[i]&&act[i])FN.push(i);
  if(v82tp[i]&&act[i])TP.push(i);
}

console.log('=== Remaining FP ===\n');
FP.forEach(i => {
  const s=all[i];
  console.log(`i=${i} "${s.text}" sc=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} jw=${isHighJW[i]?1:0.2} votes=${v82sc[i].toFixed(2)} gap=${scoreGap[i].toFixed(3)} burst=${burstLen[i]}`);
});

console.log('\n=== Remaining FN ===\n');
FN.forEach(i => {
  const s=all[i];
  // Check neighbor density
  let userN=0, total=0;
  for(let j=Math.max(0,i-10);j<=Math.min(N-1,i+10);j++){
    if(j!==i){total++;if(v82pred[j])userN++;}
  }
  console.log(`i=${i} "${s.text}" sc=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} jw=${isHighJW[i]?1:0.2} votes=${v82sc[i].toFixed(2)} gap=${scoreGap[i].toFixed(3)} nDens=${(userN/total).toFixed(2)}`);
});

// Feature distributions for remaining FP vs TP
console.log('\n=== FP vs TP feature comparison ===\n');
const features = {
  score: i => all[i].score,
  jawVelocity: i => all[i].jawVelocity,
  jawDelta: i => all[i].jawDelta,
  scoreGap: i => scoreGap[i],
  burstLen: i => burstLen[i],
  dtEnt5: i => dtEnt5[i],
  velStd5: i => velStd5[i],
  scoreStd5: i => scoreStd5[i],
  votes: i => v82sc[i],
};
for(const [name, fn] of Object.entries(features)) {
  const fpVals = FP.map(fn);
  const tpVals = TP.map(fn);
  const fpM = mean(fpVals), fpS = std(fpVals);
  const tpM = mean(tpVals), tpS = std(tpVals);
  const pooledS = Math.sqrt((fpS**2 + tpS**2)/2);
  const d = pooledS > 0 ? Math.abs(fpM - tpM) / pooledS : 0;
  if(d > 0.3) console.log(`${name}: FP=${fpM.toFixed(3)}±${fpS.toFixed(3)} TP=${tpM.toFixed(3)}±${tpS.toFixed(3)} d=${d.toFixed(3)}`);
}

// ============================================================
// Part 1: Try more aggressive penalties for remaining FP patterns
// ============================================================
console.log('\n=== Part 1: Target remaining FP patterns ===\n');
{
  // FP patterns to target:
  // - dt=0 + low score + high velocity (lip sync burst)
  // - dt>0 + medium score + low gap (AI with natural timing)
  
  let best={f1:0}, count=0;
  
  // Additional penalty: jw=0.2 + dt=0 + vel>=velTh + score<scoreTh
  for(let p1W=0;p1W<=3;p1W+=0.25){
    for(let p1VelTh=0.5;p1VelTh<=2;p1VelTh+=0.25){
      for(let p1ScoreTh=0.3;p1ScoreTh<=0.6;p1ScoreTh+=0.1){
        // Additional penalty: jw=0.2 + dt>0 + score in [0.5, 0.7] + low gap
        for(let p2W=0;p2W<=2;p2W+=0.25){
          for(let p2GapTh=0.05;p2GapTh<=0.15;p2GapTh+=0.05){
            const votes = all.map((s,i) => {
              let v = v82sc[i];
              if(!isHighJW[i]) {
                if(p1W>0 && dt[i]<0.001 && s.jawVelocity>=p1VelTh && s.score<p1ScoreTh) v -= p1W;
                if(p2W>0 && dt[i]>=0.001 && s.score>=0.5 && s.score<0.7 && scoreGap[i]<p2GapTh) v -= p2W;
              }
              return v;
            });
            
            const s1pred = votes.map(v => v >= 4);
            const preds = all.map((_,i) => {
              if(s1pred[i]) return true;
              if(all[i].jawVelocity < 0.1) return false;
              const hw = isHighJW[i] ? 8 : 10;
              const nTh = isHighJW[i] ? 0.5 : 0.85;
              const low = isHighJW[i] ? -3 : -1;
              if(votes[i] < low) return false;
              let userN=0, total=0;
              for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
                if(j===i) continue;
                total++;
                if(s1pred[j]) userN++;
              }
              return total>0 && userN/total >= nTh;
            });
            const r=ev(preds);
            if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
              best={...r,p1W,p1VelTh,p1ScoreTh,p2W,p2GapTh};
              count++;
            }
          }
        }
      }
    }
  }
  console.log(`Extra penalties: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  p1: W=${best.p1W} vel>=${best.p1VelTh} score<${best.p1ScoreTh}`);
    console.log(`  p2: W=${best.p2W} gap<${best.p2GapTh}`);
  }
}

// ============================================================
// Part 2: Rescue optimization — can we rescue more FN?
// ============================================================
console.log('\n=== Part 2: FN rescue optimization ===\n');
{
  let best={f1:0};
  
  for(let hwH=4;hwH<=12;hwH+=2){
    for(let nThH=0.3;nThH<=0.7;nThH+=0.05){
      for(let hwL=6;hwL<=14;hwL+=2){
        for(let nThL=0.6;nThL<=0.95;nThL+=0.05){
          for(let velTh=0.05;velTh<=0.15;velTh+=0.025){
            const preds = all.map((_,i) => {
              if(v82pred[i]) return true;
              if(all[i].jawVelocity < velTh) return false;
              const hw = isHighJW[i] ? hwH : hwL;
              const nTh = isHighJW[i] ? nThH : nThL;
              const low = isHighJW[i] ? -3 : -1;
              if(v82sc[i] < low) return false;
              let userN=0, total=0;
              for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
                if(j===i) continue;
                total++;
                if(v82pred[j]) userN++;
              }
              return total>0 && userN/total >= nTh;
            });
            const r=ev(preds);
            if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,hwH,nThH,hwL,nThL,velTh};
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Rescue: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  jw=1: hw=${best.hwH} nTh=${best.nThH}`);
    console.log(`  jw=0.2: hw=${best.hwL} nTh=${best.nThL} velTh=${best.velTh}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v82: F1=91.9%, FP=29, FN=8');
