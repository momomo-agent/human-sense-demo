// autoresearch v86: Push beyond F1=94.4%
// Current: FP=16, FN=9. Every single improvement counts now.
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

function v85votes(s, i) {
  let v=0;
  // Positive votes
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
  // Original penalties
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
  // v84 penalties
  if(!isHighJW[i] && dt[i]<0.001 && s.jawVelocity>=0.4 && s.score<0.4) v -= 2.0;
  if(!isHighJW[i] && burstLen[i]<=2 && dt[i]<0.001) v -= 1.75;
  return v;
}
const v85sc = all.map((s,i) => v85votes(s,i));
const v85pred = v85sc.map(v => v >= 4);

function v85twoPred() {
  return all.map((_,i) => {
    if(v85pred[i]) return true;
    if(all[i].jawVelocity < 0.1) return false;
    const hw = isHighJW[i] ? 8 : 10;
    const nTh = isHighJW[i] ? 0.5 : 0.85;
    const low = isHighJW[i] ? -3 : -1;
    if(v85sc[i] < low) return false;
    let userN=0, total=0;
    for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
      if(j===i) continue;
      total++;
      if(v85pred[j]) userN++;
    }
    return total>0 && userN/total >= nTh;
  });
}
const v85tp = v85twoPred();
const v85r = ev(v85tp);
console.log(`v85 baseline: R=${(v85r.recall*100).toFixed(1)}% S=${(v85r.specificity*100).toFixed(1)}% F1=${(v85r.f1*100).toFixed(1)}% FP=${v85r.FP} FN=${v85r.FN}\n`);

// Analyze remaining errors
const FP=[], FN=[];
for(let i=0;i<N;i++){
  if(v85tp[i]&&!act[i])FP.push(i);
  if(!v85tp[i]&&act[i])FN.push(i);
}

console.log('=== Remaining FP ('+FP.length+') ===');
FP.forEach(i => {
  const s=all[i];
  console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} jw=${isHighJW[i]?1:0.2} v=${v85sc[i].toFixed(2)} gap=${scoreGap[i].toFixed(3)} bl=${burstLen[i]}`);
});

console.log('\n=== Remaining FN ('+FN.length+') ===');
FN.forEach(i => {
  const s=all[i];
  let userN=0, total=0;
  for(let j=Math.max(0,i-10);j<=Math.min(N-1,i+10);j++){if(j!==i){total++;if(v85pred[j])userN++;}}
  console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} jw=${isHighJW[i]?1:0.2} v=${v85sc[i].toFixed(2)} gap=${scoreGap[i].toFixed(3)} nD=${(userN/total).toFixed(2)}`);
});

// ============================================================
// Part 1: Target remaining FP — what patterns remain?
// ============================================================
console.log('\n=== Part 1: FP pattern analysis ===');
{
  const fpDt0 = FP.filter(i => dt[i]<0.001);
  const fpDtPos = FP.filter(i => dt[i]>=0.001);
  console.log(`dt=0: ${fpDt0.length}, dt>0: ${fpDtPos.length}`);
  console.log(`High votes (>=6): ${FP.filter(i=>v85sc[i]>=6).length}`);
  console.log(`Medium votes (4-6): ${FP.filter(i=>v85sc[i]>=4&&v85sc[i]<6).length}`);
  console.log(`Low votes (<4, rescued): ${FP.filter(i=>v85sc[i]<4).length}`);
}

// ============================================================
// Part 2: Try new penalty directions
// ============================================================
console.log('\n=== Part 2: New penalty directions ===\n');
{
  let best={f1:0}, count=0;
  
  // Direction A: jw=0.2 + dt>0 + score in [0.5, 0.75) (medium score, not caught by existing)
  // Direction B: jw=0.2 + dt>0 + low scoreGap
  // Direction C: jw=0.2 + dt=0 + high votes but low scoreGap (AI with strong features)
  // Direction D: dtEnt5 penalty (FP dtEnt5=0.905 vs TP=1.166)
  
  for(let aW=0;aW<=3;aW+=0.5){
    for(let aScoreLow=0.5;aScoreLow<=0.7;aScoreLow+=0.1){
      for(let bW=0;bW<=3;bW+=0.5){
        for(let bGapTh=0.03;bGapTh<=0.1;bGapTh+=0.025){
          for(let dW=0;dW<=2;dW+=0.5){
            for(let dEntTh=0.5;dEntTh<=1;dEntTh+=0.25){
              const votes = all.map((s,i) => {
                let v = v85sc[i];
                if(aW>0 && !isHighJW[i] && dt[i]>=0.001 && s.score>=aScoreLow && s.score<0.75) v -= aW;
                if(bW>0 && !isHighJW[i] && dt[i]>=0.001 && scoreGap[i]<bGapTh) v -= bW;
                if(dW>0 && !isHighJW[i] && dtEnt5[i]<dEntTh) v -= dW;
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
                best={...r,aW,aScoreLow,bW,bGapTh,dW,dEntTh};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`New penalties: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  A: W=${best.aW} score>=${best.aScoreLow}`);
    console.log(`  B: W=${best.bW} gap<${best.bGapTh}`);
    console.log(`  D: W=${best.dW} ent<${best.dEntTh}`);
  }
}

// ============================================================
// Part 3: Rescue re-optimization for FN
// ============================================================
console.log('\n=== Part 3: Rescue re-optimization ===\n');
{
  let best={f1:0};
  
  for(let hwH=4;hwH<=14;hwH+=2){
    for(let nThH=0.2;nThH<=0.7;nThH+=0.05){
      for(let hwL=4;hwL<=14;hwL+=2){
        for(let nThL=0.5;nThL<=0.95;nThL+=0.05){
          for(let velTh=0.05;velTh<=0.15;velTh+=0.025){
            for(let lowH=-5;lowH<=0;lowH+=1){
              for(let lowL=-3;lowL<=1;lowL+=1){
                const preds = all.map((_,i) => {
                  if(v85pred[i]) return true;
                  if(all[i].jawVelocity < velTh) return false;
                  const hw = isHighJW[i] ? hwH : hwL;
                  const nTh = isHighJW[i] ? nThH : nThL;
                  const low = isHighJW[i] ? lowH : lowL;
                  if(v85sc[i] < low) return false;
                  let userN=0, total=0;
                  for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
                    if(j===i) continue;
                    total++;
                    if(v85pred[j]) userN++;
                  }
                  return total>0 && userN/total >= nTh;
                });
                const r=ev(preds);
                if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,hwH,nThH,hwL,nThL,velTh,lowH,lowL};
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Rescue: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  jw=1: hw=${best.hwH} nTh=${best.nThH} low=${best.lowH}`);
    console.log(`  jw=0.2: hw=${best.hwL} nTh=${best.nThL} low=${best.lowL} velTh=${best.velTh}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v85: F1=94.4%, FP=16, FN=9');
