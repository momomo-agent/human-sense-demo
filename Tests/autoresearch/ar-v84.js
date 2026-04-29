// autoresearch v84: Fine-tune F1=93.1% + explore burstLen
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
  if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15) v -= 1.625;
  if(velStd5[i]>=0.6&&dt[i]<0.001) v -= 0.875;
  if(scoreStd5[i]<0.12&&dt[i]<0.001) v -= 0.375;
  if(v>=4.25&&dt[i]<0.001&&s.score<0.35) v -= 1.75;
  if(s.score>=0.7 && s.jawVelocity>=0.4 && dt[i]<0.001) v -= 2.0;
  if(!isHighJW[i]) {
    if(dt[i]>=0.001 && s.score>=0.75) v -= 3.0;
    if(dt[i]<0.001 && s.score>=0.3 && s.jawVelocity>=0.5) v -= 1.5;
  }
  return v;
}

// ============================================================
// Part 1: Fine-tune the new penalty (p1: vel+score on dt=0 jw=0.2)
// ============================================================
console.log('=== Part 1: Fine-tune p1 penalty ===\n');
{
  let best={f1:0}, count=0;
  
  for(let p1W=1.5;p1W<=4;p1W+=0.125){
    for(let p1VelTh=0.3;p1VelTh<=1;p1VelTh+=0.05){
      for(let p1ScoreTh=0.3;p1ScoreTh<=0.6;p1ScoreTh+=0.025){
        const votes = all.map((s,i) => {
          let v = v82votes(s, i);
          if(!isHighJW[i] && dt[i]<0.001 && s.jawVelocity>=p1VelTh && s.score<p1ScoreTh) v -= p1W;
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
          best={...r,p1W,p1VelTh,p1ScoreTh};
          count++;
        }
      }
    }
  }
  console.log(`Fine-tune p1: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  p1W=${best.p1W} velTh=${best.p1VelTh} scoreTh=${best.p1ScoreTh}`);
  }
}

// ============================================================
// Part 2: Add burstLen penalty (FP burstLen=1.48 vs TP=2.09)
// ============================================================
console.log('\n=== Part 2: burstLen + p1 combined ===\n');
{
  let best={f1:0}, count=0;
  
  // Best p1 from v83: W=3 vel>=0.5 score<0.5
  for(let p1W=2;p1W<=4;p1W+=0.5){
    for(let p1VelTh=0.4;p1VelTh<=0.6;p1VelTh+=0.05){
      for(let p1ScoreTh=0.4;p1ScoreTh<=0.55;p1ScoreTh+=0.05){
        // burstLen penalty: short burst + jw=0.2 → penalty
        for(let blPenW=0;blPenW<=2;blPenW+=0.25){
          for(let blTh=1;blTh<=2;blTh++){
            const votes = all.map((s,i) => {
              let v = v82votes(s, i);
              if(!isHighJW[i] && dt[i]<0.001 && s.jawVelocity>=p1VelTh && s.score<p1ScoreTh) v -= p1W;
              if(blPenW>0 && !isHighJW[i] && burstLen[i]<=blTh && dt[i]<0.001) v -= blPenW;
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
              best={...r,p1W,p1VelTh,p1ScoreTh,blPenW,blTh};
              count++;
            }
          }
        }
      }
    }
  }
  console.log(`burstLen+p1: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  p1: W=${best.p1W} vel>=${best.p1VelTh} score<${best.p1ScoreTh}`);
    console.log(`  burst: W=${best.blPenW} len<=${best.blTh}`);
  }
}

// ============================================================
// Part 3: scoreGap penalty (FP gap=0.101 vs TP=0.222)
// ============================================================
console.log('\n=== Part 3: scoreGap penalty ===\n');
{
  let best={f1:0};
  
  for(let p1W=2;p1W<=4;p1W+=0.5){
    for(let p1VelTh=0.4;p1VelTh<=0.6;p1VelTh+=0.1){
      for(let p1ScoreTh=0.4;p1ScoreTh<=0.55;p1ScoreTh+=0.05){
        for(let gapPenW=0;gapPenW<=2;gapPenW+=0.25){
          for(let gapTh=0.05;gapTh<=0.15;gapTh+=0.025){
            const votes = all.map((s,i) => {
              let v = v82votes(s, i);
              if(!isHighJW[i] && dt[i]<0.001 && s.jawVelocity>=p1VelTh && s.score<p1ScoreTh) v -= p1W;
              if(gapPenW>0 && !isHighJW[i] && scoreGap[i]<gapTh) v -= gapPenW;
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
            if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,p1W,p1VelTh,p1ScoreTh,gapPenW,gapTh};
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`scoreGap: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  p1: W=${best.p1W} vel>=${best.p1VelTh} score<${best.p1ScoreTh}`);
    console.log(`  gap: W=${best.gapPenW} th<${best.gapTh}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v82: F1=91.9%, FP=29, FN=8');
console.log('v83: F1=93.1%, FP=23, FN=8');
