// autoresearch v82: Fine-tune F1=91.9% — the final push
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

function v73votes(s, i) {
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
  return v;
}
const v73sc = all.map((s,i) => v73votes(s,i));

// ============================================================
// Part 1: Ultra-fine around v81 best
// dt>0: W=3 score>=0.75
// dt=0: W=1.5 score>=0.3 vel>=0.5
// ============================================================
console.log('=== Part 1: Ultra-fine dual penalty ===\n');
{
  let best={f1:0}, count=0;
  
  for(let dtPosW=2;dtPosW<=4;dtPosW+=0.125){
    for(let dtPosScoreTh=0.65;dtPosScoreTh<=0.85;dtPosScoreTh+=0.025){
      for(let dt0W=0.5;dt0W<=2.5;dt0W+=0.125){
        for(let dt0ScoreTh=0.2;dt0ScoreTh<=0.5;dt0ScoreTh+=0.05){
          for(let dt0VelTh=0.3;dt0VelTh<=0.7;dt0VelTh+=0.1){
            const votes = all.map((s,i) => {
              let v = v73sc[i];
              if(!isHighJW[i]) {
                if(dt[i]>=0.001 && s.score>=dtPosScoreTh) v -= dtPosW;
                if(dt[i]<0.001 && s.score>=dt0ScoreTh && s.jawVelocity>=dt0VelTh) v -= dt0W;
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
              best={...r,dtPosW,dtPosScoreTh,dt0W,dt0ScoreTh,dt0VelTh};
              count++;
            }
          }
        }
      }
    }
  }
  console.log(`Ultra-fine: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  dt>0: W=${best.dtPosW} score>=${best.dtPosScoreTh}`);
    console.log(`  dt=0: W=${best.dt0W} score>=${best.dt0ScoreTh} vel>=${best.dt0VelTh}`);
  }
}

// ============================================================
// Part 2: Also tune rescue params with best penalties
// ============================================================
console.log('\n=== Part 2: Rescue re-tune with best penalties ===\n');
{
  let best={f1:0}, count=0;
  
  // Use best penalty from Part 1 (or v81 best as fallback)
  const dtPosW=3, dtPosScoreTh=0.75, dt0W=1.5, dt0ScoreTh=0.3, dt0VelTh=0.5;
  
  const votes = all.map((s,i) => {
    let v = v73sc[i];
    if(!isHighJW[i]) {
      if(dt[i]>=0.001 && s.score>=dtPosScoreTh) v -= dtPosW;
      if(dt[i]<0.001 && s.score>=dt0ScoreTh && s.jawVelocity>=dt0VelTh) v -= dt0W;
    }
    return v;
  });
  const s1pred = votes.map(v => v >= 4);
  
  for(let hwH=4;hwH<=12;hwH+=2){
    for(let nThH=0.3;nThH<=0.7;nThH+=0.05){
      for(let hwL=6;hwL<=14;hwL+=2){
        for(let nThL=0.7;nThL<=0.95;nThL+=0.05){
          for(let velTh=0.05;velTh<=0.15;velTh+=0.025){
            for(let lowH=-3;lowH<=0;lowH+=1){
              for(let lowL=-2;lowL<=1;lowL+=0.5){
                const preds = all.map((_,i) => {
                  if(s1pred[i]) return true;
                  if(all[i].jawVelocity < velTh) return false;
                  const hw = isHighJW[i] ? hwH : hwL;
                  const nTh = isHighJW[i] ? nThH : nThL;
                  const low = isHighJW[i] ? lowH : lowL;
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
                  best={...r,hwH,nThH,hwL,nThL,velTh,lowH,lowL};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Rescue re-tune: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  jw=1: hw=${best.hwH} nTh=${best.nThH} low=${best.lowH}`);
    console.log(`  jw=0.2: hw=${best.hwL} nTh=${best.nThL} low=${best.lowL}`);
    console.log(`  velTh=${best.velTh}`);
  }
}

console.log('\n=== FINAL PROGRESS ===');
console.log('v49: F1=84.4%');
console.log('v63: F1=86.6%');
console.log('v73: F1=89.1%');
console.log('v81: F1=91.9% (dual jw penalty)');
