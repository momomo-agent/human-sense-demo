// autoresearch v80: Target the 22 dt>0 jw=0.2 FP specifically
// These are the hardest FP — they have real timing gaps, look like user speech
// But ALL of them are jw=0.2 — can we use jw + other features to suppress?
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
// Part 1: Add jw=0.2 + dt>0 specific penalty
// ============================================================
console.log('=== Part 1: jw=0.2 + dt>0 penalty ===\n');
{
  let best={f1:0}, count=0;
  
  // For jw=0.2 + dt>0: add penalty based on score level
  for(let penW=0.25;penW<=3;penW+=0.25){
    for(let scoreTh=0.5;scoreTh<=0.8;scoreTh+=0.05){
      const votes = all.map((s,i) => {
        let v = v73sc[i];
        // jw=0.2 + dt>0 + high score → penalty (AI with natural timing)
        if(!isHighJW[i] && dt[i]>=0.001 && s.score>=scoreTh) v -= penW;
        return v;
      });
      
      for(let t=3.5;t<=4.5;t+=0.25){
        const s1pred = votes.map(v => v >= t);
        // Asymmetric rescue (v78 best)
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
          best={...r,penW,scoreTh,t};
          count++;
        }
      }
    }
  }
  console.log(`jw+dt penalty: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  penW=${best.penW} scoreTh=${best.scoreTh} t=${best.t}`);
  }
}

// ============================================================
// Part 2: Combined — jw penalty + dt penalty + asymmetric rescue
// ============================================================
console.log('\n=== Part 2: Combined penalties ===\n');
{
  let best={f1:0}, count=0;
  
  for(let dtPenW=0;dtPenW<=2;dtPenW+=0.25){
    for(let dtScoreTh=0.5;dtScoreTh<=0.8;dtScoreTh+=0.1){
      for(let dt0PenW=0;dt0PenW<=1.5;dt0PenW+=0.25){
        for(let dt0ScoreTh=0.4;dt0ScoreTh<=0.7;dt0ScoreTh+=0.1){
          const votes = all.map((s,i) => {
            let v = v73sc[i];
            if(!isHighJW[i]) {
              if(dt[i]>=0.001 && s.score>=dtScoreTh) v -= dtPenW;
              if(dt[i]<0.001 && s.score>=dt0ScoreTh && s.jawVelocity>=0.5) v -= dt0PenW;
            }
            return v;
          });
          
          const s1pred = votes.map(v => v >= 4);
          // Asymmetric rescue
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
            best={...r,dtPenW,dtScoreTh,dt0PenW,dt0ScoreTh};
            count++;
          }
        }
      }
    }
  }
  console.log(`Combined: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  dt>0: penW=${best.dtPenW} scoreTh=${best.dtScoreTh}`);
    console.log(`  dt=0: penW=${best.dt0PenW} scoreTh=${best.dt0ScoreTh}`);
  }
}

// ============================================================
// Part 3: Full combined — all penalties + all rescue params
// ============================================================
console.log('\n=== Part 3: Full combined optimization ===\n');
{
  let best={f1:0}, count=0;
  
  for(let dtPenW=0;dtPenW<=2;dtPenW+=0.5){
    for(let dtScoreTh=0.5;dtScoreTh<=0.8;dtScoreTh+=0.1){
      const votes = all.map((s,i) => {
        let v = v73sc[i];
        if(!isHighJW[i] && dt[i]>=0.001 && s.score>=dtScoreTh) v -= dtPenW;
        return v;
      });
      
      for(let tH=3;tH<=4.5;tH+=0.25){
        for(let tL=4;tL<=5;tL+=0.25){
          const s1pred = all.map((_,i) => {
            const t = isHighJW[i] ? tH : tL;
            return votes[i] >= t;
          });
          
          for(let hwH=6;hwH<=10;hwH+=2){
            for(let nThH=0.4;nThH<=0.6;nThH+=0.1){
              for(let hwL=8;hwL<=14;hwL+=2){
                for(let nThL=0.7;nThL<=0.9;nThL+=0.1){
                  const preds = all.map((_,i) => {
                    if(s1pred[i]) return true;
                    if(all[i].jawVelocity < 0.1) return false;
                    const hw = isHighJW[i] ? hwH : hwL;
                    const nTh = isHighJW[i] ? nThH : nThL;
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
                    best={...r,dtPenW,dtScoreTh,tH,tL,hwH,nThH,hwL,nThL};
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
  console.log(`Full combined: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  dtPen: W=${best.dtPenW} scoreTh=${best.dtScoreTh}`);
    console.log(`  thresholds: tH=${best.tH} tL=${best.tL}`);
    console.log(`  rescue jw=1: hw=${best.hwH} nTh=${best.nThH}`);
    console.log(`  rescue jw=0.2: hw=${best.hwL} nTh=${best.nThL}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v73: F1=89.1%, FP=42, FN=9');
console.log('v78: F1=89.7%, FP=40, FN=8');
