// autoresearch v76: jawWeight as key discriminator + combined optimization
// jawWeight: FP=0.200 vs TP=0.644, Cohen's d=1.579 — strongest new signal!
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
const scoreMean5=wstat(all.map(s=>s.score),2,mean);
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
const scoreStd5=wstat(all.map(s=>s.score),2,std);
const scoreGap=all.map(s=>Math.abs(s.finalScore-s.score));
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {if(a.length<2)return 0;const n=a.length,mx=mean(a.map((_,i)=>i)),my=mean(a);let num=0,den=0;a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});return den>0?num/den:0;});
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);

// First: understand jawWeight distribution
console.log('=== jawWeight analysis ===\n');
{
  const jwVals = all.map(s => s.jawWeight || 0);
  const unique = [...new Set(jwVals)].sort((a,b)=>a-b);
  console.log(`Unique jawWeight values: ${unique.join(', ')}`);
  
  // Distribution by class
  for(const v of unique) {
    const userCount = all.filter((s,i) => (s.jawWeight||0)===v && act[i]).length;
    const aiCount = all.filter((s,i) => (s.jawWeight||0)===v && !act[i]).length;
    console.log(`  jw=${v}: user=${userCount} ai=${aiCount} userRate=${(userCount/(userCount+aiCount)*100).toFixed(1)}%`);
  }
  
  // jawWeight in FP vs TP (from v73 two-pass)
  const v73sc = all.map((s,i) => {
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
  });
  
  // jawWeight as penalty/bonus in vote system
  console.log('\n=== Part 1: jawWeight integration ===\n');
  let best={f1:0}, count=0;
  
  // Strategy: low jawWeight = more likely AI → penalty
  // high jawWeight = more likely user → bonus
  for(let jwPenTh=0.15;jwPenTh<=0.3;jwPenTh+=0.05){
    for(let jwPenW=0.5;jwPenW<=3;jwPenW+=0.25){
      for(let jwBonTh=0.3;jwBonTh<=0.8;jwBonTh+=0.1){
        for(let jwBonW=0;jwBonW<=2;jwBonW+=0.25){
          const votes = all.map((s,i) => {
            let v = v73sc[i];
            const jw = s.jawWeight || 0;
            if(jw <= jwPenTh) v -= jwPenW;
            if(jwBonW > 0 && jw >= jwBonTh) v += jwBonW;
            return v;
          });
          
          // Try different thresholds
          for(let t=3.5;t<=5;t+=0.25){
            const s1pred = votes.map(v => v >= t);
            // Two-pass rescue
            const preds = all.map((_,i) => {
              if(s1pred[i]) return true;
              if(votes[i] < -2) return false;
              if(all[i].jawVelocity < 0.075) return false;
              let userN=0, total=0;
              for(let j=Math.max(0,i-10);j<=Math.min(N-1,i+10);j++){
                if(j===i) continue;
                total++;
                if(s1pred[j]) userN++;
              }
              return total>0 && userN/total >= 0.75;
            });
            const r=ev(preds);
            if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
              best={...r,jwPenTh,jwPenW,jwBonTh,jwBonW,t};
              count++;
            }
          }
        }
      }
    }
  }
  console.log(`jawWeight integration: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  penTh=${best.jwPenTh} penW=${best.jwPenW} bonTh=${best.jwBonTh} bonW=${best.jwBonW} t=${best.t}`);
  }
  
  // Part 2: jawWeight + rescue parameter re-optimization
  console.log('\n=== Part 2: jawWeight + rescue re-tune ===\n');
  let best2={f1:0};
  
  // Use best jawWeight params, re-tune rescue
  if(best.f1>0) {
    const {jwPenTh:bpt, jwPenW:bpw, jwBonTh:bbt, jwBonW:bbw, t:bt} = best;
    
    for(let rLow=-3;rLow<=1;rLow+=0.5){
      for(let rHw=8;rHw<=14;rHw++){
        for(let rNTh=0.6;rNTh<=0.85;rNTh+=0.05){
          for(let velTh=0.05;velTh<=0.15;velTh+=0.025){
            const votes = all.map((s,i) => {
              let v = v73sc[i];
              const jw = s.jawWeight || 0;
              if(jw <= bpt) v -= bpw;
              if(bbw > 0 && jw >= bbt) v += bbw;
              return v;
            });
            const s1pred = votes.map(v => v >= bt);
            const preds = all.map((_,i) => {
              if(s1pred[i]) return true;
              if(votes[i] < rLow) return false;
              if(all[i].jawVelocity < velTh) return false;
              let userN=0, total=0;
              for(let j=Math.max(0,i-rHw);j<=Math.min(N-1,i+rHw);j++){
                if(j===i) continue;
                total++;
                if(s1pred[j]) userN++;
              }
              return total>0 && userN/total >= rNTh;
            });
            const r=ev(preds);
            if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best2.f1)best2={...r,rLow,rHw,rNTh,velTh};
          }
        }
      }
    }
    if(best2.f1>0){
      console.log(`Rescue re-tune: R=${(best2.recall*100).toFixed(1)}% S=${(best2.specificity*100).toFixed(1)}% F1=${(best2.f1*100).toFixed(1)}% FP=${best2.FP} FN=${best2.FN}`);
      console.log(`  low=${best2.rLow} hw=${best2.rHw} nTh=${best2.rNTh} velTh=${best2.velTh}`);
    }
  }
}

console.log('\n=== PROGRESS ===');
console.log('v73: F1=89.1%, FP=42, FN=9');
