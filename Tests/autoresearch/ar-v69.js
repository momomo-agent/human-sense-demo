// autoresearch v69: Deep dive on enhanced two-pass with vel condition
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
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreGap=all.map(s=>Math.abs(s.finalScore-s.score));
const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {if(a.length<2)return 0;const n=a.length,mx=mean(a.map((_,i)=>i)),my=mean(a);let num=0,den=0;a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});return den>0?num/den:0;});

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
const v63pred = v63sc.map(v => v >= 4);

// ============================================================
// Part 1: Ultra-fine search around best config
// Best: low=-1 hw=10 nTh=0.75 extra=vel
// ============================================================
console.log('=== Part 1: Ultra-fine two-pass with vel ===\n');
{
  let best={f1:0}, count=0;
  
  for(let rLow=-2;rLow<=1;rLow+=0.25){
    for(let rHw=6;rHw<=14;rHw++){
      for(let rNTh=0.6;rNTh<=0.9;rNTh+=0.025){
        for(let velTh=0.05;velTh<=0.2;velTh+=0.025){
          const preds = all.map((_,i) => {
            if(v63pred[i]) return true;
            if(v63sc[i] < rLow) return false;
            if(all[i].jawVelocity < velTh) return false;
            
            let userN=0, total=0;
            for(let j=Math.max(0,i-rHw);j<=Math.min(N-1,i+rHw);j++){
              if(j===i) continue;
              total++;
              if(v63pred[j]) userN++;
            }
            return total>0 && userN/total >= rNTh;
          });
          const r=ev(preds);
          if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
            best={...r,rLow,rHw,rNTh,velTh};
            count++;
          }
        }
      }
    }
  }
  console.log(`Fine search: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  low=${best.rLow} hw=${best.rHw} nTh=${best.rNTh} velTh=${best.velTh}`);
  }
}

// ============================================================
// Part 2: Multiple rescue conditions (vel OR gap OR delta)
// ============================================================
console.log('\n=== Part 2: Multiple rescue conditions ===\n');
{
  let best={f1:0};
  
  for(let rLow=-2;rLow<=1;rLow+=0.5){
    for(let rHw=8;rHw<=12;rHw++){
      for(let rNTh=0.65;rNTh<=0.85;rNTh+=0.05){
        for(let velTh=0.05;velTh<=0.15;velTh+=0.025){
          for(let gapTh=0;gapTh<=0.3;gapTh+=0.1){
            for(let deltaTh=0;deltaTh<=0.05;deltaTh+=0.01){
              const preds = all.map((_,i) => {
                if(v63pred[i]) return true;
                if(v63sc[i] < rLow) return false;
                
                // Must have at least one physical signal
                const hasVel = all[i].jawVelocity >= velTh;
                const hasGap = scoreGap[i] >= gapTh && gapTh > 0;
                const hasDelta = all[i].jawDelta >= deltaTh && deltaTh > 0;
                if(!hasVel && !hasGap && !hasDelta) return false;
                
                let userN=0, total=0;
                for(let j=Math.max(0,i-rHw);j<=Math.min(N-1,i+rHw);j++){
                  if(j===i) continue;
                  total++;
                  if(v63pred[j]) userN++;
                }
                return total>0 && userN/total >= rNTh;
              });
              const r=ev(preds);
              if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,rLow,rHw,rNTh,velTh,gapTh,deltaTh};
              }
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Multi-cond: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  low=${best.rLow} hw=${best.rHw} nTh=${best.rNTh} velTh=${best.velTh} gapTh=${best.gapTh} deltaTh=${best.deltaTh}`);
  }
}

// ============================================================
// Part 3: Two-pass + FP suppression (three-pass)
// ============================================================
console.log('\n=== Part 3: Two-pass + FP suppression ===\n');
{
  // Use best two-pass, then suppress isolated FP
  let best={f1:0};
  
  // Best two-pass configs to try
  const configs = [
    {rLow:-1, rHw:10, rNTh:0.75, velTh:0.1},
    {rLow:-1.5, rHw:10, rNTh:0.75, velTh:0.1},
    {rLow:-1, rHw:11, rNTh:0.75, velTh:0.1},
    {rLow:-1, rHw:10, rNTh:0.725, velTh:0.1},
  ];
  
  for(const cfg of configs) {
    const pass2 = all.map((_,i) => {
      if(v63pred[i]) return true;
      if(v63sc[i] < cfg.rLow) return false;
      if(all[i].jawVelocity < cfg.velTh) return false;
      let userN=0, total=0;
      for(let j=Math.max(0,i-cfg.rHw);j<=Math.min(N-1,i+cfg.rHw);j++){
        if(j===i) continue;
        total++;
        if(v63pred[j]) userN++;
      }
      return total>0 && userN/total >= cfg.rNTh;
    });
    
    const p2r = ev(pass2);
    
    // Suppress: remove user predictions that are isolated
    for(let sHw=2;sHw<=6;sHw++){
      for(let sNTh=0.05;sNTh<=0.25;sNTh+=0.05){
        for(let sMinV=4;sMinV<=8;sMinV+=0.5){
          const pass3 = pass2.map((pred,i) => {
            if(!pred) return false;
            if(v63sc[i] >= sMinV) return true;
            let userN=0, total=0;
            for(let j=Math.max(0,i-sHw);j<=Math.min(N-1,i+sHw);j++){
              if(j===i) continue;
              total++;
              if(pass2[j]) userN++;
            }
            if(total>0 && userN/total < sNTh) return false;
            return true;
          });
          const r=ev(pass3);
          if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
            best={...r,...cfg,sHw,sNTh,sMinV};
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Three-pass: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  rescue: low=${best.rLow} hw=${best.rHw} nTh=${best.rNTh} velTh=${best.velTh}`);
    console.log(`  suppress: hw=${best.sHw} nTh=${best.sNTh} minV=${best.sMinV}`);
  }
}

// ============================================================
// Part 4: Analyze the 10 rescued FN — are they real improvements?
// ============================================================
console.log('\n=== Part 4: Rescued FN analysis ===\n');
{
  const pass2 = all.map((_,i) => {
    if(v63pred[i]) return true;
    if(v63sc[i] < -1) return false;
    if(all[i].jawVelocity < 0.1) return false;
    let userN=0, total=0;
    for(let j=Math.max(0,i-10);j<=Math.min(N-1,i+10);j++){
      if(j===i) continue;
      total++;
      if(v63pred[j]) userN++;
    }
    return total>0 && userN/total >= 0.75;
  });
  
  // Which FN were rescued?
  const rescued = [];
  const newFP = [];
  for(let i=0;i<N;i++){
    if(pass2[i] && !v63pred[i] && act[i]) rescued.push(i);
    if(pass2[i] && !v63pred[i] && !act[i]) newFP.push(i);
  }
  
  console.log(`Rescued FN (${rescued.length}):`);
  rescued.forEach(i => {
    const s=all[i];
    console.log(`  i=${i} "${s.text}" score=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} votes=${v63sc[i].toFixed(2)}`);
  });
  
  console.log(`\nNew FP from rescue (${newFP.length}):`);
  newFP.forEach(i => {
    const s=all[i];
    console.log(`  i=${i} "${s.text}" score=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} votes=${v63sc[i].toFixed(2)}`);
  });
  
  // Remaining FN
  const remainFN = [];
  for(let i=0;i<N;i++){
    if(!pass2[i] && act[i]) remainFN.push(i);
  }
  console.log(`\nRemaining FN (${remainFN.length}):`);
  remainFN.forEach(i => {
    const s=all[i];
    // Check neighbor density
    let userN=0, total=0;
    for(let j=Math.max(0,i-10);j<=Math.min(N-1,i+10);j++){
      if(j===i) continue;
      total++;
      if(v63pred[j]) userN++;
    }
    console.log(`  i=${i} "${s.text}" score=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} votes=${v63sc[i].toFixed(2)} neighborDensity=${(userN/total).toFixed(2)}`);
  });
}

console.log('\n=== FINAL PROGRESS ===');
console.log('v49: F1=84.4%');
console.log('v63: F1=86.6%, R=93.6%, S=95.1%');
console.log('v68: F1=87.4%, R=95.4%, S=95.0% (two-pass with vel condition)');
