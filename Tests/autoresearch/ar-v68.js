// autoresearch v68: Deep dive on two-pass + FP suppression
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

// All features
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
// Part 1: Three-pass system — rescue + suppress
// Pass 1: v63 classification
// Pass 2: rescue borderline FN using neighbor context
// Pass 3: suppress borderline FP using neighbor context
// ============================================================
console.log('=== Part 1: Three-pass (rescue + suppress) ===\n');
{
  let best={f1:0}, count=0;
  
  for(let rLow=0;rLow<=2;rLow+=0.5){
    for(let rHw=3;rHw<=8;rHw++){
      for(let rNTh=0.5;rNTh<=0.8;rNTh+=0.1){
        // Pass 2: rescue
        const pass2 = all.map((_,i) => {
          if(v63pred[i]) return true;
          if(v63sc[i] < rLow) return false;
          let userN=0, total=0;
          for(let j=Math.max(0,i-rHw);j<=Math.min(N-1,i+rHw);j++){
            if(j===i) continue;
            total++;
            if(v63pred[j]) userN++;
          }
          return total>0 && userN/total >= rNTh;
        });
        
        // Pass 3: suppress — for tokens that are user but borderline, check if isolated
        for(let sMax=5;sMax<=8;sMax+=0.5){
          for(let sHw=2;sHw<=6;sHw++){
            for(let sNTh=0.1;sNTh<=0.4;sNTh+=0.1){
              const pass3 = pass2.map((pred,i) => {
                if(!pred) return false;
                if(v63sc[i] >= sMax) return true; // high confidence, keep
                // Check if isolated user token
                let userN=0, total=0;
                for(let j=Math.max(0,i-sHw);j<=Math.min(N-1,i+sHw);j++){
                  if(j===i) continue;
                  total++;
                  if(pass2[j]) userN++;
                }
                if(total>0 && userN/total < sNTh) return false; // isolated, suppress
                return true;
              });
              const r=ev(pass3);
              if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1){
                best={...r,rLow,rHw,rNTh,sMax,sHw,sNTh};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Three-pass: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  rescue: low=${best.rLow} hw=${best.rHw} nTh=${best.rNTh}`);
    console.log(`  suppress: max=${best.sMax} hw=${best.sHw} nTh=${best.sNTh}`);
  }
}

// ============================================================
// Part 2: Fine-grained two-pass with additional rescue conditions
// ============================================================
console.log('\n=== Part 2: Enhanced two-pass ===\n');
{
  let best={f1:0}, count=0;
  
  for(let rLow=-1;rLow<=3;rLow+=0.5){
    for(let rHw=3;rHw<=10;rHw++){
      for(let rNTh=0.4;rNTh<=0.9;rNTh+=0.05){
        // Also try: rescue only if scoreGap is high OR velocity is high
        for(let extraCond of ['none', 'gap', 'vel', 'gapOrVel']) {
          const preds = all.map((_,i) => {
            if(v63pred[i]) return true;
            if(v63sc[i] < rLow) return false;
            
            // Extra condition
            let extraOk = true;
            if(extraCond === 'gap') extraOk = scoreGap[i] >= 0.2;
            else if(extraCond === 'vel') extraOk = all[i].jawVelocity >= 0.1;
            else if(extraCond === 'gapOrVel') extraOk = scoreGap[i] >= 0.2 || all[i].jawVelocity >= 0.1;
            if(!extraOk) return false;
            
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
            best={...r,rLow,rHw,rNTh,extraCond};
            count++;
          }
        }
      }
    }
  }
  console.log(`Enhanced two-pass: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  low=${best.rLow} hw=${best.rHw} nTh=${best.rNTh} extra=${best.extraCond}`);
  }
}

// ============================================================
// Part 3: Weighted neighbor density (closer neighbors count more)
// ============================================================
console.log('\n=== Part 3: Distance-weighted neighbor rescue ===\n');
{
  let best={f1:0};
  
  for(let rLow=-1;rLow<=2;rLow+=0.5){
    for(let rHw=3;rHw<=10;rHw++){
      for(let rNTh=0.3;rNTh<=0.8;rNTh+=0.05){
        const preds = all.map((_,i) => {
          if(v63pred[i]) return true;
          if(v63sc[i] < rLow) return false;
          
          // Distance-weighted: closer neighbors have more influence
          let wUser=0, wTotal=0;
          for(let j=Math.max(0,i-rHw);j<=Math.min(N-1,i+rHw);j++){
            if(j===i) continue;
            const dist = Math.abs(j-i);
            const w = 1/dist; // inverse distance weighting
            wTotal += w;
            if(v63pred[j]) wUser += w;
          }
          return wTotal>0 && wUser/wTotal >= rNTh;
        });
        const r=ev(preds);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,rLow,rHw,rNTh};
      }
    }
  }
  if(best.f1>0){
    console.log(`Dist-weighted: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  low=${best.rLow} hw=${best.rHw} nTh=${best.rNTh}`);
  }
}

// ============================================================
// Part 4: Vote-weighted neighbor density
// ============================================================
console.log('\n=== Part 4: Vote-weighted neighbor rescue ===\n');
{
  let best={f1:0};
  
  for(let rLow=-1;rLow<=2;rLow+=0.5){
    for(let rHw=3;rHw<=10;rHw++){
      for(let rNTh=1;rNTh<=5;rNTh+=0.25){
        const preds = all.map((_,i) => {
          if(v63pred[i]) return true;
          if(v63sc[i] < rLow) return false;
          
          // Vote-weighted: neighbors with higher votes count more
          let wSum=0;
          for(let j=Math.max(0,i-rHw);j<=Math.min(N-1,i+rHw);j++){
            if(j===i) continue;
            if(v63pred[j]) wSum += Math.max(0, v63sc[j] - 4); // excess votes
          }
          return wSum >= rNTh;
        });
        const r=ev(preds);
        if(r.recall>=0.9&&r.specificity>=0.9&&r.f1>best.f1)best={...r,rLow,rHw,rNTh};
      }
    }
  }
  if(best.f1>0){
    console.log(`Vote-weighted: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  low=${best.rLow} hw=${best.rHw} nTh=${best.rNTh}`);
  }
}

// ============================================================
// Part 5: Hybrid — v63 + two-pass rescue + FP suppression
// ============================================================
console.log('\n=== Part 5: Best combo — rescue + suppress ===\n');
{
  // Use best rescue params from Part 2, then add FP suppression
  // Best from v67: lowTh=0 hw=7 neighborTh=0.8
  
  let best={f1:0};
  
  // Try multiple rescue configs
  const rescueConfigs = [
    {rLow:0, rHw:7, rNTh:0.8},
    {rLow:0, rHw:6, rNTh:0.7},
    {rLow:0.5, rHw:7, rNTh:0.75},
    {rLow:0, rHw:8, rNTh:0.8},
    {rLow:-0.5, rHw:7, rNTh:0.8},
  ];
  
  for(const rc of rescueConfigs) {
    const pass2 = all.map((_,i) => {
      if(v63pred[i]) return true;
      if(v63sc[i] < rc.rLow) return false;
      let userN=0, total=0;
      for(let j=Math.max(0,i-rc.rHw);j<=Math.min(N-1,i+rc.rHw);j++){
        if(j===i) continue;
        total++;
        if(v63pred[j]) userN++;
      }
      return total>0 && userN/total >= rc.rNTh;
    });
    
    // FP suppression: remove isolated user predictions
    for(let sHw=2;sHw<=5;sHw++){
      for(let sNTh=0.05;sNTh<=0.3;sNTh+=0.05){
        for(let sMinVotes=4;sMinVotes<=7;sMinVotes+=0.5){
          const pass3 = pass2.map((pred,i) => {
            if(!pred) return false;
            if(v63sc[i] >= sMinVotes) return true;
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
            best={...r,...rc,sHw,sNTh,sMinVotes};
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Rescue+suppress: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  rescue: low=${best.rLow} hw=${best.rHw} nTh=${best.rNTh}`);
    console.log(`  suppress: hw=${best.sHw} nTh=${best.sNTh} minVotes=${best.sMinVotes}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v63: F1=86.6%, R=93.6%, S=95.1%');
console.log('v67 two-pass: F1=86.6%, R=94.5%, S=94.8% (more recall, same F1)');
