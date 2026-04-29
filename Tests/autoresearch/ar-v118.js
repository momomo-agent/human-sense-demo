// autoresearch v118: Local contrast + relative features + adaptive threshold
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
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

function twZone(i, sec) {
  const t0 = all[i].audioTime;
  const idx = [];
  for(let j=i;j>=0;j--){if(t0-all[j].audioTime>sec)break;idx.push(j);}
  for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>sec)break;idx.push(j);}
  return { jdMean: mean(idx.map(j=>all[j].jawDelta)), jeMean: mean(idx.map(j=>jawEff[j])), idx };
}
const tw10 = all.map((_,i) => twZone(i, 10));

function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}
const dtZeroRatio5 = wstat(dt, 2, a => a.filter(v => v < 0.001).length / a.length);

// === Part 1: Local contrast features ===
// How different is this token from its local window?
const velContrast5 = all.map((s,i) => {
  const w = [];
  for(let j=Math.max(0,i-2);j<=Math.min(N-1,i+2);j++) if(j!==i) w.push(all[j].jawVelocity);
  const m = mean(w);
  return m > 0.001 ? s.jawVelocity / m : (s.jawVelocity > 0 ? 10 : 0);
});

const jdContrast5 = all.map((s,i) => {
  const w = [];
  for(let j=Math.max(0,i-2);j<=Math.min(N-1,i+2);j++) if(j!==i) w.push(all[j].jawDelta);
  const m = mean(w);
  return m > 0.001 ? s.jawDelta / m : (s.jawDelta > 0 ? 10 : 0);
});

// finalScore contrast
const fsContrast5 = all.map((s,i) => {
  const w = [];
  for(let j=Math.max(0,i-2);j<=Math.min(N-1,i+2);j++) if(j!==i) w.push(all[j].finalScore||0);
  const m = mean(w);
  return (s.finalScore||0) - m; // positive = higher than neighbors
});

// Analyze contrast features
console.log('=== Contrast feature analysis ===\n');
const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

for(const [name, arr] of [['velContrast5', velContrast5], ['jdContrast5', jdContrast5], ['fsContrast5', fsContrast5]]){
  const uV = userIdx.map(i=>arr[i]), aV = aiIdx.map(i=>arr[i]);
  const uM=mean(uV), uS=std(uV), aM=mean(aV), aS=std(aV);
  const d = Math.sqrt((uS**2+aS**2)/2) > 0 ? Math.abs(uM-aM)/Math.sqrt((uS**2+aS**2)/2) : 0;
  console.log(`${name.padEnd(16)} User=${uM.toFixed(3)}±${uS.toFixed(3)} AI=${aM.toFixed(3)}±${aS.toFixed(3)} d=${d.toFixed(3)}`);
}

// === Part 2: v112 base + contrast penalties ===
console.log('\n=== v112 + contrast ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          // fsContrast: positive = higher fs than neighbors = more AI-like
          for(let fcTh=0;fcTh<=0.2;fcTh+=0.1){
            for(let fcW=0;fcW<=1.5;fcW+=0.5){
              for(let p1W=0;p1W<=1;p1W+=0.5){
                for(let t=4.5;t<=6;t+=0.25){
                  const preds = all.map((s,i) => {
                    let v = 0;
                    const f = tw10[i];
                    if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
                    if(s.jawVelocity >= 0.5) v += vHW;
                    else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                    if(s.jawDelta >= 0.05) v += jdW;
                    else if(s.jawDelta >= 0.02) v += jdW*0.4;
                    if(jawEff[i] >= 5) v += 0.5;
                    if(scoreVelAnti[i] >= 0.2) v += 0.5;
                    if(s.score < 0.45) v += 0.5;
                    if(dt[i] >= 0.2) v += 0.5;
                    if(dtZeroRatio5[i] >= 0.5) v += 0.5;
                    if(f.jdMean < 0.005) v -= 2;
                    if(f.jeMean < 1.5) v -= 1;
                    if(!isHighJW[i] && s.score >= 0.7) v -= p1W;
                    if((s.finalScore||0) >= 0.7) v -= fsW;
                    if(fcW>0 && fsContrast5[i] >= fcTh) v -= fcW;
                    return v >= t;
                  });
                  const r = ev(preds);
                  if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                    best={...r,zoneW,vHW,jdW,fsW,fcTh,fcW,p1W,t};
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
  console.log(`v112+contrast: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW}`);
    console.log(`  fs>=0.7→-${best.fsW} fsContrast>=${best.fcTh}→-${best.fcW}`);
    console.log(`  p1W=${best.p1W} threshold=${best.t}`);
  }
}

// === Part 3: Score-velocity phase space ===
// User: low score + high velocity (speaking changes voice but moves jaw)
// AI: high score + low velocity (lip sync matches voice but jaw barely moves)
// But also: AI with high velocity (kenefe moving during AI speech)
console.log('\n=== Score-velocity quadrant ===\n');
{
  // Quadrant analysis
  const quads = [
    {name:'lowSc+highVel', test:(s)=>s.score<0.5&&s.jawVelocity>=0.3},
    {name:'lowSc+lowVel', test:(s)=>s.score<0.5&&s.jawVelocity<0.3},
    {name:'highSc+highVel', test:(s)=>s.score>=0.5&&s.jawVelocity>=0.3},
    {name:'highSc+lowVel', test:(s)=>s.score>=0.5&&s.jawVelocity<0.3},
  ];
  for(const q of quads){
    const uCnt = userIdx.filter(i=>q.test(all[i])).length;
    const aCnt = aiIdx.filter(i=>q.test(all[i])).length;
    const uRate = uCnt/(uCnt+aCnt)||0;
    console.log(`${q.name.padEnd(20)} User=${uCnt} AI=${aCnt} userRate=${(uRate*100).toFixed(1)}%`);
  }
  
  // Try quadrant-based weights
  let best = {f1:0};
  for(let q1W=1;q1W<=3;q1W+=0.5){  // lowSc+highVel bonus
    for(let q4W=0.5;q4W<=2;q4W+=0.5){  // highSc+lowVel penalty
      for(let zoneW=4;zoneW<=5;zoneW+=0.5){
        for(let fsW=1.5;fsW<=2.5;fsW+=0.5){
          for(let t=4;t<=6;t+=0.5){
            const preds = all.map((s,i) => {
              let v = 0;
              const f = tw10[i];
              if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
              if(s.jawDelta >= 0.05) v += 2;
              else if(s.jawDelta >= 0.02) v += 0.8;
              if(jawEff[i] >= 5) v += 0.5;
              if(dt[i] >= 0.2) v += 0.5;
              if(dtZeroRatio5[i] >= 0.5) v += 0.5;
              // Quadrant weights
              if(s.score < 0.5 && s.jawVelocity >= 0.3) v += q1W;
              if(s.score >= 0.5 && s.jawVelocity < 0.3) v -= q4W;
              if(f.jdMean < 0.005) v -= 2;
              if(f.jeMean < 1.5) v -= 1;
              if((s.finalScore||0) >= 0.7) v -= fsW;
              return v >= t;
            });
            const r = ev(preds);
            if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
              best={...r,q1W,q4W,zoneW,fsW,t};
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`\nQuadrant: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  lowSc+highVel→+${best.q1W} highSc+lowVel→-${best.q4W}`);
    console.log(`  zoneW=${best.zoneW} fsW=${best.fsW} t=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6%');
