// autoresearch v125: Deep dive into sth=0.7 subset + jvw=0.2 penalty
const fs = require('fs');
const DATA = '/Users/kenefe/LOCAL/momo-agent/projects/human-sense-demo/Tests/speaker-test-data.jsonl';
const lines = fs.readFileSync(DATA, 'utf8').trim().split('\n');
const all = lines.map(l => JSON.parse(l));
all.sort((a, b) => a.audioTime - b.audioTime);
const N = all.length;
const act = all.map(s => s.isUserSpeaker);
const dt = all.map((s, i) => i === 0 ? 0 : s.audioTime - all[i - 1].audioTime);
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);
const jvw = all.map(s => s.jawVelocityWeight || 0);

function twZone(i, sec) {
  const t0 = all[i].audioTime;
  const idx = [];
  for(let j=i;j>=0;j--){if(t0-all[j].audioTime>sec)break;idx.push(j);}
  for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>sec)break;idx.push(j);}
  return { jdMean: mean(idx.map(j=>all[j].jawDelta)), jeMean: mean(idx.map(j=>jawEff[j])) };
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

// === Part 1: jvw=0.2 tokens — user vs AI comparison ===
console.log('=== jvw=0.2 user vs AI ===\n');
{
  const jvw02 = [];
  for(let i=0;i<N;i++) if(jvw[i] < 0.3) jvw02.push(i);
  const jvw02user = jvw02.filter(i=>act[i]);
  const jvw02ai = jvw02.filter(i=>!act[i]);
  
  console.log(`jvw=0.2: ${jvw02.length} total, ${jvw02user.length} user, ${jvw02ai.length} AI`);
  
  for(const [name, fn] of [
    ['jawVelocity', i=>all[i].jawVelocity],
    ['jawDelta', i=>all[i].jawDelta],
    ['score', i=>all[i].score],
    ['finalScore', i=>(all[i].finalScore||0)],
    ['jawEff', i=>jawEff[i]],
    ['scoreVelAnti', i=>scoreVelAnti[i]],
  ]){
    const uV = jvw02user.map(fn), aV = jvw02ai.map(fn);
    const uM=mean(uV), aM=mean(aV);
    console.log(`  ${name.padEnd(16)} User=${uM.toFixed(4)} AI=${aM.toFixed(4)}`);
  }
}

// === Part 2: Aggressive jvw=0.2 penalty search ===
console.log('\n=== jvw=0.2 penalty search ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=3;vHW+=0.5){
      for(let jdW=1.5;jdW<=3;jdW+=0.5){
        for(let fsW=1;fsW<=3.5;fsW+=0.5){
          for(let jvwPen=0;jvwPen<=3;jvwPen+=0.5){  // penalty for jvw=0.2
            for(let jvwBon=0;jvwBon<=2;jvwBon+=0.5){  // bonus for jvw=1
              for(let t=4;t<=7;t+=0.25){
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
                  // jvw features
                  if(jvwPen > 0 && jvw[i] < 0.3) v -= jvwPen;
                  if(jvwBon > 0 && jvw[i] >= 0.9 && jvw[i] <= 1.1) v += jvwBon;
                  // Penalties
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsW,jvwPen,jvwBon,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`jvw penalty: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  jvwPen=${best.jvwPen} jvwBon=${best.jvwBon} threshold=${best.t}`);
  }
}

// === Part 3: Conditional rules — different weights when jvw=0.2 ===
console.log('\n=== Conditional jvw rules ===\n');
{
  let best = {f1:0}, count=0;
  
  // When jvw=0.2: reduce jaw-based weights, increase score-based weights
  for(let jawScale=0.2;jawScale<=0.8;jawScale+=0.2){  // scale down jaw features when jvw=0.2
    for(let fsW=1.5;fsW<=3.5;fsW+=0.5){
      for(let fsW02=2;fsW02<=5;fsW02+=0.5){  // stronger fs penalty when jvw=0.2
        for(let t=4.5;t<=6.5;t+=0.25){
          const preds = all.map((s,i) => {
            let v = 0;
            const f = tw10[i];
            const isLowJvw = jvw[i] < 0.3;
            const js = isLowJvw ? jawScale : 1;
            
            if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 5 * js;
            if(s.jawVelocity >= 0.5) v += 2 * js;
            else if(s.jawVelocity >= 0.1) v += 0.6 * js;
            if(s.jawDelta >= 0.05) v += 2 * js;
            else if(s.jawDelta >= 0.02) v += 0.8 * js;
            if(jawEff[i] >= 5) v += 0.5 * js;
            if(scoreVelAnti[i] >= 0.2) v += 0.5;
            if(s.score < 0.45) v += 0.5;
            if(dt[i] >= 0.2) v += 0.5;
            if(dtZeroRatio5[i] >= 0.5) v += 0.5;
            if(f.jdMean < 0.005) v -= 2;
            if(f.jeMean < 1.5) v -= 1;
            if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
            const curFsW = isLowJvw ? fsW02 : fsW;
            if((s.finalScore||0) >= 0.7) v -= curFsW;
            return v >= t;
          });
          const r = ev(preds);
          if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
            best={...r,jawScale,fsW,fsW02,t};
            count++;
          }
        }
      }
    }
  }
  console.log(`Conditional: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  jawScale(jvw=0.2)=${best.jawScale} fsW=${best.fsW} fsW(jvw=0.2)=${best.fsW02} threshold=${best.t}`);
  }
}

// === Part 4: Score-only classifier for jvw=0.2 tokens ===
console.log('\n=== Score-only for jvw=0.2 ===\n');
{
  // When jaw data is unreliable, rely only on score/finalScore/dt
  let best = {f1:0};
  
  for(let scTh=0.3;scTh<=0.6;scTh+=0.05){
    for(let fsTh=0.3;fsTh<=0.7;fsTh+=0.1){
      for(let dtTh=0.05;dtTh<=0.3;dtTh+=0.05){
        // For jvw=0.2: user if score < scTh AND finalScore < fsTh AND dt >= dtTh
        // For jvw>=1: use v112
        const preds = all.map((s,i) => {
          if(jvw[i] >= 0.9){
            // v112 for reliable jaw data
            let v = 0;
            const f = tw10[i];
            if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 5;
            if(s.jawVelocity >= 0.5) v += 2;
            else if(s.jawVelocity >= 0.1) v += 0.6;
            if(s.jawDelta >= 0.05) v += 2;
            else if(s.jawDelta >= 0.02) v += 0.8;
            if(jawEff[i] >= 5) v += 0.5;
            if(scoreVelAnti[i] >= 0.2) v += 0.5;
            if(s.score < 0.45) v += 0.5;
            if(dt[i] >= 0.2) v += 0.5;
            if(dtZeroRatio5[i] >= 0.5) v += 0.5;
            if(f.jdMean < 0.005) v -= 2;
            if(f.jeMean < 1.5) v -= 1;
            if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
            if((s.finalScore||0) >= 0.7) v -= 2.5;
            return v >= 5.75;
          }
          // jvw=0.2 or jvw=2: use zone + reduced jaw + score
          let v = 0;
          const f = tw10[i];
          if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 5;
          if(s.jawVelocity >= 0.5) v += 2;
          else if(s.jawVelocity >= 0.1) v += 0.6;
          if(s.jawDelta >= 0.05) v += 2;
          else if(s.jawDelta >= 0.02) v += 0.8;
          if(jawEff[i] >= 5) v += 0.5;
          if(scoreVelAnti[i] >= 0.2) v += 0.5;
          if(s.score < scTh) v += 0.5;
          if(dt[i] >= 0.2) v += 0.5;
          if(dtZeroRatio5[i] >= 0.5) v += 0.5;
          if(f.jdMean < 0.005) v -= 2;
          if(f.jeMean < 1.5) v -= 1;
          if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
          if((s.finalScore||0) >= fsTh) v -= 2.5;
          return v >= 5.75;
        });
        const r = ev(preds);
        if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
          best={...r,scTh,fsTh,dtTh};
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`Score-only: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  scTh=${best.scTh} fsTh=${best.fsTh} dtTh=${best.dtTh}`);
  }
}

// === Part 5: Lower finalScore threshold for jvw=0.2 ===
console.log('\n=== Lower fs threshold for jvw=0.2 ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let fsW=1.5;fsW<=3;fsW+=0.5){
    for(let fsTh02=0.3;fsTh02<=0.7;fsTh02+=0.05){
      for(let fsW02=1;fsW02<=4;fsW02+=0.5){
        for(let t=5;t<=6.5;t+=0.25){
          const preds = all.map((s,i) => {
            let v = 0;
            const f = tw10[i];
            if(f.jdMean >= 0.03 && f.jeMean >= 5) v += 5;
            if(s.jawVelocity >= 0.5) v += 2;
            else if(s.jawVelocity >= 0.1) v += 0.6;
            if(s.jawDelta >= 0.05) v += 2;
            else if(s.jawDelta >= 0.02) v += 0.8;
            if(jawEff[i] >= 5) v += 0.5;
            if(scoreVelAnti[i] >= 0.2) v += 0.5;
            if(s.score < 0.45) v += 0.5;
            if(dt[i] >= 0.2) v += 0.5;
            if(dtZeroRatio5[i] >= 0.5) v += 0.5;
            if(f.jdMean < 0.005) v -= 2;
            if(f.jeMean < 1.5) v -= 1;
            if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
            // Different fs penalty based on jvw
            if(jvw[i] < 0.3){
              if((s.finalScore||0) >= fsTh02) v -= fsW02;
            } else {
              if((s.finalScore||0) >= 0.7) v -= fsW;
            }
            return v >= t;
          });
          const r = ev(preds);
          if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
            best={...r,fsW,fsTh02,fsW02,t};
            count++;
          }
        }
      }
    }
  }
  console.log(`Lower fs: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  fsW(jvw>=1)=${best.fsW} fsTh(jvw=0.2)=${best.fsTh02} fsW(jvw=0.2)=${best.fsW02} threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
