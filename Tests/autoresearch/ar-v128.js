// autoresearch v128: score-finalScore gap + zone fused + isFinal + text length
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

function twZoneIdx(i, sec) {
  const t0 = all[i].audioTime;
  const idx = [];
  for(let j=i;j>=0;j--){if(t0-all[j].audioTime>sec)break;idx.push(j);}
  for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>sec)break;idx.push(j);}
  return idx;
}
const tw10idx = all.map((_,i) => twZoneIdx(i, 10));
const tw10 = tw10idx.map((idx,i) => ({
  jdMean: mean(idx.map(j=>all[j].jawDelta)),
  jeMean: mean(idx.map(j=>jawEff[j]))
}));

function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}
const dtZeroRatio5 = wstat(dt, 2, a => a.filter(v => v < 0.001).length / a.length);

// === New features ===
// score - finalScore gap: how much does jaw movement change the score?
const scoreGap = all.map(s => s.score - (s.finalScore || 0));
// isFinal
const isFinal = all.map(s => s.isFinal ? 1 : 0);
// text length (single char vs multi)
const textLen = all.map(s => (s.text || '').length);

console.log('=== New feature analysis ===\n');
const userIdx = [], aiIdx = [];
for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }

for(const [name, arr] of [
  ['scoreGap', scoreGap], ['isFinal', isFinal], ['textLen', textLen]
]){
  const uV = userIdx.map(i=>arr[i]), aV = aiIdx.map(i=>arr[i]);
  const uM=mean(uV), uS=std(uV), aM=mean(aV), aS=std(aV);
  const d = Math.sqrt((uS**2+aS**2)/2) > 0 ? Math.abs(uM-aM)/Math.sqrt((uS**2+aS**2)/2) : 0;
  console.log(`${name.padEnd(14)} User=${uM.toFixed(4)}±${uS.toFixed(4)} AI=${aM.toFixed(4)}±${aS.toFixed(4)} d=${d.toFixed(3)}`);
}

// scoreGap distribution
console.log('\nscoreGap distribution:');
for(const th of [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]){
  const uCnt = userIdx.filter(i=>scoreGap[i]>=th).length;
  const aCnt = aiIdx.filter(i=>scoreGap[i]>=th).length;
  console.log(`  gap>=${th.toFixed(1)}: User=${uCnt}(${(uCnt/505*100).toFixed(0)}%) AI=${aCnt}(${(aCnt/1794*100).toFixed(0)}%)`);
}

// === Part 2: scoreGap as feature ===
console.log('\n=== scoreGap feature ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let sgTh=-0.3;sgTh<=0.1;sgTh+=0.1){
            for(let sgW=0.5;sgW<=2;sgW+=0.5){
              for(let t=4.5;t<=6.5;t+=0.25){
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
                  // scoreGap: positive = score > finalScore = jaw helped
                  if(scoreGap[i] >= sgTh) v += sgW;
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsW,sgTh,sgW,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`scoreGap: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  scoreGap>=${best.sgTh}→+${best.sgW} threshold=${best.t}`);
  }
}

// === Part 3: Zone-level fused (fixed) ===
console.log('\n=== Zone-level fused ===\n');
{
  const zoneFsMean10 = all.map((_,i) => mean(tw10idx[i].map(j => all[j].finalScore || 0)));
  const zoneJdFused = all.map((_,i) => tw10[i].jdMean * (1 - zoneFsMean10[i]));
  
  const uV = userIdx.map(i=>zoneJdFused[i]), aV = aiIdx.map(i=>zoneJdFused[i]);
  const d = Math.abs(mean(uV)-mean(aV))/Math.sqrt((std(uV)**2+std(aV)**2)/2);
  console.log(`zoneJdFused: User=${mean(uV).toFixed(4)} AI=${mean(aV).toFixed(4)} d=${d.toFixed(3)}`);
  
  let best = {f1:0}, count=0;
  for(let zjfTh=0.005;zjfTh<=0.03;zjfTh+=0.005){
    for(let zjfW=3;zjfW<=6;zjfW+=0.5){
      for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
        for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
          for(let fsW=0;fsW<=3;fsW+=0.5){
            for(let t=4;t<=6.5;t+=0.25){
              const preds = all.map((s,i) => {
                let v = 0;
                if(zoneJdFused[i] >= zjfTh) v += zjfW;
                if(s.jawVelocity >= 0.5) v += vHW;
                else if(s.jawVelocity >= 0.1) v += vHW*0.3;
                if(s.jawDelta >= 0.05) v += jdW;
                else if(s.jawDelta >= 0.02) v += jdW*0.4;
                if(jawEff[i] >= 5) v += 0.5;
                if(scoreVelAnti[i] >= 0.2) v += 0.5;
                if(s.score < 0.45) v += 0.5;
                if(dt[i] >= 0.2) v += 0.5;
                if(dtZeroRatio5[i] >= 0.5) v += 0.5;
                if(tw10[i].jdMean < 0.005) v -= 2;
                if(tw10[i].jeMean < 1.5) v -= 1;
                if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                if(fsW > 0 && (s.finalScore||0) >= 0.7) v -= fsW;
                return v >= t;
              });
              const r = ev(preds);
              if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                best={...r,zjfTh,zjfW,vHW,jdW,fsW,t};
                count++;
              }
            }
          }
        }
      }
    }
  }
  console.log(`Zone fused: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zjf>=${best.zjfTh}→${best.zjfW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW} t=${best.t}`);
  }
}

// === Part 4: Negative scoreGap as penalty (score < finalScore = jaw hurt) ===
console.log('\n=== Negative scoreGap penalty ===\n');
{
  let best = {f1:0}, count=0;
  
  for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let nsgTh=-0.5;nsgTh<=-0.1;nsgTh+=0.1){
            for(let nsgW=0.5;nsgW<=2;nsgW+=0.5){
              for(let t=4.5;t<=6.5;t+=0.25){
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
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  // Negative gap penalty: score < finalScore means jaw movement hurt the score
                  if(scoreGap[i] <= nsgTh) v -= nsgW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsW,nsgTh,nsgW,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`Neg scoreGap: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  scoreGap<=${best.nsgTh}→-${best.nsgW} threshold=${best.t}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
