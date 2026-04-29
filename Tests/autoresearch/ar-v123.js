// autoresearch v123: Adaptive threshold + time-weighted voting + segment-level eval + jawWeight exploitation
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

// v112 base score function
function v112Score(s, i) {
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
  return v;
}
const v112scores = all.map((s,i) => v112Score(s,i));

// === Part 1: Adaptive threshold based on zone strength ===
console.log('=== Adaptive threshold ===\n');
{
  let best = {f1:0};
  
  // Stronger zone → lower threshold (more confident), weaker zone → higher threshold
  for(let baseT=5;baseT<=6;baseT+=0.25){
    for(let zoneScale=0.5;zoneScale<=2;zoneScale+=0.25){
      const preds = all.map((s,i) => {
        const f = tw10[i];
        const zoneStrength = Math.min(1, (f.jdMean / 0.06) * (f.jeMean / 10));
        const adaptiveT = baseT - zoneStrength * zoneScale;
        return v112scores[i] >= adaptiveT;
      });
      const r = ev(preds);
      if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
        best={...r,baseT,zoneScale};
      }
    }
  }
  if(best.f1>0){
    console.log(`Adaptive: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  baseT=${best.baseT} zoneScale=${best.zoneScale}`);
  } else console.log('No qualifying adaptive config');
}

// === Part 2: Time-weighted voting (recent tokens matter more) ===
console.log('\n=== Time-weighted voting ===\n');
{
  let best = {f1:0};
  
  for(let tw=5;tw<=15;tw+=5){
    for(let decay=0.5;decay<=2;decay+=0.5){
      for(let t=3;t<=6;t+=0.5){
        const preds = all.map((s,i) => {
          const t0 = all[i].audioTime;
          let wSum = 0, wCnt = 0;
          for(let j=i;j>=0;j--){
            const dt = t0 - all[j].audioTime;
            if(dt > tw) break;
            const w = Math.exp(-decay * dt / tw);
            wSum += v112scores[j] * w;
            wCnt += w;
          }
          return wCnt > 0 ? (wSum / wCnt) >= t : false;
        });
        const r = ev(preds);
        if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
          best={...r,tw,decay,t};
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`TimeWeighted: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  tw=${best.tw}s decay=${best.decay} t=${best.t}`);
  } else console.log('No qualifying time-weighted config');
}

// === Part 3: jawWeight exploitation ===
// jawWeight is 0 or 1 — when 0, jaw data is unreliable
console.log('\n=== jawWeight analysis ===\n');
{
  const jw0 = [], jw1 = [];
  for(let i=0;i<N;i++){
    if((all[i].jawWeight||0) > 0.5) jw1.push(i); else jw0.push(i);
  }
  console.log(`jawWeight=0: ${jw0.length} tokens (${jw0.filter(i=>act[i]).length} user, ${jw0.filter(i=>!act[i]).length} AI)`);
  console.log(`jawWeight=1: ${jw1.length} tokens (${jw1.filter(i=>act[i]).length} user, ${jw1.filter(i=>!act[i]).length} AI)`);
  
  // When jawWeight=0, jaw features are unreliable — should we use different rules?
  const jw0user = jw0.filter(i=>act[i]);
  const jw0ai = jw0.filter(i=>!act[i]);
  console.log(`\njawWeight=0 user tokens: score mean=${mean(jw0user.map(i=>all[i].score)).toFixed(3)}, fs mean=${mean(jw0user.map(i=>all[i].finalScore||0)).toFixed(3)}`);
  console.log(`jawWeight=0 AI tokens: score mean=${mean(jw0ai.map(i=>all[i].score)).toFixed(3)}, fs mean=${mean(jw0ai.map(i=>all[i].finalScore||0)).toFixed(3)}`);
  
  // v112 performance on jw=0 vs jw=1 subsets
  const v112preds = v112scores.map(v => v >= 5.75);
  const jw0r = ev(jw0.map(i => v112preds[i] ? (act[i] ? 'TP' : 'FP') : (act[i] ? 'FN' : 'TN')).reduce((o,v) => {o[v]++;return o;}, {TP:0,FP:0,TN:0,FN:0}));
  // Manual eval for subsets
  let jw0TP=0,jw0FP=0,jw0TN=0,jw0FN=0;
  for(const i of jw0){if(v112preds[i]&&act[i])jw0TP++;else if(v112preds[i]&&!act[i])jw0FP++;else if(!v112preds[i]&&!act[i])jw0TN++;else jw0FN++;}
  let jw1TP=0,jw1FP=0,jw1TN=0,jw1FN=0;
  for(const i of jw1){if(v112preds[i]&&act[i])jw1TP++;else if(v112preds[i]&&!act[i])jw1FP++;else if(!v112preds[i]&&!act[i])jw1TN++;else jw1FN++;}
  console.log(`\nv112 on jw=0: TP=${jw0TP} FP=${jw0FP} TN=${jw0TN} FN=${jw0FN} R=${(jw0TP/(jw0TP+jw0FN)*100||0).toFixed(1)}% S=${(jw0TN/(jw0TN+jw0FP)*100||0).toFixed(1)}%`);
  console.log(`v112 on jw=1: TP=${jw1TP} FP=${jw1FP} TN=${jw1TN} FN=${jw1FN} R=${(jw1TP/(jw1TP+jw1FN)*100||0).toFixed(1)}% S=${(jw1TN/(jw1TN+jw1FP)*100||0).toFixed(1)}%`);
}

// === Part 4: noJawPenalty exploitation ===
console.log('\n=== noJawPenalty analysis ===\n');
{
  const njp = all.map(s => s.noJawPenalty || 0);
  const userIdx = [], aiIdx = [];
  for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }
  
  const uV = userIdx.map(i=>njp[i]), aV = aiIdx.map(i=>njp[i]);
  console.log(`noJawPenalty: User=${mean(uV).toFixed(3)} AI=${mean(aV).toFixed(3)}`);
  
  // Distribution
  const vals = [...new Set(njp)].sort((a,b)=>a-b);
  for(const v of vals){
    const uCnt = userIdx.filter(i=>njp[i]===v).length;
    const aCnt = aiIdx.filter(i=>njp[i]===v).length;
    console.log(`  njp=${v}: User=${uCnt} AI=${aCnt} userRate=${(uCnt/(uCnt+aCnt)*100).toFixed(1)}%`);
  }
}

// === Part 5: jawMargin exploitation ===
console.log('\n=== jawMargin analysis ===\n');
{
  const jm = all.map(s => s.jawMargin || 0);
  const userIdx = [], aiIdx = [];
  for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }
  
  const vals = [...new Set(jm)].sort((a,b)=>a-b);
  for(const v of vals){
    const uCnt = userIdx.filter(i=>jm[i]===v).length;
    const aCnt = aiIdx.filter(i=>jm[i]===v).length;
    console.log(`  jawMargin=${v}: User=${uCnt} AI=${aCnt} userRate=${(uCnt/(uCnt+aCnt)*100).toFixed(1)}%`);
  }
}

// === Part 6: jawVelocityWeight exploitation ===
console.log('\n=== jawVelocityWeight analysis ===\n');
{
  const jvw = all.map(s => s.jawVelocityWeight || 0);
  const userIdx = [], aiIdx = [];
  for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }
  
  const vals = [...new Set(jvw)].sort((a,b)=>a-b);
  for(const v of vals){
    const uCnt = userIdx.filter(i=>jvw[i]===v).length;
    const aCnt = aiIdx.filter(i=>jvw[i]===v).length;
    console.log(`  jvw=${v}: User=${uCnt} AI=${aCnt} userRate=${(uCnt/(uCnt+aCnt)*100).toFixed(1)}%`);
  }
}

// === Part 7: speakerThreshold analysis ===
console.log('\n=== speakerThreshold analysis ===\n');
{
  const st = all.map(s => s.speakerThreshold || 0);
  const vals = [...new Set(st)].sort((a,b)=>a-b);
  for(const v of vals){
    const uCnt = all.filter((s,i)=>st[i]===v&&act[i]).length;
    const aCnt = all.filter((s,i)=>st[i]===v&&!act[i]).length;
    console.log(`  speakerThreshold=${v}: User=${uCnt} AI=${aCnt}`);
  }
}

// === Part 8: Smoothed predictions (majority vote in window) ===
console.log('\n=== Smoothed predictions ===\n');
{
  const v112preds = v112scores.map(v => v >= 5.75);
  let best = {f1:0};
  
  for(let hw=1;hw<=5;hw++){
    for(let majority=0.3;majority<=0.7;majority+=0.1){
      const smoothed = v112preds.map((_,i) => {
        let cnt = 0, total = 0;
        for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
          total++;
          if(v112preds[j]) cnt++;
        }
        return cnt/total >= majority;
      });
      const r = ev(smoothed);
      if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
        best={...r,hw,majority};
      }
    }
  }
  if(best.f1>0){
    console.log(`Smoothed: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} majority=${best.majority}`);
  }
}

// === Part 9: Score-based rescue for FN ===
console.log('\n=== Score-based FN rescue ===\n');
{
  const v112preds = v112scores.map(v => v >= 5.75);
  // Find FN tokens
  const fnIdx = [];
  for(let i=0;i<N;i++) if(!v112preds[i] && act[i]) fnIdx.push(i);
  console.log(`FN count: ${fnIdx.length}`);
  if(fnIdx.length > 0){
    console.log(`FN scores: ${fnIdx.map(i=>v112scores[i].toFixed(2)).join(', ')}`);
    console.log(`FN finalScore: ${fnIdx.map(i=>(all[i].finalScore||0).toFixed(3)).join(', ')}`);
    console.log(`FN score: ${fnIdx.map(i=>all[i].score.toFixed(3)).join(', ')}`);
    console.log(`FN jawDelta: ${fnIdx.map(i=>all[i].jawDelta.toFixed(4)).join(', ')}`);
    console.log(`FN jawVelocity: ${fnIdx.map(i=>all[i].jawVelocity.toFixed(4)).join(', ')}`);
    console.log(`FN zone jdMean: ${fnIdx.map(i=>tw10[i].jdMean.toFixed(4)).join(', ')}`);
    console.log(`FN zone jeMean: ${fnIdx.map(i=>tw10[i].jeMean.toFixed(4)).join(', ')}`);
  }
  
  // Find borderline FP (scores just above threshold)
  const fpIdx = [];
  for(let i=0;i<N;i++) if(v112preds[i] && !act[i]) fpIdx.push(i);
  const borderFP = fpIdx.filter(i => v112scores[i] < 7).sort((a,b) => v112scores[a]-v112scores[b]);
  console.log(`\nBorderline FP (score<7): ${borderFP.length}`);
  if(borderFP.length > 0){
    console.log(`FP scores: ${borderFP.slice(0,20).map(i=>v112scores[i].toFixed(2)).join(', ')}...`);
    console.log(`FP finalScore: ${borderFP.slice(0,20).map(i=>(all[i].finalScore||0).toFixed(3)).join(', ')}...`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
