// autoresearch v130: timestamp vs audioTime + processing delay + token arrival patterns
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

// === Part 1: Analyze timestamp field ===
console.log('=== timestamp analysis ===\n');
{
  const ts = all.map(s => s.timestamp || 0);
  const tsdt = ts.map((t,i) => i === 0 ? 0 : t - ts[i-1]);
  
  console.log(`timestamp range: ${ts[0].toFixed(3)} - ${ts[N-1].toFixed(3)}`);
  console.log(`audioTime range: ${all[0].audioTime.toFixed(3)} - ${all[N-1].audioTime.toFixed(3)}`);
  
  // timestamp - audioTime offset
  const offset = all.map(s => (s.timestamp || 0) - s.audioTime);
  console.log(`\ntimestamp-audioTime offset: min=${Math.min(...offset).toFixed(3)} max=${Math.max(...offset).toFixed(3)}`);
  
  // timestamp dt vs audioTime dt
  const userIdx = [], aiIdx = [];
  for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }
  
  console.log(`\ntimestamp dt: User=${mean(userIdx.map(i=>tsdt[i])).toFixed(6)} AI=${mean(aiIdx.map(i=>tsdt[i])).toFixed(6)}`);
  console.log(`audioTime dt: User=${mean(userIdx.map(i=>dt[i])).toFixed(6)} AI=${mean(aiIdx.map(i=>dt[i])).toFixed(6)}`);
  
  // Are timestamps all the same? (batch arrival)
  const uniqueTs = [...new Set(ts.map(t=>t.toFixed(6)))];
  console.log(`\nUnique timestamps: ${uniqueTs.length} (out of ${N} tokens)`);
  
  // Tokens per unique timestamp
  const tsCounts = {};
  for(const t of ts) { const k = t.toFixed(6); tsCounts[k] = (tsCounts[k]||0)+1; }
  const counts = Object.values(tsCounts);
  console.log(`Tokens per timestamp: min=${Math.min(...counts)} max=${Math.max(...counts)} mean=${mean(counts).toFixed(1)}`);
  
  // Batch analysis: tokens with same timestamp
  const batchSizes = counts.filter(c => c > 1);
  console.log(`Batches (>1 token): ${batchSizes.length}, sizes: ${batchSizes.slice(0,20).join(', ')}...`);
  
  // User vs AI in batches
  let batchUser = 0, batchAI = 0, singleUser = 0, singleAI = 0;
  const tsMap = {};
  for(let i=0;i<N;i++){
    const k = ts[i].toFixed(6);
    if(!tsMap[k]) tsMap[k] = [];
    tsMap[k].push(i);
  }
  for(const [k, indices] of Object.entries(tsMap)){
    if(indices.length > 1){
      for(const i of indices){
        if(act[i]) batchUser++; else batchAI++;
      }
    } else {
      if(act[indices[0]]) singleUser++; else singleAI++;
    }
  }
  console.log(`\nBatch tokens: User=${batchUser} AI=${batchAI} userRate=${(batchUser/(batchUser+batchAI)*100).toFixed(1)}%`);
  console.log(`Single tokens: User=${singleUser} AI=${singleAI} userRate=${(singleUser/(singleUser+singleAI)*100).toFixed(1)}%`);
}

// === Part 2: Processing delay (timestamp - audioTime) as feature ===
console.log('\n=== Processing delay feature ===\n');
{
  const delay = all.map(s => (s.timestamp || 0) - s.audioTime);
  const userIdx = [], aiIdx = [];
  for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }
  
  const uV = userIdx.map(i=>delay[i]), aV = aiIdx.map(i=>delay[i]);
  const uM=mean(uV), uS=std(uV), aM=mean(aV), aS=std(aV);
  const d = Math.sqrt((uS**2+aS**2)/2) > 0 ? Math.abs(uM-aM)/Math.sqrt((uS**2+aS**2)/2) : 0;
  console.log(`delay: User=${uM.toFixed(3)}±${uS.toFixed(3)} AI=${aM.toFixed(3)}±${aS.toFixed(3)} d=${d.toFixed(3)}`);
}

// === Part 3: Token arrival rate (tokens per second in local window) ===
console.log('\n=== Token arrival rate ===\n');
{
  const arrivalRate = all.map((_,i) => {
    const t0 = all[i].audioTime;
    let cnt = 0;
    for(let j=i;j>=0;j--){if(t0-all[j].audioTime>2)break;cnt++;}
    for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>2)break;cnt++;}
    return cnt / 4; // tokens per second in 4-second window
  });
  
  const userIdx = [], aiIdx = [];
  for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }
  
  const uV = userIdx.map(i=>arrivalRate[i]), aV = aiIdx.map(i=>arrivalRate[i]);
  const uM=mean(uV), uS=std(uV), aM=mean(aV), aS=std(aV);
  const d = Math.sqrt((uS**2+aS**2)/2) > 0 ? Math.abs(uM-aM)/Math.sqrt((uS**2+aS**2)/2) : 0;
  console.log(`arrivalRate: User=${uM.toFixed(2)}±${uS.toFixed(2)} AI=${aM.toFixed(2)}±${aS.toFixed(2)} d=${d.toFixed(3)}`);
  
  // Try as feature
  const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
  const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
  const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);
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
  
  let best = {f1:0};
  for(let arTh=5;arTh<=20;arTh+=2.5){
    for(let arW=0.5;arW<=2;arW+=0.5){
      for(let zoneW=4.5;zoneW<=5.5;zoneW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let t=4.5;t<=6.5;t+=0.25){
            const preds = all.map((s,i) => {
              let v = 0;
              const f = tw10[i];
              if(f.jdMean >= 0.03 && f.jeMean >= 5) v += zoneW;
              if(s.jawVelocity >= 0.5) v += 2;
              else if(s.jawVelocity >= 0.1) v += 0.6;
              if(s.jawDelta >= 0.05) v += 2;
              else if(s.jawDelta >= 0.02) v += 0.8;
              if(jawEff[i] >= 5) v += 0.5;
              if(scoreVelAnti[i] >= 0.2) v += 0.5;
              if(s.score < 0.45) v += 0.5;
              if(dt[i] >= 0.2) v += 0.5;
              if(dtZeroRatio5[i] >= 0.5) v += 0.5;
              // Arrival rate: high rate = more tokens = AI batch
              if(arrivalRate[i] >= arTh) v -= arW;
              if(f.jdMean < 0.005) v -= 2;
              if(f.jeMean < 1.5) v -= 1;
              if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
              if((s.finalScore||0) >= 0.7) v -= fsW;
              return v >= t;
            });
            const r = ev(preds);
            if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
              best={...r,arTh,arW,zoneW,fsW,t};
            }
          }
        }
      }
    }
  }
  if(best.f1>0){
    console.log(`\narrivalRate: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  ar>=${best.arTh}→-${best.arW} zoneW=${best.zoneW} fsW=${best.fsW} t=${best.t}`);
  }
}

// === Part 4: Unique audioTime count in window (how many distinct times) ===
console.log('\n=== Unique time density ===\n');
{
  const uniqueTimeDensity = all.map((_,i) => {
    const t0 = all[i].audioTime;
    const times = new Set();
    for(let j=i;j>=0;j--){if(t0-all[j].audioTime>5)break;times.add(all[j].audioTime.toFixed(2));}
    for(let j=i+1;j<N;j++){if(all[j].audioTime-t0>5)break;times.add(all[j].audioTime.toFixed(2));}
    return times.size;
  });
  
  const userIdx = [], aiIdx = [];
  for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }
  
  const uV = userIdx.map(i=>uniqueTimeDensity[i]), aV = aiIdx.map(i=>uniqueTimeDensity[i]);
  const d = Math.abs(mean(uV)-mean(aV))/Math.sqrt((std(uV)**2+std(aV)**2)/2);
  console.log(`uniqueTimeDensity: User=${mean(uV).toFixed(1)}±${std(uV).toFixed(1)} AI=${mean(aV).toFixed(1)}±${std(aV).toFixed(1)} d=${d.toFixed(3)}`);
}

// === Part 5: Exact duplicate audioTime (same time = batch) ===
console.log('\n=== Duplicate audioTime ===\n');
{
  const isDup = all.map((_,i) => {
    if(i > 0 && Math.abs(all[i].audioTime - all[i-1].audioTime) < 0.001) return 1;
    if(i < N-1 && Math.abs(all[i].audioTime - all[i+1].audioTime) < 0.001) return 1;
    return 0;
  });
  
  const userIdx = [], aiIdx = [];
  for(let i=0;i<N;i++) { if(act[i]) userIdx.push(i); else aiIdx.push(i); }
  
  const uDup = userIdx.filter(i=>isDup[i]).length;
  const aDup = aiIdx.filter(i=>isDup[i]).length;
  console.log(`Duplicate audioTime: User=${uDup}/${userIdx.length}(${(uDup/userIdx.length*100).toFixed(1)}%) AI=${aDup}/${aiIdx.length}(${(aDup/aiIdx.length*100).toFixed(1)}%)`);
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
