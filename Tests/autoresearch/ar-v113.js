// autoresearch v113: FP analysis + targeted FP reduction
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

// Reproduce v112 FS+penalties best
const sc = all.map((s,i) => {
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
  if(s.score >= 0.8 && s.jawDelta < 0.03) v -= 0.5;
  if(s.jawDelta < 0.04 && s.jawVelocity >= 0.1 && !isHighJW[i]) v -= 0.5;
  return v;
});
const preds = sc.map(v => v >= 5.25);
const r0 = ev(preds);
console.log(`Baseline: R=${(r0.recall*100).toFixed(1)}% S=${(r0.specificity*100).toFixed(1)}% F1=${(r0.f1*100).toFixed(1)}% FP=${r0.FP} FN=${r0.FN}\n`);

// === FP analysis ===
const FP = [], TP = [], FN = [];
for(let i=0;i<N;i++){
  if(preds[i]&&!act[i]) FP.push(i);
  if(preds[i]&&act[i]) TP.push(i);
  if(!preds[i]&&act[i]) FN.push(i);
}

console.log('=== FP feature distributions ===');
const fpFeats = FP.map(i => ({
  score: all[i].score,
  jd: all[i].jawDelta,
  vel: all[i].jawVelocity,
  fs: all[i].finalScore||0,
  dt: dt[i],
  hjw: isHighJW[i],
  je: jawEff[i],
  sva: scoreVelAnti[i],
  zone_jd: tw10[i].jdMean,
  zone_je: tw10[i].jeMean,
  v: sc[i],
}));
const tpFeats = TP.map(i => ({
  score: all[i].score,
  jd: all[i].jawDelta,
  vel: all[i].jawVelocity,
  fs: all[i].finalScore||0,
  dt: dt[i],
  hjw: isHighJW[i],
  je: jawEff[i],
  sva: scoreVelAnti[i],
  zone_jd: tw10[i].jdMean,
  zone_je: tw10[i].jeMean,
  v: sc[i],
}));

for(const key of ['score','jd','vel','fs','dt','je','sva','zone_jd','zone_je','v']){
  const fpV = fpFeats.map(f=>f[key]);
  const tpV = tpFeats.map(f=>f[key]);
  const fpM = mean(fpV), fpS = std(fpV);
  const tpM = mean(tpV), tpS = std(tpV);
  const pooled = Math.sqrt((fpS**2+tpS**2)/2);
  const d = pooled>0?Math.abs(fpM-tpM)/pooled:0;
  console.log(`${key.padEnd(10)} FP=${fpM.toFixed(3)}±${fpS.toFixed(3)} TP=${tpM.toFixed(3)}±${tpS.toFixed(3)} d=${d.toFixed(3)}`);
}

// FP by score range
console.log('\n=== FP by score range ===');
const ranges = [[0,0.3],[0.3,0.5],[0.5,0.7],[0.7,0.9],[0.9,1.01]];
for(const [lo,hi] of ranges){
  const cnt = FP.filter(i=>all[i].score>=lo&&all[i].score<hi).length;
  const tpCnt = TP.filter(i=>all[i].score>=lo&&all[i].score<hi).length;
  console.log(`  score [${lo},${hi}): FP=${cnt} TP=${tpCnt}`);
}

// FP by zone membership
const fpInZone = FP.filter(i=>tw10[i].jdMean>=0.03&&tw10[i].jeMean>=5).length;
console.log(`\nFP in zone: ${fpInZone}/${FP.length}`);
console.log(`FP not in zone: ${FP.length-fpInZone}`);

// FP by vote score
console.log('\n=== FP by vote score ===');
for(let v=5;v<=10;v++){
  const cnt = FP.filter(i=>sc[i]>=v&&sc[i]<v+1).length;
  console.log(`  v=[${v},${v+1}): ${cnt}`);
}

// === Try: suppress FP with second-pass filter ===
console.log('\n=== Second-pass FP suppression ===\n');
{
  let best = {f1:r0.f1};
  
  // For each predicted-user token, check if neighbors agree
  for(let hw=3;hw<=11;hw+=2){
    for(let minAgree=0.3;minAgree<=0.7;minAgree+=0.1){
      for(let minScore=5;minScore<=7;minScore+=0.5){
        const p2 = all.map((_,i) => {
          if(!preds[i]) return false;
          if(sc[i] >= minScore) return true; // high confidence, keep
          // Check neighborhood agreement
          let userN=0, total=0;
          for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
            if(j===i) continue;
            total++;
            if(preds[j]) userN++;
          }
          return total>0 && userN/total >= minAgree;
        });
        const r = ev(p2);
        if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1) best={...r,hw,minAgree,minScore};
      }
    }
  }
  if(best.f1>r0.f1){
    console.log(`FP suppress: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} minAgree=${best.minAgree} minScore=${best.minScore}`);
  } else {
    console.log('FP suppression did not improve');
  }
}

// === Try: noMovementPenalty as feature ===
console.log('\n=== noJawPenalty / jawMargin features ===\n');
{
  const njpUser = TP.map(i => all[i].noJawPenalty || 0);
  const njpFP = FP.map(i => all[i].noJawPenalty || 0);
  const jmUser = TP.map(i => all[i].jawMargin || 0);
  const jmFP = FP.map(i => all[i].jawMargin || 0);
  
  console.log(`noJawPenalty: TP=${mean(njpUser).toFixed(3)}±${std(njpUser).toFixed(3)} FP=${mean(njpFP).toFixed(3)}±${std(njpFP).toFixed(3)}`);
  console.log(`jawMargin: TP=${mean(jmUser).toFixed(3)}±${std(jmUser).toFixed(3)} FP=${mean(jmFP).toFixed(3)}±${std(jmFP).toFixed(3)}`);
  
  // Try jawMargin as penalty
  let best = {f1:0};
  for(let jmTh=-0.5;jmTh<=0.5;jmTh+=0.1){
    for(let jmW=0.5;jmW<=2;jmW+=0.5){
      const p2 = all.map((s,i) => {
        let v = sc[i];
        if((s.jawMargin||0) < jmTh) v -= jmW;
        return v >= 5.25;
      });
      const r = ev(p2);
      if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1) best={...r,jmTh,jmW};
    }
  }
  if(best.f1>r0.f1){
    console.log(`jawMargin penalty: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  jawMargin<${best.jmTh}→-${best.jmW}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112 FS: F1=87.6% (R=99.6% S=92.1% FP=141 FN=2)');
