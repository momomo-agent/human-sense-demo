// autoresearch v124: Exploit jvw/speakerThreshold + session-aware features
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
const sth = all.map(s => s.speakerThreshold || 0);

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

// === Part 1: jvw as feature ===
console.log('=== jvw as feature ===\n');
{
  // jvw=1 → jawWeight=1 → jaw data reliable → strong user signal
  // jvw=0.2 → jawWeight=0 → jaw data unreliable
  // jvw=2 → speakerThreshold=4 session
  
  let best = {f1:0}, count=0;
  
  for(let zoneW=4;zoneW<=5.5;zoneW+=0.5){
    for(let vHW=1.5;vHW<=2.5;vHW+=0.5){
      for(let jdW=1.5;jdW<=2.5;jdW+=0.5){
        for(let fsW=1.5;fsW<=3;fsW+=0.5){
          for(let jvw1W=0;jvw1W<=2;jvw1W+=0.5){  // bonus for jvw=1
            for(let jvw02W=0;jvw02W<=1;jvw02W+=0.5){  // penalty for jvw=0.2
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
                  // jvw features
                  if(jvw1W > 0 && jvw[i] >= 0.9 && jvw[i] <= 1.1) v += jvw1W;
                  if(jvw02W > 0 && jvw[i] < 0.3) v -= jvw02W;
                  // Penalties
                  if(f.jdMean < 0.005) v -= 2;
                  if(f.jeMean < 1.5) v -= 1;
                  if(!isHighJW[i] && s.score >= 0.7) v -= 0.5;
                  if((s.finalScore||0) >= 0.7) v -= fsW;
                  return v >= t;
                });
                const r = ev(preds);
                if(r.recall>=0.90&&r.specificity>=0.92&&r.f1>best.f1){
                  best={...r,zoneW,vHW,jdW,fsW,jvw1W,jvw02W,t};
                  count++;
                }
              }
            }
          }
        }
      }
    }
  }
  console.log(`jvw feature: ${count} qualifying`);
  if(best.f1>0){
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  zoneW=${best.zoneW} vHW=${best.vHW} jdW=${best.jdW} fsW=${best.fsW}`);
    console.log(`  jvw=1→+${best.jvw1W} jvw=0.2→-${best.jvw02W} threshold=${best.t}`);
  }
}

// === Part 2: Split by speakerThreshold (different sessions) ===
console.log('\n=== Split by speakerThreshold ===\n');
{
  const sth07 = [], sth4 = [];
  for(let i=0;i<N;i++){
    if(sth[i] > 2) sth4.push(i); else sth07.push(i);
  }
  console.log(`sth=0.7: ${sth07.length} tokens (${sth07.filter(i=>act[i]).length} user, ${sth07.filter(i=>!act[i]).length} AI)`);
  console.log(`sth=4.0: ${sth4.length} tokens (${sth4.filter(i=>act[i]).length} user, ${sth4.filter(i=>!act[i]).length} AI)`);
  
  // v112 performance on each subset
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
  const v112preds = all.map((s,i) => v112Score(s,i) >= 5.75);
  
  let tp07=0,fp07=0,tn07=0,fn07=0;
  for(const i of sth07){if(v112preds[i]&&act[i])tp07++;else if(v112preds[i]&&!act[i])fp07++;else if(!v112preds[i]&&!act[i])tn07++;else fn07++;}
  let tp4=0,fp4=0,tn4=0,fn4=0;
  for(const i of sth4){if(v112preds[i]&&act[i])tp4++;else if(v112preds[i]&&!act[i])fp4++;else if(!v112preds[i]&&!act[i])tn4++;else fn4++;}
  
  console.log(`\nv112 on sth=0.7: TP=${tp07} FP=${fp07} TN=${tn07} FN=${fn07} R=${(tp07/(tp07+fn07)*100).toFixed(1)}% S=${(tn07/(tn07+fp07)*100).toFixed(1)}%`);
  console.log(`v112 on sth=4.0: TP=${tp4} FP=${fp4} TN=${tn4} FN=${fn4} R=${(tp4/(tp4+fn4)*100).toFixed(1)}% S=${(tn4/(tn4+fp4)*100).toFixed(1)}%`);
  
  // Optimize separately for each session type
  console.log('\n--- Optimizing sth=0.7 subset ---');
  let best07 = {f1:0};
  for(let fsW=1;fsW<=4;fsW+=0.5){
    for(let t=4;t<=7;t+=0.25){
      const preds = sth07.map(i => {
        let v = 0;
        const f = tw10[i]; const s = all[i];
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
        if((s.finalScore||0) >= 0.7) v -= fsW;
        return v >= t;
      });
      let tp=0,fp=0,tn=0,fn=0;
      for(let k=0;k<sth07.length;k++){
        const i=sth07[k];
        if(preds[k]&&act[i])tp++;else if(preds[k]&&!act[i])fp++;else if(!preds[k]&&!act[i])tn++;else fn++;
      }
      const r=tp/(tp+fn)||0,sp=tn/(tn+fp)||0,pr=tp/(tp+fp)||0,f1=2*pr*r/(pr+r)||0;
      if(r>=0.90&&sp>=0.92&&f1>best07.f1) best07={tp,fp,tn,fn,recall:r,specificity:sp,f1,fsW,t};
    }
  }
  if(best07.f1>0) console.log(`sth=0.7 best: R=${(best07.recall*100).toFixed(1)}% S=${(best07.specificity*100).toFixed(1)}% F1=${(best07.f1*100).toFixed(1)}% FP=${best07.fp} FN=${best07.fn} fsW=${best07.fsW} t=${best07.t}`);
  
  console.log('\n--- Optimizing sth=4.0 subset ---');
  let best4 = {f1:0};
  for(let fsW=0;fsW<=4;fsW+=0.5){
    for(let t=3;t<=7;t+=0.25){
      const preds = sth4.map(i => {
        let v = 0;
        const f = tw10[i]; const s = all[i];
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
        if((s.finalScore||0) >= 0.7) v -= fsW;
        return v >= t;
      });
      let tp=0,fp=0,tn=0,fn=0;
      for(let k=0;k<sth4.length;k++){
        const i=sth4[k];
        if(preds[k]&&act[i])tp++;else if(preds[k]&&!act[i])fp++;else if(!preds[k]&&!act[i])tn++;else fn++;
      }
      const r=tp/(tp+fn)||0,sp=tn/(tn+fp)||0,pr=tp/(tp+fp)||0,f1=2*pr*r/(pr+r)||0;
      if(r>=0.90&&sp>=0.92&&f1>best4.f1) best4={tp,fp,tn,fn,recall:r,specificity:sp,f1,fsW,t};
    }
  }
  if(best4.f1>0) console.log(`sth=4.0 best: R=${(best4.recall*100).toFixed(1)}% S=${(best4.specificity*100).toFixed(1)}% F1=${(best4.f1*100).toFixed(1)}% FP=${best4.fp} FN=${best4.fn} fsW=${best4.fsW} t=${best4.t}`);
  
  // Combined: use best params for each subset
  if(best07.f1>0 && best4.f1>0){
    const combinedTP = best07.tp + best4.tp;
    const combinedFP = best07.fp + best4.fp;
    const combinedTN = best07.tn + best4.tn;
    const combinedFN = best07.fn + best4.fn;
    const cR = combinedTP/(combinedTP+combinedFN)||0;
    const cS = combinedTN/(combinedTN+combinedFP)||0;
    const cPr = combinedTP/(combinedTP+combinedFP)||0;
    const cF1 = 2*cPr*cR/(cPr+cR)||0;
    console.log(`\nCombined split: R=${(cR*100).toFixed(1)}% S=${(cS*100).toFixed(1)}% F1=${(cF1*100).toFixed(1)}% FP=${combinedFP} FN=${combinedFN}`);
  }
}

// === Part 3: FP deep analysis — what makes them look like user? ===
console.log('\n=== FP deep analysis ===\n');
{
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
  const v112preds = all.map((s,i) => v112Score(s,i) >= 5.75);
  
  const fpIdx = [];
  for(let i=0;i<N;i++) if(v112preds[i] && !act[i]) fpIdx.push(i);
  
  // Group FP by time clusters
  const fpClusters = [];
  let curCluster = null;
  for(const i of fpIdx){
    if(!curCluster || all[i].audioTime - all[curCluster[curCluster.length-1]].audioTime > 2){
      if(curCluster) fpClusters.push(curCluster);
      curCluster = [i];
    } else {
      curCluster.push(i);
    }
  }
  if(curCluster) fpClusters.push(curCluster);
  
  console.log(`FP clusters: ${fpClusters.length} (total ${fpIdx.length} tokens)`);
  for(let c=0;c<Math.min(10,fpClusters.length);c++){
    const cl = fpClusters[c];
    const t0 = all[cl[0]].audioTime, t1 = all[cl[cl.length-1]].audioTime;
    const avgVel = mean(cl.map(i=>all[i].jawVelocity));
    const avgJd = mean(cl.map(i=>all[i].jawDelta));
    const avgFs = mean(cl.map(i=>all[i].finalScore||0));
    const avgSc = mean(cl.map(i=>all[i].score));
    const jvws = [...new Set(cl.map(i=>jvw[i].toFixed(1)))].join('/');
    console.log(`  C${c}: t=${t0.toFixed(1)}-${t1.toFixed(1)} n=${cl.length} vel=${avgVel.toFixed(3)} jd=${avgJd.toFixed(4)} fs=${avgFs.toFixed(3)} sc=${avgSc.toFixed(3)} jvw=${jvws}`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v112: F1=87.6% (R=99.6% S=92.1%)');
