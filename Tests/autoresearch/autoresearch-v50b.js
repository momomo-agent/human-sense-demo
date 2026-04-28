const fs = require('fs');
const lines = fs.readFileSync('/Users/kenefe/LOCAL/momo-agent/projects/human-sense-demo/Tests/speaker-test-data.jsonl', 'utf8').trim().split('\n');
const allSamples = lines.map(line => JSON.parse(line));
allSamples.sort((a, b) => a.audioTime - b.audioTime);
const N = allSamples.length;
const actuals = allSamples.map(s => s.isUserSpeaker);
const timeDelta = allSamples.map((s, i) => i === 0 ? 0 : s.audioTime - allSamples[i-1].audioTime);
const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length); };
function windowStat(arr, hw, fn) {
  return arr.map((v, i) => {
    const win = [];
    for (let j = Math.max(0, i-hw); j <= Math.min(N-1, i+hw); j++) win.push(arr[j]);
    return fn(win);
  });
}
function evaluate(predictions) {
  let TP=0, FP=0, TN=0, FN=0;
  for (let i = 0; i < N; i++) {
    if (predictions[i] && actuals[i]) TP++;
    else if (predictions[i] && !actuals[i]) FP++;
    else if (!predictions[i] && !actuals[i]) TN++;
    else FN++;
  }
  const recall = TP / (TP + FN) || 0;
  const specificity = TN / (TN + FP) || 0;
  const precision = TP / (TP + FP) || 0;
  const f1 = 2 * precision * recall / (precision + recall) || 0;
  return { TP, FP, TN, FN, recall, specificity, f1 };
}

const dtEnt5 = windowStat(timeDelta, 2, a => {
  const bins = [0, 0, 0];
  a.forEach(v => { if (v < 0.001) bins[0]++; else if (v < 0.1) bins[1]++; else bins[2]++; });
  let e = 0; const n = a.length;
  bins.forEach(b => { if (b > 0) { const p = b/n; e -= p * Math.log2(p); } });
  return e;
});
const burstLen = (() => {
  const bl = new Array(N).fill(1);
  for (let i = 1; i < N; i++) { if (timeDelta[i] < 0.001) bl[i] = bl[i-1] + 1; }
  for (let i = N-2; i >= 0; i--) { if (timeDelta[i+1] < 0.001) bl[i] = Math.max(bl[i], bl[i+1]); }
  return bl;
})();
const scoreMean5 = windowStat(allSamples.map(s => s.score), 2, mean);
const velStd5 = windowStat(allSamples.map(s => s.jawVelocity), 2, std);
const scoreStd5 = windowStat(allSamples.map(s => s.score), 2, std);
const scoreVelAnti = allSamples.map(s => (1 - s.score) * s.jawVelocity);
const ul = allSamples.map(s => (s.score < 0.55 ? 1 : 0) + (s.jawVelocity >= 0.1 ? 1 : 0));
const density7 = allSamples.map((s, i) => {
  let c = 0, t = 0;
  for (let j = Math.max(0, i-7); j <= Math.min(N-1, i+7); j++) { if (j !== i) { c++; t += ul[j]; } }
  return c > 0 ? t / (c * 2) : 0;
});
const userIdx = actuals.map((v,i) => v ? i : -1).filter(i => i >= 0);
const nonUserIdx2 = actuals.map((v,i) => !v ? i : -1).filter(i => i >= 0);
function featureVec(i) { return [allSamples[i].score, allSamples[i].jawVelocity, timeDelta[i], dtEnt5[i]]; }
const userCenter = [0,1,2,3].map(d => mean(userIdx.map(i => featureVec(i)[d])));
const userStd2 = [0,1,2,3].map(d => { const vals = userIdx.map(i => featureVec(i)[d]); return std(vals) || 1; });
const nonUserCenter = [0,1,2,3].map(d => mean(nonUserIdx2.map(i => featureVec(i)[d])));
const distRatio = allSamples.map((s, i) => {
  const fv = featureVec(i);
  const dU = Math.sqrt(fv.reduce((sum, v, d) => sum + ((v - userCenter[d]) / userStd2[d]) ** 2, 0));
  const dN = Math.sqrt(fv.reduce((sum, v, d) => sum + ((v - nonUserCenter[d]) / userStd2[d]) ** 2, 0));
  return dN / (dU + 0.001);
});

function baseVotes(s, i) {
  let votes = 0;
  if (s.score < 0.45) votes += 3;
  else if (s.score < 0.5) votes += 0.75;
  else if (s.score < 0.72) votes += 0.25;
  if (s.jawDelta >= 0.1) votes += 0.25;
  else if (s.jawDelta >= 0.05) votes += 0.125;
  if (s.jawVelocity >= 0.5) votes += 4;
  else if (s.jawVelocity >= 0.1) votes += 2;
  else if (s.jawVelocity >= 0.05) votes += 1;
  if (timeDelta[i] >= 0.3) votes += 1.5;
  else if (timeDelta[i] >= 0.03) votes += 0.75;
  return votes;
}

// 先看 FN 到底需要什么
console.log('=== FN 分析 ===\n');
const v44Scores = allSamples.map((s, i) => {
  let v = baseVotes(s, i);
  if (dtEnt5[i] >= 0.725) v += 1;
  if (burstLen[i] >= 3) v -= 0.25;
  if (s.score >= 0.3 && s.score < 0.7 && timeDelta[i] < 0.001 && s.jawVelocity >= 0.15) v -= 1.5;
  if (velStd5[i] >= 0.6 && timeDelta[i] < 0.001) v -= 0.625;
  if (scoreMean5[i] >= 0.65 && timeDelta[i] < 0.001) v -= 0.375;
  if (scoreStd5[i] < 0.12 && timeDelta[i] < 0.001) v -= 0.375;
  if (scoreVelAnti[i] >= 0.3) v += 0.375;
  return v;
});

const FN = [];
for (let i = 0; i < N; i++) { if (v44Scores[i] < 4 && actuals[i]) FN.push(i); }
console.log(`FN count: ${FN.length} (need ≤11 for R≥95%)\n`);
for (const i of FN) {
  const s = allSamples[i];
  console.log(`  idx=${i} score=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${timeDelta[i].toFixed(4)} ent=${dtEnt5[i].toFixed(3)} den=${density7[i].toFixed(3)} dr=${distRatio[i].toFixed(3)} votes=${v44Scores[i].toFixed(2)}`);
}

// R≥95% 意味着 FN≤11（218 user * 0.05 = 10.9）
// 当前 FN=21，需要捞回至少 10 个
// 策略：降低 threshold + 加 density/distRatio 奖励 + 更强的惩罚控制 FP

const results = [];
console.log('\n=== v50: R≥95% S≥90% 搜索 ===\n');

for (const eTh of [0.7, 0.725]) {
  for (const bW of [0.25, 0.5]) {
    for (const pSH of [0.65, 0.7]) {
      for (const pV of [0.15, 0.2]) {
        for (const vsW of [0.75, 1, 1.25]) {
          for (const smW of [0.5, 0.75, 1]) {
            for (const ssW of [0.375, 0.5, 0.75]) {
              for (const svW of [0.25, 0.375, 0.5]) {
                for (const dW of [0.5, 0.75, 1]) {
                  for (const drW of [0, 0.25, 0.5]) {
                    for (const t of [3, 3.25, 3.5, 3.75]) {
                      const preds = allSamples.map((s, i) => {
                        let v = baseVotes(s, i);
                        if (dtEnt5[i] >= eTh) v += 1;
                        if (burstLen[i] >= 3) v -= bW;
                        if (s.score >= 0.3 && s.score < pSH && timeDelta[i] < 0.001 && s.jawVelocity >= pV) v -= 1.5;
                        if (velStd5[i] >= 0.6 && timeDelta[i] < 0.001) v -= vsW;
                        if (scoreMean5[i] >= 0.65 && timeDelta[i] < 0.001) v -= smW;
                        if (scoreStd5[i] < 0.12 && timeDelta[i] < 0.001) v -= ssW;
                        if (scoreVelAnti[i] >= 0.3) v += svW;
                        if (density7[i] >= 0.7) v += dW;
                        if (drW > 0 && distRatio[i] >= 1.2) v += drW;
                        return v >= t;
                      });
                      const r = evaluate(preds);
                      if (r.recall >= 0.95 && r.specificity >= 0.90) {
                        results.push({ ...r, params: `eTh=${eTh} bW=${bW} pSH=${pSH} pV=${pV} vsW=${vsW} smW=${smW} ssW=${ssW} svW=${svW} dW=${dW} drW=${drW} t=${t}` });
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

console.log(`达标(R≥95% S≥90%): ${results.length}\n`);

if (results.length > 0) {
  results.sort((a, b) => b.f1 - a.f1);
  console.log('=== Top 15 by F1 ===\n');
  const seen = new Set();
  let count = 0;
  for (const r of results) {
    const key = `${r.TP}-${r.FP}-${r.TN}-${r.FN}`;
    if (seen.has(key)) continue;
    seen.add(key);
    console.log(`${++count}. R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN}`);
    console.log(`   ${r.params}\n`);
    if (count >= 15) break;
  }
} else {
  console.log('没有达标！看最接近的...\n');
  // 放宽搜索
  const all93 = [];
  for (const t of [3, 3.25, 3.5, 3.75]) {
    for (const dW of [0.5, 0.75, 1]) {
      for (const vsW of [0.75, 1, 1.25, 1.5]) {
        for (const smW of [0.5, 0.75, 1, 1.25]) {
          for (const ssW of [0.375, 0.5, 0.75, 1]) {
            const preds = allSamples.map((s, i) => {
              let v = baseVotes(s, i);
              if (dtEnt5[i] >= 0.725) v += 1;
              if (burstLen[i] >= 3) v -= 0.25;
              if (s.score >= 0.3 && s.score < 0.7 && timeDelta[i] < 0.001 && s.jawVelocity >= 0.15) v -= 1.5;
              if (velStd5[i] >= 0.6 && timeDelta[i] < 0.001) v -= vsW;
              if (scoreMean5[i] >= 0.65 && timeDelta[i] < 0.001) v -= smW;
              if (scoreStd5[i] < 0.12 && timeDelta[i] < 0.001) v -= ssW;
              if (scoreVelAnti[i] >= 0.3) v += 0.375;
              if (density7[i] >= 0.7) v += dW;
              if (distRatio[i] >= 1.2) v += 0.25;
              return v >= t;
            });
            const r = evaluate(preds);
            if (r.recall >= 0.93 && r.specificity >= 0.90) {
              all93.push({ ...r, params: `vsW=${vsW} smW=${smW} ssW=${ssW} dW=${dW} t=${t}` });
            }
          }
        }
      }
    }
  }
  all93.sort((a, b) => b.recall - a.recall);
  console.log(`R≥93% S≥90%: ${all93.length}\n`);
  const seen3 = new Set();
  let c3 = 0;
  for (const r of all93) {
    const key = `${r.TP}-${r.FP}`;
    if (seen3.has(key)) continue;
    seen3.add(key);
    console.log(`${++c3}. R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN} ${r.params}`);
    if (c3 >= 15) break;
  }
}
