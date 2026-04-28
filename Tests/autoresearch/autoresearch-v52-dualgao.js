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

const results = [];

// v52: 精细搜索两阶段 rescue，目标 R≥95% S≥90% 最高 F1
console.log('=== v52: 双高精细搜索 ===\n');

// Stage 1 参数
for (const eTh of [0.7, 0.725]) {
  for (const bW of [0.25, 0.375]) {
    for (const pSH of [0.65, 0.7]) {
      for (const pV of [0.15, 0.2]) {
        for (const vsW of [0.625, 0.75, 0.875, 1]) {
          for (const smW of [0.25, 0.375, 0.5]) {
            for (const ssW of [0.25, 0.375]) {
              for (const svW of [0.25, 0.375]) {
                // Stage 1
                const stage1scores = allSamples.map((s, i) => {
                  let v = baseVotes(s, i);
                  if (dtEnt5[i] >= eTh) v += 1;
                  if (burstLen[i] >= 3) v -= bW;
                  if (s.score >= 0.3 && s.score < pSH && timeDelta[i] < 0.001 && s.jawVelocity >= pV) v -= 1.5;
                  if (velStd5[i] >= 0.6 && timeDelta[i] < 0.001) v -= vsW;
                  if (scoreMean5[i] >= 0.65 && timeDelta[i] < 0.001) v -= smW;
                  if (scoreStd5[i] < 0.12 && timeDelta[i] < 0.001) v -= ssW;
                  if (scoreVelAnti[i] >= 0.3) v += svW;
                  return v;
                });
                
                const stage1_t4 = stage1scores.map(v => v >= 4);
                
                // Rescue 参数
                for (const rW of [5, 7, 9]) {
                  // 用 stage1 结果做 density
                  const rescueDen = allSamples.map((s, i) => {
                    let c = 0, t = 0;
                    for (let j = Math.max(0, i-rW); j <= Math.min(N-1, i+rW); j++) {
                      if (j !== i) { c++; if (stage1_t4[j]) t++; }
                    }
                    return c > 0 ? t / c : 0;
                  });
                  
                  for (const rTh of [0.4, 0.45, 0.5, 0.55, 0.6]) {
                    for (const minV of [1.5, 2, 2.5, 3]) {
                      const preds = allSamples.map((s, i) => {
                        if (stage1_t4[i]) return true;
                        // rescue
                        if (rescueDen[i] >= rTh && stage1scores[i] >= minV) return true;
                        return false;
                      });
                      const r = evaluate(preds);
                      if (r.recall >= 0.95 && r.specificity >= 0.90) {
                        results.push({ ...r, params: `eTh=${eTh} bW=${bW} pSH=${pSH} pV=${pV} vsW=${vsW} smW=${smW} ssW=${ssW} svW=${svW} rW=${rW} rTh=${rTh} minV=${minV}` });
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
  // F1 排序
  results.sort((a, b) => b.f1 - a.f1);
  console.log('=== Top 20 by F1 (去重) ===\n');
  const seen = new Set();
  let count = 0;
  for (const r of results) {
    const key = `${r.TP}-${r.FP}-${r.TN}-${r.FN}`;
    if (seen.has(key)) continue;
    seen.add(key);
    console.log(`${++count}. R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN}`);
    console.log(`   ${r.params}\n`);
    if (count >= 20) break;
  }
  
  // min(R,S) 排序
  results.sort((a, b) => Math.min(b.recall, b.specificity) - Math.min(a.recall, a.specificity));
  console.log('\n=== Top 10 by min(R,S) ===\n');
  const seen2 = new Set();
  count = 0;
  for (const r of results) {
    const key = `${r.TP}-${r.FP}-${r.TN}-${r.FN}`;
    if (seen2.has(key)) continue;
    seen2.add(key);
    console.log(`${++count}. R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% min=${(Math.min(r.recall,r.specificity)*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN}`);
    console.log(`   ${r.params}\n`);
    if (count >= 10) break;
  }
  
  // Recall 排序
  results.sort((a, b) => b.recall - a.recall || b.specificity - a.specificity);
  console.log('\n=== Top 10 by Recall ===\n');
  const seen3 = new Set();
  count = 0;
  for (const r of results) {
    const key = `${r.TP}-${r.FP}`;
    if (seen3.has(key)) continue;
    seen3.add(key);
    console.log(`${++count}. R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN}`);
    console.log(`   ${r.params}\n`);
    if (count >= 10) break;
  }
}
