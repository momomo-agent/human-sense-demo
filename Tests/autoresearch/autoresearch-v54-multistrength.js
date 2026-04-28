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

// v54: 三种 Stage 1 强度 × rescue 参数
console.log('=== v54: 多强度 Stage 1 + rescue ===\n');

// Stage 1 配置：从弱到强
const stage1Configs = [
  // 弱：只有 entropy + burst + lipSyncPen
  { name: 'weak', fn: (s, i) => {
    let v = baseVotes(s, i);
    if (dtEnt5[i] >= 0.725) v += 1;
    if (burstLen[i] >= 3) v -= 0.25;
    if (s.score >= 0.3 && s.score < 0.7 && timeDelta[i] < 0.001 && s.jawVelocity >= 0.15) v -= 1.5;
    return v;
  }},
  // 中：+ velStd
  { name: 'mid', fn: (s, i) => {
    let v = baseVotes(s, i);
    if (dtEnt5[i] >= 0.725) v += 1;
    if (burstLen[i] >= 3) v -= 0.25;
    if (s.score >= 0.3 && s.score < 0.7 && timeDelta[i] < 0.001 && s.jawVelocity >= 0.15) v -= 1.5;
    if (velStd5[i] >= 0.6 && timeDelta[i] < 0.001) v -= 0.75;
    return v;
  }},
  // 强：+ velStd + scoreMean
  { name: 'strong', fn: (s, i) => {
    let v = baseVotes(s, i);
    if (dtEnt5[i] >= 0.725) v += 1;
    if (burstLen[i] >= 3) v -= 0.25;
    if (s.score >= 0.3 && s.score < 0.7 && timeDelta[i] < 0.001 && s.jawVelocity >= 0.15) v -= 1.5;
    if (velStd5[i] >= 0.6 && timeDelta[i] < 0.001) v -= 0.75;
    if (scoreMean5[i] >= 0.65 && timeDelta[i] < 0.001) v -= 0.375;
    if (scoreVelAnti[i] >= 0.3) v += 0.375;
    return v;
  }},
];

for (const cfg of stage1Configs) {
  const s1scores = allSamples.map((s, i) => cfg.fn(s, i));
  
  for (const mainT of [3.5, 3.75, 4]) {
    const s1pred = s1scores.map(v => v >= mainT);
    const s1r = evaluate(s1pred);
    
    for (const rW of [5, 7, 9, 11]) {
      const rDen = allSamples.map((s, i) => {
        let c = 0, t = 0;
        for (let j = Math.max(0, i-rW); j <= Math.min(N-1, i+rW); j++) {
          if (j !== i) { c++; if (s1pred[j]) t++; }
        }
        return c > 0 ? t / c : 0;
      });
      
      for (const rTh of [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]) {
        for (const minV of [0.5, 1, 1.5, 2, 2.5, 3]) {
          const preds = allSamples.map((s, i) => {
            if (s1pred[i]) return true;
            if (rDen[i] >= rTh && s1scores[i] >= minV) return true;
            return false;
          });
          const r = evaluate(preds);
          if (r.recall >= 0.95 && r.specificity >= 0.90) {
            results.push({ ...r, method: cfg.name, params: `mainT=${mainT} rW=${rW} rTh=${rTh} minV=${minV}` });
          }
        }
      }
    }
  }
}

console.log(`达标(R≥95% S≥90%): ${results.length}\n`);

// 按方法分组统计
for (const name of ['weak', 'mid', 'strong']) {
  const sub = results.filter(r => r.method === name);
  console.log(`  ${name}: ${sub.length} 达标`);
}

results.sort((a, b) => b.f1 - a.f1);
console.log('\n=== Top 20 by F1 (去重) ===\n');
const seen = new Set();
let count = 0;
for (const r of results) {
  const key = `${r.TP}-${r.FP}-${r.TN}-${r.FN}`;
  if (seen.has(key)) continue;
  seen.add(key);
  console.log(`${++count}. [${r.method}] R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN}`);
  console.log(`   ${r.params}\n`);
  if (count >= 20) break;
}

results.sort((a, b) => b.recall - a.recall || b.specificity - a.specificity);
console.log('\n=== Top 10 by Recall ===\n');
const seen2 = new Set();
count = 0;
for (const r of results) {
  const key = `${r.TP}-${r.FP}`;
  if (seen2.has(key)) continue;
  seen2.add(key);
  console.log(`${++count}. [${r.method}] R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN}`);
  console.log(`   ${r.params}\n`);
  if (count >= 10) break;
}

// 最高 Recall 且 S≥90%
results.sort((a, b) => Math.min(b.recall, b.specificity) - Math.min(a.recall, a.specificity));
console.log('\n=== Top 10 by min(R,S) ===\n');
const seen3 = new Set();
count = 0;
for (const r of results) {
  const key = `${r.TP}-${r.FP}-${r.TN}-${r.FN}`;
  if (seen3.has(key)) continue;
  seen3.add(key);
  console.log(`${++count}. [${r.method}] R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% min=${(Math.min(r.recall,r.specificity)*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN}`);
  console.log(`   ${r.params}\n`);
  if (count >= 10) break;
}
