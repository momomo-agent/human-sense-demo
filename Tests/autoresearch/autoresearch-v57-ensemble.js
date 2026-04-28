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

// v57: OR ensemble — 两个独立分类器，任一判 user 就是 user
// Classifier A: 投票方案（高精度，R~90%）
// Classifier B: 连续值加权（不同决策边界，能捞 A 漏掉的）
console.log('=== v57: OR ensemble ===\n');

// Classifier A: v44 冠军
const classA = allSamples.map((s, i) => {
  let v = baseVotes(s, i);
  if (dtEnt5[i] >= 0.725) v += 1;
  if (burstLen[i] >= 3) v -= 0.25;
  if (s.score >= 0.3 && s.score < 0.7 && timeDelta[i] < 0.001 && s.jawVelocity >= 0.15) v -= 1.5;
  if (velStd5[i] >= 0.6 && timeDelta[i] < 0.001) v -= 0.625;
  if (scoreMean5[i] >= 0.65 && timeDelta[i] < 0.001) v -= 0.375;
  if (scoreVelAnti[i] >= 0.3) v += 0.375;
  return v >= 4;
});

// Classifier B: 连续值线性组合（不同特征权重）
for (const sw of [-2, -3, -4]) {  // score 权重（负=低 score 好）
  for (const vw of [3, 4, 5]) {   // vel 权重
    for (const dw of [1, 2, 3]) {  // dt 权重
      for (const ew of [1, 1.5, 2]) { // entropy 权重
        for (const bst of [0, -1, -2]) { // burst 惩罚
          for (const tB of [1, 1.5, 2, 2.5, 3]) { // B 的阈值
            const classB = allSamples.map((s, i) => {
              let v = s.score * sw + s.jawVelocity * vw + timeDelta[i] * dw + dtEnt5[i] * ew;
              if (burstLen[i] >= 3) v += bst;
              return v >= tB;
            });
            
            // OR: A || B
            const predsOR = allSamples.map((s, i) => classA[i] || classB[i]);
            const rOR = evaluate(predsOR);
            if (rOR.recall >= 0.95 && rOR.specificity >= 0.90) {
              results.push({ ...rOR, method: 'OR', params: `sw=${sw} vw=${vw} dw=${dw} ew=${ew} bst=${bst} tB=${tB}` });
            }
            
            // AND-boost: A || (B && 有一些 user 信号)
            for (const minBase of [1, 2]) {
              const predsAB = allSamples.map((s, i) => {
                if (classA[i]) return true;
                if (classB[i] && baseVotes(s, i) >= minBase) return true;
                return false;
              });
              const rAB = evaluate(predsAB);
              if (rAB.recall >= 0.95 && rAB.specificity >= 0.90) {
                results.push({ ...rAB, method: 'AND-boost', params: `sw=${sw} vw=${vw} dw=${dw} ew=${ew} bst=${bst} tB=${tB} minBase=${minBase}` });
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
  console.log('=== Top 20 by F1 (去重) ===\n');
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
} else {
  console.log('没有达标。分析 Classifier B 单独表现...\n');
  // 看 B 单独能到什么水平
  const bResults = [];
  for (const sw of [-2, -3, -4]) {
    for (const vw of [3, 4, 5]) {
      for (const dw of [1, 2, 3]) {
        for (const ew of [1, 1.5, 2]) {
          for (const tB of [0.5, 1, 1.5, 2, 2.5, 3]) {
            const classB = allSamples.map((s, i) => {
              let v = s.score * sw + s.jawVelocity * vw + timeDelta[i] * dw + dtEnt5[i] * ew;
              return v >= tB;
            });
            const r = evaluate(classB);
            bResults.push({ ...r, params: `sw=${sw} vw=${vw} dw=${dw} ew=${ew} tB=${tB}` });
          }
        }
      }
    }
  }
  bResults.sort((a, b) => b.recall - a.recall);
  console.log('B 单独最高 Recall (S≥85%):');
  for (const r of bResults.filter(r => r.specificity >= 0.85).slice(0, 10)) {
    console.log(`  R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} ${r.params}`);
  }
  
  // OR 最接近的
  console.log('\nOR 最接近 R≥95% S≥90%:');
  const orAll = [];
  for (const sw of [-3, -4]) {
    for (const vw of [4, 5]) {
      for (const dw of [2, 3]) {
        for (const ew of [1.5, 2]) {
          for (const tB of [2, 2.5, 3, 3.5]) {
            const classB = allSamples.map((s, i) => {
              let v = s.score * sw + s.jawVelocity * vw + timeDelta[i] * dw + dtEnt5[i] * ew;
              return v >= tB;
            });
            const predsOR = allSamples.map((s, i) => classA[i] || classB[i]);
            const r = evaluate(predsOR);
            orAll.push({ ...r, params: `sw=${sw} vw=${vw} dw=${dw} ew=${ew} tB=${tB}` });
          }
        }
      }
    }
  }
  orAll.sort((a, b) => (a.recall >= 0.95 && a.specificity >= 0.9 ? 0 : 1) - (b.recall >= 0.95 && b.specificity >= 0.9 ? 0 : 1) || b.f1 - a.f1);
  for (const r of orAll.slice(0, 10)) {
    console.log(`  R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% TP:${r.TP} FP:${r.FP} FN:${r.FN} ${r.params}`);
  }
}
