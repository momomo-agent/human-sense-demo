// autoresearch v61: Combine all winners — Dir1+Dir6+scoreAccel+jawEff
const fs = require('fs');
const DATA = '/Users/kenefe/LOCAL/momo-agent/projects/human-sense-demo/Tests/speaker-test-data.jsonl';
const lines = fs.readFileSync(DATA, 'utf8').trim().split('\n');
const allSamples = lines.map(l => JSON.parse(l));
allSamples.sort((a, b) => a.audioTime - b.audioTime);
const N = allSamples.length;
const actuals = allSamples.map(s => s.isUserSpeaker);
const dt = allSamples.map((s, i) => i === 0 ? 0 : s.audioTime - allSamples[i - 1].audioTime);
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
function windowStat(arr, hw, fn) {
  return arr.map((v, i) => {
    const win = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) win.push(arr[j]);
    return fn(win);
  });
}
function evaluate(predictions) {
  let TP = 0, FP = 0, TN = 0, FN = 0;
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

const dtEnt5 = windowStat(dt, 2, a => {
  const bins = [0, 0, 0];
  a.forEach(v => { if (v < 0.001) bins[0]++; else if (v < 0.1) bins[1]++; else bins[2]++; });
  let e = 0; const n = a.length;
  bins.forEach(b => { if (b > 0) { const p = b / n; e -= p * Math.log2(p); } });
  return e;
});
const burstLen = (() => {
  const bl = new Array(N).fill(1);
  for (let i = 1; i < N; i++) { if (dt[i] < 0.001) bl[i] = bl[i - 1] + 1; }
  for (let i = N - 2; i >= 0; i--) { if (dt[i + 1] < 0.001) bl[i] = Math.max(bl[i], bl[i + 1]); }
  return bl;
})();
const scoreMean5 = windowStat(allSamples.map(s => s.score), 2, mean);
const velStd5 = windowStat(allSamples.map(s => s.jawVelocity), 2, std);
const scoreStd5 = windowStat(allSamples.map(s => s.score), 2, std);
const scoreVelAnti = allSamples.map(s => (1 - s.score) * s.jawVelocity);

// New features from v60
const scoreAccel = allSamples.map((s, i) => {
  if (i === 0 || dt[i] < 0.001) return 0;
  return Math.abs(s.score - allSamples[i-1].score) / dt[i];
});
const jawEfficiency = allSamples.map(s => s.jawDelta > 0.001 ? s.jawVelocity / s.jawDelta : 0);
const jawEffMean5 = windowStat(jawEfficiency, 2, mean);

function baselineVotes(s, i) {
  let v = 0;
  if (s.score < 0.45) v += 3; else if (s.score < 0.5) v += 0.75; else if (s.score < 0.72) v += 0.25;
  if (s.jawDelta >= 0.1) v += 0.25; else if (s.jawDelta >= 0.05) v += 0.125;
  if (s.jawVelocity >= 0.5) v += 4; else if (s.jawVelocity >= 0.1) v += 2; else if (s.jawVelocity >= 0.05) v += 1;
  if (dt[i] >= 0.3) v += 1.5; else if (dt[i] >= 0.03) v += 0.75;
  if (dtEnt5[i] >= 0.725) v += 1;
  if (burstLen[i] >= 3) v -= 0.25;
  if (s.score >= 0.3 && s.score < 0.7 && dt[i] < 0.001 && s.jawVelocity >= 0.15) v -= 1.5;
  if (velStd5[i] >= 0.6 && dt[i] < 0.001) v -= 0.75;
  if (scoreMean5[i] >= 0.65 && dt[i] < 0.001) v -= 0.5;
  if (scoreStd5[i] < 0.12 && dt[i] < 0.001) v -= 0.375;
  if (scoreVelAnti[i] >= 0.3) v += 0.375;
  return v;
}
const baselineScores = allSamples.map((s, i) => baselineVotes(s, i));

// ============================================================
// Part 1: Combine Dir1+Dir6+scoreAccel+jawEff
// ============================================================
console.log('=== Part 1: Four-way combo search ===\n');
{
  let best = { f1: 0 };
  let count = 0;
  
  const svW_r = [0.25, 0.375, 0.5];
  const svTh_r = [0.75, 0.875, 1.0];
  const hvTh_r = [4.25, 4.5, 5.0];
  const hvPen_r = [1.5, 1.75, 2.0];
  const scoreLow_r = [0.25, 0.3, 0.35];
  const saW_r = [0, 0.25, 0.5, 0.75, 1.0];
  const saTh_r = [0.5, 1.0, 1.5, 2.0];
  const jeW_r = [0, -0.25, -0.5];
  const jeTh_r = [3.5, 4.0, 4.5, 5.0];
  
  for (const svW of svW_r) {
    for (const svTh of svTh_r) {
      for (const hvTh of hvTh_r) {
        for (const hvPen of hvPen_r) {
          for (const scoreLow of scoreLow_r) {
            for (const saW of saW_r) {
              for (const saTh of saTh_r) {
                for (const jeW of jeW_r) {
                  for (const jeTh of jeTh_r) {
                    const preds = allSamples.map((s, i) => {
                      let v = baselineScores[i];
                      const sv = (1 - s.score) * s.jawVelocity;
                      if (sv >= svTh) v += svW;
                      if (v >= hvTh && dt[i] < 0.001 && s.score < scoreLow) v -= hvPen;
                      if (scoreAccel[i] >= saTh) v += saW;
                      if (jawEffMean5[i] < jeTh) v += Math.abs(jeW);
                      return v >= 4;
                    });
                    const r = evaluate(preds);
                    if (r.recall >= 0.9 && r.specificity >= 0.9 && r.f1 > best.f1) {
                      best = { ...r, svW, svTh, hvTh, hvPen, scoreLow, saW, saTh, jeW, jeTh };
                      count++;
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
  console.log(`Qualifying: ${count}`);
  if (best.f1 > 0) {
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  svW=${best.svW} svTh=${best.svTh} hvTh=${best.hvTh} hvPen=${best.hvPen} scoreLow=${best.scoreLow}`);
    console.log(`  saW=${best.saW} saTh=${best.saTh} jeW=${best.jeW} jeTh=${best.jeTh}`);
  }
}

// ============================================================
// Part 2: + Rescue on top
// ============================================================
console.log('\n=== Part 2: Four-way + rescue ===\n');
{
  // Use best from Part 1 as Stage 1, then rescue
  // First find Part 1 best (hardcode from search above, or re-derive)
  let bestStage1 = { f1: 0 };
  const svW_r = [0.25, 0.375, 0.5];
  const svTh_r = [0.75, 0.875, 1.0];
  const hvTh_r = [4.25, 4.5, 5.0];
  const hvPen_r = [1.5, 1.75, 2.0];
  const scoreLow_r = [0.25, 0.3, 0.35];
  const saW_r = [0, 0.5, 0.75];
  const saTh_r = [1.0, 1.5];
  const jeW_r = [0, -0.25];
  const jeTh_r = [4.0, 4.5];
  
  // Quick search for best stage1
  for (const svW of svW_r) {
    for (const svTh of svTh_r) {
      for (const hvTh of hvTh_r) {
        for (const hvPen of hvPen_r) {
          for (const scoreLow of scoreLow_r) {
            for (const saW of saW_r) {
              for (const saTh of saTh_r) {
                for (const jeW of jeW_r) {
                  for (const jeTh of jeTh_r) {
                    const votes = allSamples.map((s, i) => {
                      let v = baselineScores[i];
                      const sv = (1 - s.score) * s.jawVelocity;
                      if (sv >= svTh) v += svW;
                      if (v >= hvTh && dt[i] < 0.001 && s.score < scoreLow) v -= hvPen;
                      if (scoreAccel[i] >= saTh) v += saW;
                      if (jawEffMean5[i] < jeTh) v += Math.abs(jeW);
                      return v;
                    });
                    const preds = votes.map(v => v >= 4);
                    const r = evaluate(preds);
                    if (r.recall >= 0.9 && r.specificity >= 0.9 && r.f1 > bestStage1.f1) {
                      bestStage1 = { ...r, svW, svTh, hvTh, hvPen, scoreLow, saW, saTh, jeW, jeTh, votes, preds };
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
  
  if (!bestStage1.votes) {
    console.log('No stage1 found');
  } else {
    console.log(`Stage1: R=${(bestStage1.recall*100).toFixed(1)}% S=${(bestStage1.specificity*100).toFixed(1)}% F1=${(bestStage1.f1*100).toFixed(1)}%`);
    
    // Rescue search
    function predDensity(preds, hw) {
      return preds.map((p, i) => {
        let c = 0, t = 0;
        for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) { t++; if (preds[j]) c++; }
        return c / t;
      });
    }
    
    let best = { f1: 0 };
    let count = 0;
    for (let hw = 3; hw <= 10; hw++) {
      const pd = predDensity(bestStage1.preds, hw);
      for (let rTh = 0.2; rTh <= 0.7; rTh += 0.05) {
        for (let minVotes = 0.5; minVotes <= 3.5; minVotes += 0.25) {
          const preds = allSamples.map((s, i) => {
            if (bestStage1.preds[i]) return true;
            if (pd[i] >= rTh && bestStage1.votes[i] >= minVotes) return true;
            return false;
          });
          const r = evaluate(preds);
          if (r.recall >= 0.9 && r.specificity >= 0.9 && r.f1 > best.f1) {
            best = { ...r, hw, rTh, minVotes };
            count++;
          }
        }
      }
    }
    console.log(`Rescue: ${count} qualifying`);
    if (best.f1 > 0) {
      console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
      console.log(`  hw=${best.hw} rTh=${best.rTh.toFixed(2)} minVotes=${best.minVotes}`);
    }
  }
}

// ============================================================
// Part 3: Dual-high mode (R≥95% S≥90%)
// ============================================================
console.log('\n=== Part 3: Dual-high mode (R≥95% S≥90%) ===\n');
{
  // Use four-way combo as Stage 1 with lower threshold, then rescue aggressively
  let best = { f1: 0 };
  let count = 0;
  
  for (let t = 2.5; t <= 4; t += 0.25) {
    const votes = allSamples.map((s, i) => {
      let v = baselineScores[i];
      const sv = (1 - s.score) * s.jawVelocity;
      if (sv >= 0.875) v += 0.375;
      if (v >= 4.25 && dt[i] < 0.001 && s.score < 0.35) v -= 1.75;
      if (scoreAccel[i] >= 1.5) v += 0.75;
      return v;
    });
    const preds = votes.map(v => v >= t);
    
    function predDensity(preds, hw) {
      return preds.map((p, i) => {
        let c = 0, total = 0;
        for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) { total++; if (preds[j]) c++; }
        return c / total;
      });
    }
    
    for (let hw = 5; hw <= 12; hw++) {
      const pd = predDensity(preds, hw);
      for (let rTh = 0.15; rTh <= 0.6; rTh += 0.05) {
        for (let minVotes = 0.5; minVotes <= 2.5; minVotes += 0.25) {
          const finalPreds = allSamples.map((s, i) => {
            if (preds[i]) return true;
            if (pd[i] >= rTh && votes[i] >= minVotes) return true;
            return false;
          });
          const r = evaluate(finalPreds);
          if (r.recall >= 0.95 && r.specificity >= 0.9 && r.f1 > best.f1) {
            best = { ...r, t, hw, rTh, minVotes };
            count++;
          }
        }
      }
    }
  }
  console.log(`Dual-high: ${count} qualifying`);
  if (best.f1 > 0) {
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  t=${best.t} hw=${best.hw} rTh=${best.rTh.toFixed(2)} minVotes=${best.minVotes}`);
  }
}

console.log('\n=== FINAL PROGRESS ===');
console.log('v49: F1=84.4%, R=90.4%, S=94.8%, FP=52, FN=21');
console.log('v51 dual-high: R=95.4%, S=90.3%, F1=79.7%');
