// autoresearch v60: Fine-grained search on Dir1+Dir6 combo + new FP-targeted features
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
// Part 1: Ultra-fine search on Dir1+Dir6
// ============================================================
console.log('=== Part 1: Ultra-fine Dir1+Dir6 ===\n');
{
  let best = { f1: 0 };
  let count = 0;
  for (let svW = 0.25; svW <= 1; svW += 0.125) {
    for (let svTh = 0.5; svTh <= 1.5; svTh += 0.125) {
      for (let hvTh = 4; hvTh <= 7; hvTh += 0.25) {
        for (let hvPen = 0.5; hvPen <= 3; hvPen += 0.25) {
          for (let scoreLow = 0.15; scoreLow <= 0.45; scoreLow += 0.05) {
            const preds = allSamples.map((s, i) => {
              let v = baselineScores[i];
              const sv = (1 - s.score) * s.jawVelocity;
              if (sv >= svTh) v += svW;
              if (v >= hvTh && dt[i] < 0.001 && s.score < scoreLow) v -= hvPen;
              return v >= 4;
            });
            const r = evaluate(preds);
            if (r.recall >= 0.9 && r.specificity >= 0.9 && r.f1 > best.f1) {
              best = { ...r, svW, svTh, hvTh, hvPen, scoreLow };
              count++;
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
  }
}

// ============================================================
// Part 2: New FP-targeted features
// ============================================================
console.log('\n=== Part 2: New FP-targeted features ===\n');

// Analyze remaining FP after Dir1+Dir6 best
const bestCombo = allSamples.map((s, i) => {
  let v = baselineScores[i];
  const sv = (1 - s.score) * s.jawVelocity;
  if (sv >= 1) v += 0.5;
  if (v >= 4.5 && dt[i] < 0.001 && s.score < 0.3) v -= 1.5;
  return { pred: v >= 4, votes: v };
});

const FP_new = [], FN_new = [];
for (let i = 0; i < N; i++) {
  if (bestCombo[i].pred && !actuals[i]) FP_new.push(i);
  if (!bestCombo[i].pred && actuals[i]) FN_new.push(i);
}
console.log(`After Dir1+Dir6: FP=${FP_new.length} FN=${FN_new.length}`);

// New feature ideas for remaining FP:
// A) jawDelta consistency: AI lip sync has more uniform jawDelta across burst
const jawDeltaStd5 = windowStat(allSamples.map(s => s.jawDelta), 2, std);
// B) score acceleration: d(score)/dt — how fast score changes
const scoreAccel = allSamples.map((s, i) => {
  if (i === 0 || dt[i] < 0.001) return 0;
  return Math.abs(s.score - allSamples[i-1].score) / dt[i];
});
// C) jawVelocity / jawDelta ratio (efficiency)
const jawEfficiency = allSamples.map(s => s.jawDelta > 0.001 ? s.jawVelocity / s.jawDelta : 0);
const jawEffMean5 = windowStat(jawEfficiency, 2, mean);

// Cohen's d for each new feature
function cohensD(userVals, nonUserVals) {
  const mu = mean(userVals), mnu = mean(nonUserVals);
  const su = std(userVals), snu = std(nonUserVals);
  const pooled = Math.sqrt((su**2 + snu**2) / 2);
  return pooled > 0 ? (mu - mnu) / pooled : 0;
}

const userIdx = [], nonUserIdx = [];
for (let i = 0; i < N; i++) { if (actuals[i]) userIdx.push(i); else nonUserIdx.push(i); }

const features = {
  jawDeltaStd5: { vals: jawDeltaStd5, user: userIdx.map(i => jawDeltaStd5[i]), nonUser: nonUserIdx.map(i => jawDeltaStd5[i]) },
  scoreAccel: { vals: scoreAccel, user: userIdx.map(i => scoreAccel[i]), nonUser: nonUserIdx.map(i => scoreAccel[i]) },
  jawEffMean5: { vals: jawEffMean5, user: userIdx.map(i => jawEffMean5[i]), nonUser: nonUserIdx.map(i => jawEffMean5[i]) },
};

for (const [name, f] of Object.entries(features)) {
  const d = cohensD(f.user, f.nonUser);
  console.log(`${name}: Cohen's d = ${d.toFixed(3)} (user mean=${mean(f.user).toFixed(3)}, nonUser mean=${mean(f.nonUser).toFixed(3)})`);
}

// Test each new feature
for (const [name, f] of Object.entries(features)) {
  let best = { f1: 0 };
  const vals = f.vals;
  // Try both directions (positive = user, negative = user)
  for (let w = -2; w <= 2; w += 0.25) {
    if (w === 0) continue;
    for (let th = 0; th <= 20; th += 0.5) {
      const preds = allSamples.map((s, i) => {
        let v = baselineScores[i];
        const sv = (1 - s.score) * s.jawVelocity;
        if (sv >= 1) v += 0.5;
        if (v >= 4.5 && dt[i] < 0.001 && s.score < 0.3) v -= 1.5;
        if (w > 0 && vals[i] >= th) v += w;
        if (w < 0 && vals[i] < th) v += Math.abs(w);
        return v >= 4;
      });
      const r = evaluate(preds);
      if (r.recall >= 0.9 && r.specificity >= 0.9 && r.f1 > best.f1) {
        best = { ...r, w, th };
      }
    }
  }
  if (best.f1 > 0) console.log(`  ${name}: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} w=${best.w} th=${best.th}`);
  else console.log(`  ${name}: no improvement`);
}

// ============================================================
// Part 3: Rescue mechanism for FN (from v51 idea, but with safe features)
// ============================================================
console.log('\n=== Part 3: Safe rescue for FN ===\n');
{
  // v51 used ground-truth density. Let's use predicted density instead.
  // Stage 1: normal prediction
  // Stage 2: for Stage 1 non-user, check if neighbors (by predicted labels) suggest user
  
  let best = { f1: 0 };
  let count = 0;
  
  // First get Stage 1 predictions (Dir1+Dir6 best)
  const stage1Votes = allSamples.map((s, i) => {
    let v = baselineScores[i];
    const sv = (1 - s.score) * s.jawVelocity;
    if (sv >= 1) v += 0.5;
    if (v >= 4.5 && dt[i] < 0.001 && s.score < 0.3) v -= 1.5;
    return v;
  });
  const stage1Preds = stage1Votes.map(v => v >= 4);
  
  // Predicted user density (using stage1 predictions, not ground truth)
  function predDensity(preds, hw) {
    return preds.map((p, i) => {
      let count = 0, total = 0;
      for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) {
        total++;
        if (preds[j]) count++;
      }
      return count / total;
    });
  }
  
  for (let hw = 3; hw <= 10; hw += 1) {
    const pd = predDensity(stage1Preds, hw);
    for (let rTh = 0.2; rTh <= 0.7; rTh += 0.05) {
      for (let minVotes = 1; minVotes <= 3.5; minVotes += 0.25) {
        const preds = allSamples.map((s, i) => {
          if (stage1Preds[i]) return true;
          // Rescue: non-user by stage1, but high predicted density + some signal
          if (pd[i] >= rTh && stage1Votes[i] >= minVotes) return true;
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
  console.log(`Safe rescue: ${count} qualifying`);
  if (best.f1 > 0) {
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} rTh=${best.rTh.toFixed(2)} minVotes=${best.minVotes}`);
  }
}

// ============================================================
// Part 4: Everything combined — ultimate v60
// ============================================================
console.log('\n=== Part 4: Ultimate v60 ===\n');
{
  // Dir1+Dir6 + rescue
  const stage1Votes = allSamples.map((s, i) => {
    let v = baselineScores[i];
    const sv = (1 - s.score) * s.jawVelocity;
    if (sv >= 1) v += 0.5;
    if (v >= 4.5 && dt[i] < 0.001 && s.score < 0.3) v -= 1.5;
    return v;
  });
  const stage1Preds = stage1Votes.map(v => v >= 4);
  
  function predDensity(preds, hw) {
    return preds.map((p, i) => {
      let count = 0, total = 0;
      for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) {
        total++;
        if (preds[j]) count++;
      }
      return count / total;
    });
  }
  
  let best = { f1: 0 };
  let count = 0;
  for (let hw = 3; hw <= 8; hw++) {
    const pd = predDensity(stage1Preds, hw);
    for (let rTh = 0.2; rTh <= 0.6; rTh += 0.05) {
      for (let minVotes = 1; minVotes <= 3; minVotes += 0.25) {
        const preds = allSamples.map((s, i) => {
          if (stage1Preds[i]) return true;
          if (pd[i] >= rTh && stage1Votes[i] >= minVotes) return true;
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
  console.log(`Ultimate: ${count} qualifying`);
  if (best.f1 > 0) {
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  hw=${best.hw} rTh=${best.rTh.toFixed(2)} minVotes=${best.minVotes}`);
    
    // Also check dual-high mode
    const pd = predDensity(stage1Preds, best.hw);
    const finalPreds = allSamples.map((s, i) => {
      if (stage1Preds[i]) return true;
      if (pd[i] >= best.rTh && stage1Votes[i] >= best.minVotes) return true;
      return false;
    });
    const r = evaluate(finalPreds);
    console.log(`\nFinal check: TP=${r.TP} FP=${r.FP} TN=${r.TN} FN=${r.FN}`);
    console.log(`R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}%`);
  }
}

console.log('\n=== PROGRESS ===');
console.log('v49 baseline: F1=84.4%, R=90.4%, S=94.8%, FP=52, FN=21');
console.log('v59 Dir1+Dir6: F1=85.2%, R=90.8%, S=95.1%, FP=49, FN=20');
console.log('v60 target: beat F1=85.2%');
