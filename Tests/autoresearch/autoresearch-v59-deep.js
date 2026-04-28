// autoresearch v59: Deep dive on text features + cross-validation
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

// Precompute features
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
// Part 1: Analyze text feature overfitting risk
// ============================================================
console.log('=== Part 1: Text feature analysis ===\n');

// Full dataset char probabilities
const charStats = {};
allSamples.forEach(s => {
  const ch = s.text;
  if (!charStats[ch]) charStats[ch] = { user: 0, nonUser: 0, total: 0 };
  if (s.isUserSpeaker) charStats[ch].user++; else charStats[ch].nonUser++;
  charStats[ch].total++;
});

// How many chars appear in both classes?
let bothClasses = 0, userOnly = 0, nonUserOnly = 0;
for (const [ch, st] of Object.entries(charStats)) {
  if (st.user > 0 && st.nonUser > 0) bothClasses++;
  else if (st.user > 0) userOnly++;
  else nonUserOnly++;
}
console.log(`Chars in both classes: ${bothClasses}, user-only: ${userOnly}, non-user-only: ${nonUserOnly}`);
console.log(`Total unique chars: ${Object.keys(charStats).length}`);

// The 21 FN tokens — what are their texts?
const FN_indices = [];
for (let i = 0; i < N; i++) {
  if (baselineScores[i] < 4 && actuals[i]) FN_indices.push(i);
}
console.log(`\nFN tokens (${FN_indices.length}):`);
FN_indices.forEach(i => {
  const s = allSamples[i];
  const st = charStats[s.text];
  const p = st.user / st.total;
  console.log(`  "${s.text}" p(user)=${p.toFixed(2)} (${st.user}u/${st.nonUser}nu) votes=${baselineScores[i].toFixed(2)}`);
});

// ============================================================
// Part 2: Leave-one-session-out cross-validation
// ============================================================
console.log('\n=== Part 2: Cross-validation (leave-one-session-out) ===\n');

// Identify sessions
const sessionId = new Array(N).fill(0);
let sid = 0;
for (let i = 1; i < N; i++) {
  if (dt[i] > 5) sid++;
  sessionId[i] = sid;
}
const numSessions = sid + 1;
console.log(`Sessions: ${numSessions}`);

// For each session, compute char probs from OTHER sessions, then evaluate
let totalTP = 0, totalFP = 0, totalTN = 0, totalFN = 0;
for (let testSid = 0; testSid <= sid; testSid++) {
  // Train: all sessions except testSid
  const trainCharStats = {};
  for (let i = 0; i < N; i++) {
    if (sessionId[i] === testSid) continue;
    const ch = allSamples[i].text;
    if (!trainCharStats[ch]) trainCharStats[ch] = { user: 0, nonUser: 0 };
    if (actuals[i]) trainCharStats[ch].user++; else trainCharStats[ch].nonUser++;
  }
  const trainCharProb = {};
  for (const [ch, st] of Object.entries(trainCharStats)) {
    trainCharProb[ch] = st.user / (st.user + st.nonUser);
  }
  
  // Test: only testSid
  for (let i = 0; i < N; i++) {
    if (sessionId[i] !== testSid) continue;
    let v = baselineScores[i];
    const p = trainCharProb[allSamples[i].text];
    if (p !== undefined && p >= 0.8) v += 3;
    const pred = v >= 4;
    if (pred && actuals[i]) totalTP++;
    else if (pred && !actuals[i]) totalFP++;
    else if (!pred && !actuals[i]) totalTN++;
    else totalFN++;
  }
}
const cvRecall = totalTP / (totalTP + totalFN) || 0;
const cvSpec = totalTN / (totalTN + totalFP) || 0;
const cvPrec = totalTP / (totalTP + totalFP) || 0;
const cvF1 = 2 * cvPrec * cvRecall / (cvPrec + cvRecall) || 0;
console.log(`CV result: R=${(cvRecall*100).toFixed(1)}% S=${(cvSpec*100).toFixed(1)}% F1=${(cvF1*100).toFixed(1)}% TP=${totalTP} FP=${totalFP} FN=${totalFN}`);

// ============================================================
// Part 3: Text feature that doesn't overfit — use char frequency instead of label
// ============================================================
console.log('\n=== Part 3: Frequency-based text features (no label leakage) ===\n');

// Idea: rare chars (low frequency in non-user) are more likely user-specific
// This is still label-dependent. Let's try something truly label-free:
// - Text length (always 1 char in this dataset, so useless)
// - Unicode category (punctuation vs CJK vs space)
// - Whether the char appeared in the PREVIOUS user turn (context)

// Actually, the real insight is: the text feature works because certain chars
// only appear when the user speaks (unique vocabulary). In production, we don't
// have labels, but we DO have the STT output — if a char appears for the first
// time in a session, it's more likely user (AI repeats known vocabulary).

// Simulate: "first occurrence in session" feature
const firstOccurrence = new Array(N).fill(false);
const seenInSession = {};
let currentSid = -1;
for (let i = 0; i < N; i++) {
  if (sessionId[i] !== currentSid) {
    currentSid = sessionId[i];
    seenInSession[currentSid] = new Set();
  }
  const ch = allSamples[i].text;
  if (!seenInSession[currentSid].has(ch)) {
    firstOccurrence[i] = true;
    seenInSession[currentSid].add(ch);
  }
}

// How discriminative is firstOccurrence?
let foUser = 0, foNonUser = 0, nfoUser = 0, nfoNonUser = 0;
for (let i = 0; i < N; i++) {
  if (firstOccurrence[i]) {
    if (actuals[i]) foUser++; else foNonUser++;
  } else {
    if (actuals[i]) nfoUser++; else nfoNonUser++;
  }
}
console.log(`First occurrence: user=${foUser} nonUser=${foNonUser} (${(foUser/(foUser+foNonUser)*100).toFixed(1)}% user)`);
console.log(`Not first: user=${nfoUser} nonUser=${nfoNonUser} (${(nfoUser/(nfoUser+nfoNonUser)*100).toFixed(1)}% user)`);

// Try adding first occurrence as feature
let best = { f1: 0 };
for (let foW = 0; foW <= 3; foW += 0.25) {
  const preds = allSamples.map((s, i) => {
    let v = baselineScores[i];
    if (firstOccurrence[i]) v += foW;
    return v >= 4;
  });
  const r = evaluate(preds);
  if (r.recall >= 0.9 && r.specificity >= 0.9 && r.f1 > best.f1) {
    best = { ...r, foW };
  }
}
if (best.f1 > 0) console.log(`First occurrence boost: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN} foW=${best.foW}`);

// ============================================================
// Part 4: Combine best directions
// ============================================================
console.log('\n=== Part 4: Combine winning directions ===\n');

// Winners from v58:
// - Dir 1: svW=0.5, svTh=1 (marginal)
// - Dir 2: tw=3, tTh=0.8 (big win but overfitting risk)
// - Dir 6: hvTh=5, hvPen=1.5, scoreLow=0.3 (small FP reduction)

// Combine Dir 1 + Dir 6 (both safe, no label leakage)
{
  let best = { f1: 0 };
  let count = 0;
  for (let svW = 0; svW <= 1; svW += 0.25) {
    for (let svTh = 0.5; svTh <= 2; svTh += 0.25) {
      for (let hvTh = 4; hvTh <= 8; hvTh += 0.5) {
        for (let hvPen = 0.5; hvPen <= 3; hvPen += 0.5) {
          for (let scoreLow = 0.1; scoreLow <= 0.5; scoreLow += 0.1) {
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
  console.log(`Dir1+Dir6 combo: ${count} qualifying`);
  if (best.f1 > 0) console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
  if (best.f1 > 0) console.log(`  svW=${best.svW} svTh=${best.svTh} hvTh=${best.hvTh} hvPen=${best.hvPen} scoreLow=${best.scoreLow}`);
}

// ============================================================
// Part 5: Deeper text analysis — window text diversity
// ============================================================
console.log('\n=== Part 5: Window text diversity ===\n');

// In a window of 5 tokens, how many unique chars? User speech has more diversity
const textDiv5 = windowStat(allSamples.map(s => s.text), 2, a => new Set(a).size / a.length);
// User vs non-user
const userDiv = [], nonUserDiv = [];
for (let i = 0; i < N; i++) {
  if (actuals[i]) userDiv.push(textDiv5[i]); else nonUserDiv.push(textDiv5[i]);
}
console.log(`Text diversity (window 5): user mean=${mean(userDiv).toFixed(3)} nonUser mean=${mean(nonUserDiv).toFixed(3)}`);

let bestDiv = { f1: 0 };
for (let divW = 0; divW <= 3; divW += 0.25) {
  for (let divTh = 0.3; divTh <= 1; divTh += 0.1) {
    const preds = allSamples.map((s, i) => {
      let v = baselineScores[i];
      if (textDiv5[i] >= divTh) v += divW;
      return v >= 4;
    });
    const r = evaluate(preds);
    if (r.recall >= 0.9 && r.specificity >= 0.9 && r.f1 > bestDiv.f1) {
      bestDiv = { ...r, divW, divTh };
    }
  }
}
if (bestDiv.f1 > 0) console.log(`Text diversity: R=${(bestDiv.recall*100).toFixed(1)}% S=${(bestDiv.specificity*100).toFixed(1)}% F1=${(bestDiv.f1*100).toFixed(1)}% FP=${bestDiv.FP} FN=${bestDiv.FN} divW=${bestDiv.divW} divTh=${bestDiv.divTh}`);
else console.log('No qualifying config');

// ============================================================
// Part 6: Combined best of everything (safe features only)
// ============================================================
console.log('\n=== Part 6: Ultimate combo (safe features) ===\n');
{
  // Dir1 interaction + Dir6 FP filter + text diversity + first occurrence
  let best = { f1: 0 };
  let count = 0;
  const svW_range = [0, 0.25, 0.5];
  const svTh_range = [0.75, 1, 1.25];
  const hvTh_range = [5, 6, 7];
  const hvPen_range = [1, 1.5, 2];
  const scoreLow_range = [0.2, 0.3, 0.4];
  const foW_range = [0, 0.25, 0.5, 0.75];
  const divW_range = [0, 0.25, 0.5];
  const divTh_range = [0.5, 0.7, 0.9];
  
  for (const svW of svW_range) {
    for (const svTh of svTh_range) {
      for (const hvTh of hvTh_range) {
        for (const hvPen of hvPen_range) {
          for (const scoreLow of scoreLow_range) {
            for (const foW of foW_range) {
              for (const divW of divW_range) {
                for (const divTh of divTh_range) {
                  const preds = allSamples.map((s, i) => {
                    let v = baselineScores[i];
                    const sv = (1 - s.score) * s.jawVelocity;
                    if (sv >= svTh) v += svW;
                    if (v >= hvTh && dt[i] < 0.001 && s.score < scoreLow) v -= hvPen;
                    if (firstOccurrence[i]) v += foW;
                    if (textDiv5[i] >= divTh) v += divW;
                    return v >= 4;
                  });
                  const r = evaluate(preds);
                  if (r.recall >= 0.9 && r.specificity >= 0.9 && r.f1 > best.f1) {
                    best = { ...r, svW, svTh, hvTh, hvPen, scoreLow, foW, divW, divTh };
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
  console.log(`Ultimate combo: ${count} qualifying`);
  if (best.f1 > 0) {
    console.log(`Best: R=${(best.recall*100).toFixed(1)}% S=${(best.specificity*100).toFixed(1)}% F1=${(best.f1*100).toFixed(1)}% FP=${best.FP} FN=${best.FN}`);
    console.log(`  svW=${best.svW} svTh=${best.svTh} hvTh=${best.hvTh} hvPen=${best.hvPen} scoreLow=${best.scoreLow} foW=${best.foW} divW=${best.divW} divTh=${best.divTh}`);
  }
}
