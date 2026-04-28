#!/usr/bin/env node

/**
 * Speaker Recognition Test Suite
 * 
 * Tests the weighted voting algorithm with context smoothing.
 * Run: node Tests/test-speaker-recognition.js
 */

const fs = require('fs');
const path = require('path');

const testDataPath = path.join(__dirname, 'speaker-test-data.jsonl');
const lines = fs.readFileSync(testDataPath, 'utf8').trim().split('\n');
const samples = lines.map(line => JSON.parse(line));
samples.sort((a, b) => a.audioTime - b.audioTime);

console.log(`Loaded ${samples.length} test samples`);
console.log(`User samples: ${samples.filter(s => s.isUserSpeaker).length}`);
console.log(`Non-user samples: ${samples.filter(s => !s.isUserSpeaker).length}\n`);

// Parameters (from GazeSpeakerEngine.swift)
const p = {
  scoreWeight: 2.0,
  jawWeight: 0.5,
  jawVelocityWeight: 2.0,
  contextWeight: 0.5,
  speakerThreshold: 3.0,
  contextWindow: 2,
  minScoreThreshold: 0.3,
  scoreChangeThreshold: 0.3
};

function calculateUserScore(s) {
  let votes = 0;
  if (s.score < 0.3) votes += p.scoreWeight * 2;
  else if (s.score < 0.5) votes += p.scoreWeight;
  else if (s.score < 0.7) votes += p.scoreWeight * 0.5;
  
  if (s.jawDelta >= 0.1) votes += p.jawWeight * 2;
  else if (s.jawDelta >= 0.05) votes += p.jawWeight;
  else if (s.jawDelta >= 0.02) votes += p.jawWeight * 0.5;
  
  if (s.jawVelocity >= 0.5) votes += p.jawVelocityWeight * 2;
  else if (s.jawVelocity >= 0.1) votes += p.jawVelocityWeight;
  else if (s.jawVelocity >= 0.05) votes += p.jawVelocityWeight * 0.5;
  
  return votes;
}

let TP = 0, FP = 0, TN = 0, FN = 0;

for (let i = 0; i < samples.length; i++) {
  const s = samples[i];
  let userScore = calculateUserScore(s);
  let predicted = userScore >= p.speakerThreshold;
  
  // Context smoothing: reconsider false negatives
  if (!predicted) {
    let minScore = Infinity, maxScore = -Infinity;
    for (let j = Math.max(0, i - p.contextWindow); j <= Math.min(samples.length - 1, i + p.contextWindow); j++) {
      minScore = Math.min(minScore, samples[j].score);
      maxScore = Math.max(maxScore, samples[j].score);
    }
    const nearbyUser = minScore < p.minScoreThreshold;
    const highChange = (maxScore - minScore) > p.scoreChangeThreshold;
    
    if (nearbyUser || highChange) {
      let contextVotes = userScore;
      if (nearbyUser) contextVotes += p.contextWeight;
      if (highChange) contextVotes += p.contextWeight * 0.5;
      if (contextVotes >= p.speakerThreshold) predicted = true;
    }
  }
  
  const actual = s.isUserSpeaker;
  if (predicted && actual) TP++;
  else if (predicted && !actual) FP++;
  else if (!predicted && !actual) TN++;
  else FN++;
}

const recall = TP / (TP + FN) || 0;
const specificity = TN / (TN + FP) || 0;
const precision = TP / (TP + FP) || 0;
const f1 = 2 * precision * recall / (precision + recall) || 0;
const accuracy = (TP + TN) / samples.length;

console.log('=== Weighted Voting Algorithm ===\n');
console.log('Parameters:', JSON.stringify(p, null, 2), '\n');

console.log('Results:');
console.log(`  Recall:      ${(recall * 100).toFixed(2)}%`);
console.log(`  Specificity: ${(specificity * 100).toFixed(2)}%`);
console.log(`  F1 Score:    ${(f1 * 100).toFixed(2)}%`);
console.log(`  Accuracy:    ${(accuracy * 100).toFixed(2)}%`);
console.log(`  Precision:   ${(precision * 100).toFixed(2)}%\n`);

console.log('Confusion Matrix:');
console.log(`  TP: ${TP}  FP: ${FP}`);
console.log(`  FN: ${FN}  TN: ${TN}\n`);

const MIN_RECALL = 0.93;
const MIN_SPECIFICITY = 0.85;

if (recall >= MIN_RECALL && specificity >= MIN_SPECIFICITY) {
  console.log('✅ PASS');
  process.exit(0);
} else {
  console.log('❌ FAIL');
  console.log(`  Required: Recall >= ${(MIN_RECALL*100)}%, Specificity >= ${(MIN_SPECIFICITY*100)}%`);
  process.exit(1);
}
