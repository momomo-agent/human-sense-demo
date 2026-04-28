#!/usr/bin/env node

/**
 * Speaker Recognition Test Suite
 * 
 * Tests the speaker recognition algorithm against labeled test data.
 * Includes context smoothing (score change rate + nearby low score).
 * Run: node Tests/test-speaker-recognition.js
 */

const fs = require('fs');
const path = require('path');

const testDataPath = path.join(__dirname, 'speaker-test-data.jsonl');
const lines = fs.readFileSync(testDataPath, 'utf8').trim().split('\n');
const samples = lines.map(line => JSON.parse(line));

// Sort by audioTime for context smoothing
samples.sort((a, b) => a.audioTime - b.audioTime);

console.log(`Loaded ${samples.length} test samples`);
console.log(`User samples: ${samples.filter(s => s.isUserSpeaker).length}`);
console.log(`Non-user samples: ${samples.filter(s => !s.isUserSpeaker).length}\n`);

// Parameters (from GazeSpeakerEngine.swift)
const params = {
  jawWeight: 0.1,
  jawVelocityWeight: 0.1,
  threshold: 0.72,
  minJawDelta: 0.015,
  minJawVelocity: 0.05,
  scoreThreshold: 0.8,
  // Context smoothing params
  contextWindow: 2,
  minScoreThreshold: 0.3,
  scoreChangeThreshold: 0.3
};

function testAlgorithm(samples, p) {
  let TP = 0, FP = 0, TN = 0, FN = 0;
  
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    let predicted;
    
    // Step 1: Base prediction
    if (s.score > p.scoreThreshold) {
      predicted = false;
    } else if (s.jawDelta < p.minJawDelta && s.jawVelocity < p.minJawVelocity) {
      predicted = false;
    } else {
      const jf = Math.max(0.1, 1.0 - p.jawWeight * s.jawDelta);
      const vf = Math.max(0.1, 1.0 - p.jawVelocityWeight * s.jawVelocity);
      predicted = s.score * jf * vf < p.threshold;
    }
    
    // Step 2: Context smoothing — reconsider false negatives
    if (!predicted) {
      let minScore = Infinity, maxScore = -Infinity;
      for (let j = Math.max(0, i - p.contextWindow); j <= Math.min(samples.length - 1, i + p.contextWindow); j++) {
        minScore = Math.min(minScore, samples[j].score);
        maxScore = Math.max(maxScore, samples[j].score);
      }
      const nearbyUser = minScore < p.minScoreThreshold;
      const highChange = (maxScore - minScore) > p.scoreChangeThreshold;
      
      if (nearbyUser || highChange) {
        const jf = Math.max(0.1, 1.0 - p.jawWeight * s.jawDelta);
        const vf = Math.max(0.1, 1.0 - p.jawVelocityWeight * s.jawVelocity);
        if (s.score * jf * vf < p.threshold) {
          predicted = true;
        }
      }
    }
    
    const actual = s.isUserSpeaker;
    if (predicted && actual) TP++;
    else if (predicted && !actual) FP++;
    else if (!predicted && !actual) TN++;
    else FN++;
  }
  
  const precision = TP / (TP + FP) || 0;
  const recall = TP / (TP + FN) || 0;
  const f1 = 2 * precision * recall / (precision + recall) || 0;
  const accuracy = (TP + TN) / samples.length;
  const specificity = TN / (TN + FP) || 0;
  return { TP, FP, TN, FN, precision, recall, f1, accuracy, specificity };
}

console.log('=== Algorithm Performance ===\n');
console.log('Parameters:', JSON.stringify(params, null, 2), '\n');

const result = testAlgorithm(samples, params);

console.log('Results:');
console.log(`  Recall:      ${(result.recall * 100).toFixed(2)}% (user speech recognition)`);
console.log(`  Specificity: ${(result.specificity * 100).toFixed(2)}% (AI speech exclusion)`);
console.log(`  F1 Score:    ${(result.f1 * 100).toFixed(2)}%`);
console.log(`  Accuracy:    ${(result.accuracy * 100).toFixed(2)}%`);
console.log(`  Precision:   ${(result.precision * 100).toFixed(2)}%\n`);

console.log('Confusion Matrix:');
console.log(`  True Positive:  ${result.TP} (correctly identified as user)`);
console.log(`  False Positive: ${result.FP} (incorrectly identified as user)`);
console.log(`  True Negative:  ${result.TN} (correctly identified as non-user)`);
console.log(`  False Negative: ${result.FN} (incorrectly identified as non-user)\n`);

const MIN_RECALL = 0.95;
const MIN_SPECIFICITY = 0.75;

if (result.recall >= MIN_RECALL && result.specificity >= MIN_SPECIFICITY) {
  console.log('✅ PASS');
  process.exit(0);
} else {
  console.log('❌ FAIL');
  console.log(`  Required: Recall >= ${(MIN_RECALL*100).toFixed(0)}%, Specificity >= ${(MIN_SPECIFICITY*100).toFixed(0)}%`);
  process.exit(1);
}
