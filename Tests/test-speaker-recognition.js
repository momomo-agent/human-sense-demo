#!/usr/bin/env node

/**
 * Speaker Recognition Test Suite
 * 
 * Tests the speaker recognition algorithm against labeled test data.
 * Run: node Tests/test-speaker-recognition.js
 */

const fs = require('fs');
const path = require('path');

// Read test data
const testDataPath = path.join(__dirname, 'speaker-test-data.jsonl');
const lines = fs.readFileSync(testDataPath, 'utf8').trim().split('\n');
const samples = lines.map(line => JSON.parse(line));

console.log(`Loaded ${samples.length} test samples`);
console.log(`User samples: ${samples.filter(s => s.isUserSpeaker).length}`);
console.log(`Non-user samples: ${samples.filter(s => !s.isUserSpeaker).length}\n`);

// Test current algorithm
function testAlgorithm(samples, jawWeight, jawVelocityWeight, threshold, minJawDelta, minJawVelocity, scoreThreshold) {
  let TP = 0, FP = 0, TN = 0, FN = 0;
  
  for (const sample of samples) {
    let predicted;
    
    // Rule 1: If score is very high, classify as non-user
    if (sample.score > scoreThreshold) {
      predicted = false;
    }
    // Rule 2: If jaw is completely still, classify as non-user
    else if (sample.jawDelta < minJawDelta && sample.jawVelocity < minJawVelocity) {
      predicted = false;
    }
    // Rule 3: Use multiplicative weighting
    else {
      const jawFactor = 1.0 - jawWeight * sample.jawDelta;
      const velocityFactor = 1.0 - jawVelocityWeight * sample.jawVelocity;
      const finalScore = sample.score * jawFactor * velocityFactor;
      predicted = finalScore < threshold;
    }
    
    const actual = sample.isUserSpeaker;
    
    if (predicted && actual) TP++;
    else if (predicted && !actual) FP++;
    else if (!predicted && !actual) TN++;
    else FN++;
  }
  
  const precision = TP / (TP + FP) || 0;
  const recall = TP / (TP + FN) || 0;
  const f1 = 2 * precision * recall / (precision + recall) || 0;
  const accuracy = (TP + TN) / samples.length;
  
  return { TP, FP, TN, FN, precision, recall, f1, accuracy };
}

// Current parameters (from GazeSpeakerEngine.swift)
const params = {
  jawWeight: 0.2,
  jawVelocityWeight: 0.2,
  threshold: 0.7,
  minJawDelta: 0.02,
  minJawVelocity: 0.1,
  scoreThreshold: 0.75
};

console.log('=== Current Algorithm Performance ===\n');
console.log('Parameters:');
console.log(`  jawWeight: ${params.jawWeight}`);
console.log(`  jawVelocityWeight: ${params.jawVelocityWeight}`);
console.log(`  threshold: ${params.threshold}`);
console.log(`  minJawDelta: ${params.minJawDelta}`);
console.log(`  minJawVelocity: ${params.minJawVelocity}`);
console.log(`  scoreThreshold: ${params.scoreThreshold}\n`);

const result = testAlgorithm(
  samples,
  params.jawWeight,
  params.jawVelocityWeight,
  params.threshold,
  params.minJawDelta,
  params.minJawVelocity,
  params.scoreThreshold
);

console.log('Results:');
console.log(`  F1 Score:  ${(result.f1 * 100).toFixed(2)}%`);
console.log(`  Accuracy:  ${(result.accuracy * 100).toFixed(2)}%`);
console.log(`  Precision: ${(result.precision * 100).toFixed(2)}%`);
console.log(`  Recall:    ${(result.recall * 100).toFixed(2)}%\n`);

console.log('Confusion Matrix:');
console.log(`  True Positive:  ${result.TP} (correctly identified as user)`);
console.log(`  False Positive: ${result.FP} (incorrectly identified as user)`);
console.log(`  True Negative:  ${result.TN} (correctly identified as non-user)`);
console.log(`  False Negative: ${result.FN} (incorrectly identified as non-user)\n`);

// Check if performance meets minimum requirements
const MIN_F1 = 0.70;
const MIN_ACCURACY = 0.85;

if (result.f1 >= MIN_F1 && result.accuracy >= MIN_ACCURACY) {
  console.log('✅ PASS: Algorithm meets minimum requirements');
  process.exit(0);
} else {
  console.log('❌ FAIL: Algorithm does not meet minimum requirements');
  console.log(`  Required: F1 >= ${(MIN_F1 * 100).toFixed(0)}%, Accuracy >= ${(MIN_ACCURACY * 100).toFixed(0)}%`);
  process.exit(1);
}
