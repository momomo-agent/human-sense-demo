// autoresearch v90: Re-eval v88 on merged dataset (2299 tokens)
const fs = require('fs');
const DATA = '/Users/kenefe/LOCAL/momo-agent/projects/human-sense-demo/Tests/speaker-test-data.jsonl';
const lines = fs.readFileSync(DATA, 'utf8').trim().split('\n');
const all = lines.map(l => JSON.parse(l));
all.sort((a, b) => a.audioTime - b.audioTime);
const N = all.length;
const act = all.map(s => s.isUserSpeaker);
const dt = all.map((s, i) => i === 0 ? 0 : s.audioTime - all[i - 1].audioTime);
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
function wstat(arr, hw, fn) {
  return arr.map((_, i) => {
    const w = [];
    for (let j = Math.max(0, i - hw); j <= Math.min(N - 1, i + hw); j++) w.push(arr[j]);
    return fn(w);
  });
}
function ev(preds) {
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<N;i++){if(preds[i]&&act[i])TP++;else if(preds[i]&&!act[i])FP++;else if(!preds[i]&&!act[i])TN++;else FN++;}
  const r=TP/(TP+FN)||0,sp=TN/(TN+FP)||0,pr=TP/(TP+FP)||0,f1=2*pr*r/(pr+r)||0;
  return {TP,FP,TN,FN,recall:r,specificity:sp,f1};
}

const scoreGap = all.map(s => {
  const jw = s.jawWeight || 0.2;
  const jvw = s.jawVelocityWeight || jw;
  const jawF = Math.max(0.1, 1.0 - jw * s.jawDelta);
  const velF = Math.max(0.1, 1.0 - jvw * s.jawVelocity);
  const nmF = (s.jawDelta < 0.02 && s.jawVelocity < 0.1) ? 1.5 : 1.0;
  const finalScore = s.score * jawF * velF * nmF;
  return Math.abs(finalScore - s.score);
});

const dtEnt5 = wstat(dt, 2, a => {const b=[0,0,0];a.forEach(v=>{if(v<0.001)b[0]++;else if(v<0.1)b[1]++;else b[2]++;});let e=0;const n=a.length;b.forEach(x=>{if(x>0){const p=x/n;e-=p*Math.log2(p);}});return e;});
const burstLen = (() => {const bl=new Array(N).fill(1);for(let i=1;i<N;i++){if(dt[i]<0.001)bl[i]=bl[i-1]+1;}for(let i=N-2;i>=0;i--){if(dt[i+1]<0.001)bl[i]=Math.max(bl[i],bl[i+1]);}return bl;})();
const velStd5=wstat(all.map(s=>s.jawVelocity),2,std);
const scoreStd5=wstat(all.map(s=>s.score),2,std);
const jawEff=all.map(s=>s.jawDelta>0.001?s.jawVelocity/s.jawDelta:0);
const jawEffMean5=wstat(jawEff,2,mean);
const scoreAccel=all.map((s,i)=>{if(i===0||dt[i]<0.001)return 0;return Math.abs(s.score-all[i-1].score)/dt[i];});
const scoreSlope5 = wstat(all.map(s=>s.score), 2, a => {if(a.length<2)return 0;const n=a.length,mx=mean(a.map((_,i)=>i)),my=mean(a);let num=0,den=0;a.forEach((y,x)=>{num+=(x-mx)*(y-my);den+=(x-mx)**2;});return den>0?num/den:0;});
const scoreVelAnti=all.map(s=>(1-s.score)*s.jawVelocity);
const isHighJW = all.map(s => (s.jawWeight || 0) > 0.5);

function v88votes(s, i) {
  let v=0;
  if(s.score<0.45)v+=3;else if(s.score<0.5)v+=0.75;else if(s.score<0.72)v+=0.25;
  if(s.jawDelta>=0.1)v+=0.25;else if(s.jawDelta>=0.05)v+=0.125;
  if(s.jawVelocity>=0.5)v+=4;else if(s.jawVelocity>=0.1)v+=2;else if(s.jawVelocity>=0.05)v+=1;
  if(dt[i]>=0.3)v+=1.5;else if(dt[i]>=0.03)v+=0.75;
  if(dtEnt5[i]>=0.725)v+=1;
  const sv=(1-s.score)*s.jawVelocity;
  if(sv>=0.875)v+=0.375;
  if(scoreAccel[i]>=1.5)v+=0.75;
  if(jawEffMean5[i]<4.5)v+=0.25;
  if(scoreGap[i]>=0.425)v+=1.75;
  if(scoreSlope5[i]<-0.1)v+=0.5;
  if(scoreVelAnti[i]>=0.3)v+=0.375;
  if(s.score>=0.3&&s.score<0.7&&dt[i]<0.001&&s.jawVelocity>=0.15) v -= 1.625;
  if(velStd5[i]>=0.6&&dt[i]<0.001) v -= 0.875;
  if(scoreStd5[i]<0.12&&dt[i]<0.001) v -= 0.375;
  if(v>=4.25&&dt[i]<0.001&&s.score<0.35) v -= 1.75;
  if(s.score>=0.7 && s.jawVelocity>=0.4 && dt[i]<0.001) v -= 2.0;
  if(!isHighJW[i]) {
    if(dt[i]>=0.001 && s.score>=0.75) v -= 3.0;
    if(dt[i]<0.001 && s.score>=0.3 && s.jawVelocity>=0.5) v -= 1.5;
  }
  if(!isHighJW[i] && dt[i]<0.001 && s.jawVelocity>=0.4 && s.score<0.4) v -= 2.0;
  if(!isHighJW[i] && burstLen[i]<=2 && dt[i]<0.001) v -= 1.75;
  if(!isHighJW[i] && dtEnt5[i]<0.75) v -= 2.25;
  return v;
}

const sc = all.map((s,i) => v88votes(s, i));
const p1 = sc.map(v => v >= 4);
const preds = all.map((_,i) => {
  if(p1[i]) return true;
  if(all[i].jawVelocity < 0.1) return false;
  const hw = isHighJW[i] ? 6 : 10;
  const nTh = isHighJW[i] ? 0.15 : 0.6;
  const low = isHighJW[i] ? -5 : -1;
  if(sc[i] < low) return false;
  let userN=0, total=0;
  for(let j=Math.max(0,i-hw);j<=Math.min(N-1,i+hw);j++){
    if(j===i) continue;
    total++;
    if(p1[j]) userN++;
  }
  return total>0 && userN/total >= nTh;
});

const r = ev(preds);
console.log(`Dataset: ${N} tokens (${act.filter(x=>x).length} user, ${act.filter(x=>!x).length} AI)`);
console.log(`v88: R=${(r.recall*100).toFixed(1)}% S=${(r.specificity*100).toFixed(1)}% F1=${(r.f1*100).toFixed(1)}% FP=${r.FP} FN=${r.FN}`);

// Breakdown: old data vs new data
const oldN = 1212;
let oTP=0,oFP=0,oTN=0,oFN=0;
let nTP=0,nFP=0,nTN=0,nFN=0;
// Note: data is sorted by audioTime, so old/new are interleaved
// Use original index tracking instead
const origIndices = lines.map((l,i) => i); // before sort
// Actually just report overall + error analysis

console.log('\n=== Error analysis ===');
const FP=[], FN=[];
for(let i=0;i<N;i++){
  if(preds[i]&&!act[i])FP.push(i);
  if(!preds[i]&&act[i])FN.push(i);
}
console.log(`FP: ${FP.length}`);
FP.slice(0, 30).forEach(i => {
  const s=all[i];
  console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} jw=${isHighJW[i]?1:0.2} v=${sc[i].toFixed(2)} bl=${burstLen[i]}`);
});
if(FP.length>30) console.log(`  ... and ${FP.length-30} more`);

console.log(`\nFN: ${FN.length}`);
FN.slice(0, 30).forEach(i => {
  const s=all[i];
  console.log(`  i=${i} "${s.text}" sc=${s.score.toFixed(3)} vel=${s.jawVelocity.toFixed(3)} dt=${dt[i].toFixed(4)} jw=${isHighJW[i]?1:0.2} v=${sc[i].toFixed(2)} bl=${burstLen[i]}`);
});
if(FN.length>30) console.log(`  ... and ${FN.length-30} more`);
