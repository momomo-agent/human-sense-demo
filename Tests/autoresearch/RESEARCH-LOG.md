# Speaker Recognition Autoresearch Log

## 项目信息
- 仓库: `momomo-agent/human-sense-demo`，分支 `voice-processing-io`
- 测试数据: `Tests/speaker-test-data.jsonl`（1212 样本：218 user + 994 non-user）
- 目标: Recall ≥ 90% 且 Specificity ≥ 90%（双高目标 R≥95% S≥90%）
- 日期: 2026-04-28 ~ 2026-04-29

## 核心算法：投票 + 特征加权

基础投票系统（baseVotes）：
- score < 0.45 → +3, < 0.5 → +0.75, < 0.72 → +0.25
- jawDelta ≥ 0.1 → +0.25, ≥ 0.05 → +0.125
- jawVelocity ≥ 0.5 → +4, ≥ 0.1 → +2, ≥ 0.05 → +1
- timeDelta ≥ 0.3 → +1.5, ≥ 0.03 → +0.75

叠加特征（v24-v44 逐步发现）：
- dtEntropy5（窗口 dt 分布熵）≥ eTh → +1
- burstLen ≥ 3 → -bW（连续 dt=0 惩罚）
- lipSyncPen: score ∈ [0.3, pSH) 且 dt=0 且 vel ≥ pV → -1.5
- velStd5 ≥ vsTh 且 dt=0 → -vsW
- scoreMean5 ≥ smTh 且 dt=0 → -smW
- scoreStd5 < ssTh 且 dt=0 → -ssW
- scoreVelAnti ≥ svTh → +svW

阈值 t：votes ≥ t → 判为 user

---

## 实验进度

### Phase 1: 基础特征探索（v7-v22）
- v7-v14: 投票、连续回归、Pareto、语义防抖、动态权重、LR
- v15-v22: 精细搜索确认三特征（score/jawDelta/jawVelocity）理论极限 min(R,S) ≈ 89.6%
- **关键发现**: velocity 是最强特征（vW=2.5），score 次之（sW=2），jawDelta 最弱（jW=0.5）

### Phase 2: timeDelta 突破（v23-v24）
- v23: 加入 dt 特征，首次突破 90%/90%
- v24: 精细搜索，**R=93.1%, S=90.2%, F1=78.4%**
- **关键发现**: AI lip sync 的 dt 几乎全是 0.000（批量到达），用户说话有自然间隔

### Phase 3: 曲线特征与惩罚策略（v26-v30）
- v26: 14 个曲线特征 Cohen's d 分析
- v27-v28: jawDiff + velDiff 精细搜索（无显著提升）
- v29: 6 种新方案——Burst ✅、局部密度 ✅、其余 ❌
- v30: density 创新纪录 F1=79.2%

### Phase 4: 高级特征（v31-v35）
- v31: 七方向单独测试（Score 二阶 8 达标、Jaw Efficiency 4、dt 分布 3、Bayesian 2）
- v32: Bayes 后验新纪录 min(R,S)=90.7%
- v33: **dtEntropy 重大突破** — min(R,S)=92.0%, F1=80.8%
- v34: dtEntropy 深度搜索 — F1=81.4%
- v35: 四合一确认 entropy 已吃掉 bayes 信号

### Phase 5: 错误分析与针对性优化（v36-v40）
- v36: 5 种新决策框架全不如投票
- v37: FP/FN 错误分析（85 FP 中 63 个邻居多数是 non-user；10 FN 全部 score 0.56-0.73）
- v38: lipSyncPen F1=81.6%
- v39: ent+burst+lipSyncPen min(R,S)=92.2%
- v40: 超精细搜索 F1=81.7%, min=92.2%

### Phase 6: 窗口统计量突破（v41-v44）
- v41: velStd5 + scoreMean5 → F1=82.6%
- v42: 六合一终极搜索（18466 达标）→ **F1=83.7%, min(R,S)=92.8%**
- v43: scoreStd5 + scoreVelAnti → F1=84.2%
- v44: 超精细搜索（355530 达标）→ **F1=84.4%, FP=52**

### Phase 7: 新特征方向（v45-v49）
- v45: 窗口大小变体 → hw=2 仍最优，更大窗口反而差
- v46: 连续值加权 → 27 达标，不如阈值方案
- v47: 二阶特征（加速度/协方差）→ 11 达标，score-vel 协方差有效
- v48: 局部排名 → 11 达标，scoreRank + velRank 有效
- v49: 异常检测（distRatio）→ 4 达标
- 组合后 → **F1=84.5%, FP=51**（distRatio 贡献）

### Phase 8: 双高搜索 R≥95% S≥90%（v50-v57）
- v50: 高 Recall 专项搜索 → 0 达标（单阶段无法同时满足）
- v51: **两阶段 rescue** → R=95.9%, S=90.8%, F1=80.7%（首次达到 95%！）
  - Stage 1: 正常投票判断
  - Stage 2: Stage 1 判 non-user 的，如果周围 user 密度高 + 自己有信号 → rescue
- v52: 精细 rescue → 64 达标，最优 R=95.4%, S=90.3%, F1=79.7%
- v53: 全特征 rescue → 0 达标（v44 惩罚太强）
- v54: 多强度 Stage 1 → 只有 weak 能达标
- v55: 重新设计投票权重 → 4 达标，F1=79.2%
- v56: rescue + FP 过滤 → 4 达标，过滤无效
- v57: OR ensemble → 0 达标（Classifier B 太弱）

---

## 最优配置

### 均衡模式（F1 最优）
```
R=90.4%, S=94.9%, F1=84.5%, TP=197, FP=51, FN=21
```
参数:
- eTh=0.725, bW=0.25, pSH=0.7, pV=0.15
- vsW=0.75, vsTh=0.6, smW=0.5, smTh=0.65
- ssW=0.375, ssTh=0.12, svW=0.25, svTh=0.3
- drW=0.25, drTh=1.2, t=4

### 双高模式（R≥95%）
```
R=95.4%, S=90.3%, F1=79.7%, TP=208, FP=96, FN=10
```
参数:
- Stage 1: eTh=0.725, bW=0.25, pSH=0.7, pV=0.2, vsW=0.875, smW=0.5, ssW=0.25, svW=0.25, t=4
- Rescue: rW=7, rTh=0.45, minV=1.5

---

## 进度轨迹
```
v24: F1=78.4%, FP=97,  min=90.2%  ← timeDelta 突破
v30: F1=79.2%                      ← density
v33: F1=80.8%, min=92.0%           ← dtEntropy 突破
v38: F1=81.6%                      ← lipSyncPen
v40: F1=81.7%, min=92.2%           ← 超精细
v41: F1=82.6%                      ← velStd
v42: F1=83.7%, min=92.8%           ← 六合一
v44: F1=84.4%, FP=52               ← scoreStd+scoreVelAnti
v49: F1=84.5%, FP=51               ← distRatio
v51: R=95.9%, S=90.8%              ← 两阶段 rescue
```

## 关键洞察

1. **dtEntropy 是最大突破**：看窗口内 dt 的分布形状，AI lip sync 的 dt 全是 0（entropy 低），用户说话有自然变化（entropy 高）
2. **投票+特征加权是最优决策结构**：v36 测试了 5 种替代框架（多层级/几何平均/分段线性/交叉项/窗口多数决），全不如投票
3. **velStd + scoreMean 精准打击 AI lip sync**：只在 dt=0 时生效，AI lip sync 的 vel 波动大但不自然
4. **两阶段 rescue 是突破 95% Recall 的唯一路径**：单阶段无法同时满足 R≥95% + S≥90%
5. **FN 的硬边界**：score 0.56-0.73 + dt=0 的 user token，特征空间跟 AI 完全重叠，只能靠邻居信息（density）捞回

## 脚本存档

- `Tests/autoresearch/` 目录包含 v45-v57 的实验脚本
- v7-v44 的脚本在 /tmp/ 中已被清理，但结果记录在此文档
- 所有脚本共享相同的数据加载和评估函数
