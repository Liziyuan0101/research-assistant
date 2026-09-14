# 检索评测集

本目录有两套评测集。**对外报数字前先看这张表。**

| 文件 | 规模 | P@5 上限 | 用途 |
|---|---|---|---|
| `corpus.json` + `retrieval_eval.json` | 23 篇 / **4 条** query | **60%** ⚠️ | 早期最小复现集，**不足以支撑任何结论** |
| `corpus_crossdomain.json` + `retrieval_eval_crossdomain.json` | **360 篇 / 30 条** query | **100%** | 跨域评测，λ 消融与分域分析用 |

## ⚠️ 为什么不能用旧的那一套报数字

旧集只有 4 条 query，且 q3/q4 各只标注 **1 篇**相关文献：

| query | 相关文献数 | P@5 上限 |
|---|---|---|
| q1 | 6 | 1.00 |
| q2 | 6 | 1.00 |
| q3 | 1 | **0.20** |
| q4 | 1 | **0.20** |
| **平均上限** | | **0.60** |

**任何高于 60% 的 Precision@5 都不可能由旧集产生。** 历史 README 里出现过的
"Precision@5 89%" 即属此类（已在 README 修正）。

## 跨域集怎么构造的

```bash
python scripts/build_crossdomain_eval.py            # 固定种子 20260914，可复现
python scripts/build_crossdomain_eval.py --dry-run   # 只看计划
```

* **语料**：OpenAlex，30 个主题 × 12 篇（8 个领域：电池/能源、时序预测、NLP/LLM、
  检索推荐、图模型、视觉、材料/健康、工业/供应链、安全/隐私、语音/机器人）。
* **相关性标签 = 论文的 OpenAlex `primary_topic.id` 是否等于该 query 的目标主题。**
  用**分类学元数据**判定，而不是"论文文本里是否出现查询词"——否则会自我实现，
  让词法方法凭空占优。
* **查询文本**由主题名**人工改写**成自然语言问句（不是直接把主题名丢进去）。
* **质量过滤**：`type:article` + `language:en` + 按被引排序取池 +
  丢弃 `primary_topic.score < 0.55` + 摘要 ≥ 300 字符。

### 已知偏差（必须在报告里声明）

1. **这是弱标注集。** 相关性来自 OpenAlex 的主题分配，不是人工标注。
   未加质量过滤时噪声极大（曾把 "Air Conditioning Repairs Mosman" 归入光伏故障检测，
   并混入乌克兰语/西语标题）；加过滤后抽查正常，**残余噪声约 1/8**。
2. **查询文本与主题名同源**，与论文文本存在词汇相关性 → **绝对指标可能偏乐观**。
   因此结论应看**对照与差分**（λ 消融、分域对比、离域画像），不要去吹绝对值。
3. 按被引排序取样 → 语料偏向**经典/高引**文献（1977–2021），不代表最新工作。

**待办**：`crossdomain_spotcheck.md` 是给人工抽查用的，剔除噪声后才适合对外使用。

## 跑评测

```bash
# 旧集（仅供回归，结论不可用）
python scripts/eval_retrieval.py --mode bm25

# 跨域集：基线
python scripts/eval_retrieval.py --mode bm25 \
    --corpus data/eval/corpus_crossdomain.json \
    --eval   data/eval/retrieval_eval_crossdomain.json

# 跨域集：个性化消融（按目标域 / 其他域分组，含离域对照组）
python scripts/eval_retrieval.py --mode crossdomain \
    --corpus data/eval/corpus_crossdomain.json \
    --eval   data/eval/retrieval_eval_crossdomain.json \
    --lam 0.0 0.1 0.2 0.3
```

## 当前基线（跨域集，360 篇 / 30 查询）

| 指标 | 全部 | 电池/能源域(6) | 其他域(24) |
|---|---|---|---|
| Precision@1 | 70.00% | — | — |
| Precision@3 | 73.33% | — | — |
| **Precision@5** | **64.67%** | 73.33% | 62.50% |
| Recall@5 | 26.94% | — | — |
| Hit@5 | 90.00% | — | — |
| MRR | 77.22% | 77.78% | 77.08% |

## 已证的负面结论（别再重复试）

`preference_profiles.json` 定义了两个画像：`battery_research`（与电池域对齐）
与 `off_domain_control`（离域对照）。三种偏好注入机制 × λ 扫描 × 分域对比的结果：

| 机制 | 结论 |
|---|---|
| `boost`（混合进文档分数） | **无正向增益**；画像对齐时反而 −3.3% |
| `tiebreak`（仅在近似并列时决定次序） | **空操作**（重排幅度小于真实分差） |
| `expand`（拼进查询） | **灾难性**，P@5 掉到 1.67%–29.17%，应弃用 |

**根因**：偏好与具体查询无关，把它混进相关性打分本质是**用先验污染相关性信号**。
详见 `crossdomain_results.json` 的逐组数据。要做个性化的话，方向应换成**不改相关性打分**的做法
（结果过滤 / 多样化重排 / 让用户显式确认查询意图）。
