# 检索评测

## 指标定义（`retrieval/evaluation.py::evaluate_retriever`）

| 指标 | 定义 |
|---|---|
| P@k | `|relevant ∩ top_k| / k` |
| R@k | `|relevant ∩ top_k| / |relevant|` |
| Hit@k | `top_k` 中是否命中至少一篇相关 |
| MRR | 第一篇相关文档的倒数排名 |

## 可复现命令

```bash
python scripts/eval_retrieval.py --mode bm25                     # 纯离线基线
python scripts/eval_retrieval.py --mode hybrid                   # 需 FlagEmbedding
python scripts/eval_retrieval.py --mode personalized --lam 0 0.1 0.2 0.3
python scripts/memory_recall_demo.py --save                      # 记忆召回对照
```

## ⚠️ 当前评测集的硬伤

`data/eval/retrieval_eval.json` 只有 **4 条 query**，而 `corpus.json` 只有 **23 篇**。

| query | 相关文献数 | P@5 上限 |
|---|---|---|
| q1 | 6 | 1.00 |
| q2 | 6 | 1.00 |
| q3 | **1** | 0.20 |
| q4 | **1** | 0.20 |
| **平均上限** | | **0.60** |

⇒ **4 条 query 的评测集上，P@5 物理上限是 60%**，任何"89%"之类的说法都不可能由本评测集产生。

## 扩标要求（M9）

1. query 数 **≥ 50**；单域最多占 1/3，其余跨域（NLP / 推荐 / CV / 时序）。
2. 每条 query 标注 **≥ 5 篇**相关文献（否则 P@k 上限被单条 query 卡死）。
3. 标注口径固定：以 **摘要** 是否直接回答该 query 为准，不以标题字面匹配为准。
4. 标注完成后重跑 `--mode bm25` 与 `--mode personalized`，把
   `total_queries`、`P@5` 一起记进结果 JSON。
5. 报告任何提升时，**必须同时给出对照组**（off-domain 画像）与 λ 扫描。
