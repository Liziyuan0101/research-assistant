# 检索打分与个性化注入点

## 三层存储

```
data/
├── index/      faiss_hnsw.index + bm25.pkl      ← 索引层
├── metadata/   papers.db (papers + chunks)      ← 元数据层
└── cache/      *.pdf                            ← 缓存层
```

## 召回链路

```
query ──┬─ BM25 稀疏 (rank_bm25, jieba 分词)      ──┐
        └─ BGE-M3 稠密 (FAISS HNSW, 余弦)          ──┴─ RRF 融合 ── BGE-Reranker 精排
```

RRF 公式（`hybrid_retriever.py:_rrf_fusion`）：

```
RRF(chunk) = Σ_signal  weight_signal / (k + rank_signal + 1),   k = 60
```

## 个性化注入点（三选一，按侵入性排序）

| 位置 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. 文档打分混合** | `final = (1-λ)·minmax(base) + λ·minmax(pref_sim)` | λ=0 可复现基线，消融最干净 | 会稀释检索信号 |
| **B. 查询侧扩展** | `query' = query + 偏好文本` 后重算 BM25 | 不用改打分 | 长偏好会稀释查询 |
| **C. 重排前过滤** | 先取 top-N，再用偏好重排 | 不影响召回 | 召回阶段仍漏 |

**实现位置**：`research_assistant/memory/personalize.py::preference_prior`；
评测入口 `scripts/eval_retrieval.py --mode personalized --prior-mode {boost,expand}`。

## ⚠️ 实测结论（务必先读再宣称收益）

在**同域语料**（23 篇全为锂电池）上实测：

| 机制 | λ=0 | 0.1 | 0.2 | 0.3 | 结论 |
|---|---|---|---|---|---|
| boost | 55.0% | 55.0% | 50.0% | 50.0% | 无害或略有害 |
| expand | 55.0% | 35.0% | 35.0% | 35.0% | **显著有害** |

同域语料下，用户画像在"领域"这一维对所有文档近似等权，**没有区分度**；
先验只在**跨域语料**（用户库里混有电池/NLP/推荐）上才可能产生增益。

**因此：在评测集扩标且跨域之前，不得在任何对外材料中宣称个性化带来的检索增益。**
对照组（off-domain 画像）表现与同域画像几乎一致，进一步说明当前评测集
**无法分辨**个性化信号 —— 这是统计功效问题，不是机制问题。
