---
name: paper-retrieval
description: 当用户要检索学术论文、查找相关工作、构建文献库，或解读单篇论文时使用。触发词：检索/找论文/相关工作/文献综述/这篇论文讲了什么/检索评测。
tools: [PaperRetrievalTool, PythonExecutorTool]
version: 1.0.0
---

# paper-retrieval（论文检索与解读）

## 何时用我

- 需要**找到**论文：给定研究问题/关键词，从外部源检索并建立本地索引
- 需要**精准召回**：在已建索引中按语义+关键词做混合检索
- 需要**解读**单篇：给出该论文的结构化总结（背景/方法/结果/局限）
- 需要**评估**检索质量：算 P@k / Recall@k / MRR / Hit@k

## 何时不用我

- 用户要的是**实验方案**而不是文献 → 用 `experiment-design`
- 用户要的是**成稿写作/润色** → 用 `academic-writing`
- 用户只想要论文链接、不需要建索引 → 直接 `search_papers`，不必走完整 skill

## 输入 / 输出契约

| 项 | 说明 |
|---|---|
| 输入 | `query`（str，中英文均可）、可选 `top_k`、可选 `papers`（已检索到的论文列表） |
| 输出 | `{'papers': [...], 'scores': [...], 'backend': 'hybrid'|'bm25', 'trace': [...]} ` |
| 必须回填 | `backend` 字段。若稠密检索不可用而降级到 BM25，**必须如实标注**，不得静默降级 |
| 个性化 | 若 `memory` 提供偏好画像，把偏好先验并入打分（见 `references/scoring-and-personalization.md`） |

## 执行步骤

1. **判断检索阶段**：本地索引为空 → 先 `search_papers` 建索引；已有索引 → 直接 `hybrid_search`。
2. **查询处理**：中文查询先转标准英文学术术语（QueryEnhancer），再检索。
3. **检索**：BM25 稀疏 + BGE-M3 稠密 → RRF 融合 →（可选）BGE-Reranker 精排。
4. **个性化**（可选）：`memory/preference_prior` 以 λ 混合偏好先验；**λ=0 必须可复现基线**。
5. **回填 `backend` 与 `trace`**，并把本次问答写入长期记忆。

## 细节参考（按需加载）

- `references/data-sources.md` — 五个外部论文源的字段、限流、鉴权与失败处理
- `references/scoring-and-personalization.md` — 三层存储、RRF 公式、偏好先验的注入点与消融口径
- `references/evaluation.md` — 指标定义、可复现评测命令、评测集构造与扩标方法
