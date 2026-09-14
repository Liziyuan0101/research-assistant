---
name: academic-writing
description: 当用户要撰写或润色学术文本时使用，含摘要/引言/方法/结果/讨论/文献综述与引用管理。触发词：写摘要/润色/引言/写论文/改写/投稿/引用格式。
tools: [CitationTool, PythonExecutorTool]
version: 1.0.0
---

# academic-writing（学术写作与润色）

## 何时用我

- 生成摘要（中/英）、引言、方法论、结果、讨论章节
- 润色已有文本，提升学术性与流畅度
- 文献综述的组织与对比
- 引用格式整理（IEEE 等）

## 何时不用我

- 用户还没有内容可写（缺方法/结果）→ 先用 `experiment-design` 或 `paper-retrieval`
- 用户要的是整篇论文从零到有 → 走 `generate_full_paper` 编排（本 skill 的六个章节依次调用）

## 输入 / 输出契约

| 项 | 说明 |
|---|---|
| 输入 | 章节所需字段（如摘要需 `title/keywords/background/methods/results/conclusion`） |
| 输出 | `{'section': str, 'text': str, 'word_count': int, 'citations': [...]}` |
| 字数约束 | 摘要 200–300 字 / 引言 1000–1500 字；**必须保持原意**，不得新增未提供的事实 |
| 风格 | 若 `memory` 中有 `style` 类偏好，按该偏好调整语气；LoRA 微调权重存在时优先使用 |

## 执行步骤

1. **确认章节与字段**：字段缺失时**先问用户**（或其上游 skill），不要臆造背景/结果。
2. **选择生成路径**：有 LoRA 微调权重 → 用微调模型；否则用基座 API。
3. **生成**：套用 `references/prompt-templates.md` 对应章节模板。
4. **引用**：需要引用时调 `CitationTool`，风格由 `config.multi_agent.citation_style` 决定。
5. **回填**：`word_count` 与 `citations` 必须真实统计，不得估算写死。

## 细节参考（按需加载）

- `references/prompt-templates.md` — 七个章节的模板（含中英摘要）
- `references/style-guide.md` — 学术写作规范、常见 AI 味句式黑名单、术语一致性表
