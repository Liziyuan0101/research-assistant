---
name: experiment-design
description: 当用户要为研究问题设计实验方案、推荐超参、或生成可运行实验代码时使用。触发词：实验设计/实验方案/对比实验/消融/baseline/超参/生成实验代码。
tools: [PythonExecutorTool, StatisticalAnalysisTool, DataVisualizationTool]
version: 1.0.0
---

# experiment-design（实验设计与代码生成）

## 何时用我

- 给定**研究问题**，产出结构化实验方案（目标/假设/数据/方法/设置/指标/基线/风险）
- 给定**模型与数据规模**，推荐超参并说明理由
- 把实验方案落成**可运行 Python 代码**
- 对实验结果做统计分析与可视化

## 何时不用我

- 用户手上还没有相关工作/文献 → 先用 `paper-retrieval`（本 skill 的 `papers_context` 依赖它）
- 用户要的是论文正文（引言/方法章节）→ 用 `academic-writing`

## 输入 / 输出契约

| 项 | 说明 |
|---|---|
| 输入 | `research_question`（str，必需）、可选 `papers`（相关工作，用于 `{papers_context}`） |
| 输出 | `{'plan': {目标/假设/数据/方法/设置/指标/基线/风险}, 'code': str|None, 'trace': [...]}` |
| 硬约束 | **不得编造数据集或已有结果**；`papers` 为空时须显式说明"未提供相关工作，方案为通用模板" |
| 可执行性 | 生成的代码必须能直接跑；若依赖外部数据，须在首行注明数据获取方式 |

## 执行步骤

1. **取其上游**：如果 `papers` 为空且问题需要相关工作支撑，先调用 `paper-retrieval`。
2. **出方案**：按七要素生成（prompt 模板见 `references/prompt-templates.md`）。
3. **生成代码**（用户要求时）：数据加载→预处理→模型→训练→评估，含早停与日志。
4. **统计分析 / 可视化**（用户要求时）：用 `StatisticalAnalysisTool` / `DataVisualizationTool`。
5. **回填 `trace`**：记录用到的工具与关键参数，便于复现与面试复盘。

## 细节参考（按需加载）

- `references/prompt-templates.md` — 实验方案 / 代码生成 / 超参推荐三套模板
- `references/plan-schema.md` — 实验方案的字段 schema 与验收标准
