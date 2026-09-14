# Prompt 模板（实验设计）

从 `config/prompts.yaml` 的 `experiment_design:` 段迁入本 skill，作为按需加载资源
（原先全量常驻，现在只在 `experiment-design` skill 被选中后才进入上下文）。

## planning

```
作为实验设计专家，请为以下研究问题设计完整的实验方案：
研究问题: {research_question}
相关论文参考: {papers_context}
请提供：1 目标与假设 2 数据需求与预处理 3 方法选择及理由
4 实验设置（参数/超参）5 评估指标与基线 6 预期结果与验证 7 潜在挑战与解决方案
以结构化方式输出，便于后续实现。
```

`papers_context` 为空时，**必须在输出开头声明"未提供相关工作，以下为通用模板"**。

## code_generation

```
基于以下实验方案，生成完整的 Python 实现代码：
实验方案: {experiment_plan}
要求：结构清晰含注释；含数据加载/预处理/模型/训练/评估；
使用早停与学习率调度；含日志与结果保存；代码可直接运行。
```

## parameter_suggestion

```
为以下实验推荐参数：模型类型 {model_type} / 数据集规模 {dataset_size} / 任务类型 {task_type}
参考论文设置: {paper_parameters}
请推荐：1 学习率与优化器 2 批次大小 3 训练轮数 4 正则化 5 模型特定参数，并说明理由。
```
