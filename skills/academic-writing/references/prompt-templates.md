# Prompt 模板（学术写作）

从 `config/prompts.yaml` 的 `academic_writing:` 段迁入，按需加载。

## abstract（中文）

```
请为以下研究生成学术论文摘要（中文）：
标题 {title} / 关键词 {keywords} / 研究背景 {background} / 方法 {methods}
/ 结果 {results} / 结论 {conclusion}
要求：200-300 字；含背景/方法/结果/结论四要素；语言简洁学术规范；突出创新点。
```

## abstract_en

```
Generate an academic abstract in English for the following research:
Title / Keywords / Background / Methods / Results / Conclusion
Requirements: 200-300 words; include background, methods, results, conclusion;
clear academic tone; highlight novelty and main contributions.
```

## introduction

```
请撰写论文引言部分：研究主题 {topic} / 研究问题 {research_question}
/ 相关工作 {related_work} / 本文贡献 {contributions}
要求：宏观背景→聚焦问题；综述相关研究并指出现有不足；明确创新点与贡献；
说明论文组织结构；1000-1500 字。
```

## methodology

```
请撰写方法论章节：方法名称 {method_name} / 技术细节 {technical_details} / 算法流程 {algorithm}
要求：清晰描述原理与步骤；含必要数学公式；说明图表内容；说明优势与创新；逻辑连贯。
```

## results

```
请撰写实验结果章节：实验设置 {experimental_setup} / 数据集 {datasets}
/ 评估指标 {metrics} / 结果数据 {results_data}
要求：客观呈现结果；用表格图表展示关键数据；与基线对比；含消融分析；讨论结果意义。
```

**注意**：`results_data` 必须由用户/实验真实提供；缺失时先索取，**不得生成数字**。

## discussion

```
请撰写讨论章节：主要发现 {findings} / 结果解释 {interpretation}
/ 局限性 {limitations} / 未来工作 {future_work}
要求：深入解释结果；讨论适用性与局限；与相关研究比较；提出未来方向；保持客观与批判性。
```

## polish

```
请润色以下学术文本，提升专业性与流畅性：原文 {original_text}
要求：保持原意不变；使用更学术化表达；改进句式结构；确保逻辑连贯；修正语法错误。
```
