# 学术写作规范

## AI 味句式黑名单（润色时优先删）

| 黑名单 | 替换 |
|---|---|
| "在当今快速发展的……时代" | 直接说问题 |
| "值得注意的是" / "需要指出的是" | 删 |
| "综上所述"（段落内滥用） | 只在章末用一次 |
| "不仅……而且……"（连续出现） | 拆句 |
| "极大地" / "显著地"（无数据支撑） | 给数字或删 |
| "本文首次提出"（未做查新） | 改"本文提出" |

## 术语一致性表（全文统一，勿混用）

| 概念 | 统一用法 | 不混用 |
|---|---|---|
| 剩余使用寿命 | remaining useful life (RUL) | remaining lifetime / 剩余寿命 |
| 健康状态 | state of health (SOH) | health state |
| 梯次利用 | second-life utilization | echelon use |
| 降阶模型 | reduced-order model (ROM) | reduced model |
| 物理信息神经网络 | physics-informed neural network (PINN) | physics-informed NN |

## 硬约束

- 摘要字数 200–300；引言 1000–1500。写完**如实统计 `word_count`**，不要估算。
- 保持原意：润色不得引入原文没有的事实、数字或引用。
- 引用：需要引用时经 `CitationTool` 生成，风格取 `config.multi_agent.citation_style`。
- 中文稿与英文稿的摘要必须**信息等价**，不能一篇多写一篇少写。
