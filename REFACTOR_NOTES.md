# 重构说明：4-Agent supervisor 编排 → Single-Agent + Skills + 独立记忆层

> 本文记录 2026-09 的重构，以及**所有实测数字的复现方式**。
> 原则：文档里出现的每个数字都必须能由仓库内脚本重跑出来。

## 1. 动机

| 原设计的问题 | 说明 |
|---|---|
| 路由消耗 LLM 调用且不可靠 | ``_supervisor_node`` 用 LLM 生成 JSON 执行计划，解析失败即静默回退到关键词匹配 |
| 4 份 system prompt 常驻 | SUPERVISOR + 检索 + 实验 + 写作，无论任务只需要哪一个 |
| 上下文重复搬运 | ``AgentState`` 在 agent 间反复传递 ``retrieved_papers`` / ``tool_outputs`` |
| 记忆是检索的子模块 | ``retrieval/memory.py``，架构上撑不起"个性化"这一核心能力 |
| 记忆召回对中文失效 | ``query.lower().split()`` 词重叠，中文无空格 ⇒ 重叠恒为 0 |
| 偏好不影响检索 | 只把偏好拼进 prompt（"软个性化"），检索器看不到偏好 |

## 2. 改了什么

### 2.1 记忆层提升为顶层模块

```
research_assistant/memory/
├── schema.py      偏好类别的可配置 schema（类别/半衰期/权重上限/严格模式）
├── encoder.py     BGE-M3 懒加载编码器（离线可用，不可用时显式降级）
├── store.py       MemoryStore v2（衰减 / 负反馈 / 语义召回 / 成功轨迹 / 自动迁移老库）
└── personalize.py 偏好回流：preference_prior() 把偏好混合进检索打分
```

`retrieval/memory.py` 保留为 shim 转发，老导入不破。

### 2.2 能力 skill 化

```
skills/
├── paper-retrieval/SKILL.md      + references/{data-sources,scoring-and-personalization,evaluation}.md
├── experiment-design/SKILL.md    + references/{prompt-templates,plan-schema}.md
└── academic-writing/SKILL.md     + references/{prompt-templates,style-guide}.md
```

`research_assistant/skills/{registry,loader}.py` 实现渐进披露三级加载：
L0 frontmatter 常驻 → L1 SKILL.md 正文按需 → L2 references 二次按需。

### 2.3 路由去 LLM 化

删除 `SUPERVISOR_PROMPT`、`AgentType.SUPERVISOR`、`_supervisor_node`、`_simple_task_classification`；
新增 `_dispatch_node`（`SkillRegistry` 词元覆盖率打分）与 `_SKILL_TO_TASK_TYPE` / `_SKILL_TO_AGENT_KEY` 映射。
多 skill 链式能力（原 `task_type == 'complex'`）由 `skill_queue` 保留。

`ReActAgent.get_prompt()` 改为优先返回按需注入的 `skill_prompt`。

## 3. 实测结果

复现：

```bash
python scripts/measure_refactor.py --save      # M1 / M2 / M4
python scripts/memory_recall_demo.py --save    # 记忆召回对照
python scripts/eval_retrieval.py --mode bm25   # 检索基线
python scripts/eval_retrieval.py --mode personalized --lam 0 0.1 0.2 0.3 --prior-mode boost expand
python -m pytest -q                            # 74 passed
```

### 3.1 M1 常驻上下文（tiktoken cl100k_base）

| | tokens |
|---|---|
| 重构前：SUPERVISOR_PROMPT | 224 |
| 重构前：RETRIEVAL_AGENT_PROMPT | 167 |
| 重构前：EXPERIMENT_AGENT_PROMPT | 228 |
| 重构前：WRITING_AGENT_PROMPT | 232 |
| **重构前合计（每请求都付）** | **851** |
| 重构后：3 个 skill 的 frontmatter 索引 | **242** |
| **常驻下降** | **↓71.6%** |
| 命中单个 skill 后的最坏情况 | 1011 |

各 skill 正文（按需，不进常驻）：paper-retrieval 769 / experiment-design 626 / academic-writing 593。

### 3.2 M2 路由

| | 重构前 | 重构后 |
|---|---|---|
| 路由函数 | 4 个 | 3 个 |
| **路由 LLM 调用** | **1 次** | **0 次** |
| 每请求决策 | 1(supervisor) + 1(_route_task) + N(_check_next_agent) | 1(registry.select) |

skill 选择冒烟测试（5 条中英任务，0 次 LLM 调用）全部命中预期 skill。

### 3.3 M4 代码行数

| | 重构前 | 重构后 |
|---|---|---|
| `multi_agent.py` 文件总行数 | 820 | 855 (+35) |
| 其中**路由层**行数 | 60 | **50 (↓16.7%)** |

> 文件总行数上升是因为度量埋点与 docstring 写在同一文件；路由层本身是净减少。

### 3.4 记忆召回（修复中文失效）

| query | 旧实现（词重叠） | 新实现（BGE-M3 cos） |
|---|---|---|
| 电池剩余寿命预测 | 5/5 条历史重叠为 0（排序退化） | top1 = 锂离子电池剩余寿命预测方法，cos 0.908 |
| remaining useful life prediction... | 词重叠全为 0 | top1 = 锂离子电池剩余寿命预测方法，cos 0.764 |

扩标后新增覆盖：可配置类别（含自动登记）、时间衰减（半衰期处减半，已单测）、权重上限、负反馈、
成功轨迹读取、老库自动迁移、降级时 `backend='lexical'` 如实标注。

### 3.5 ⚠️ 偏好回流检索：**未测出增益（负面结果）**

同一 corpus / query，仅变先验强度与机制：

| 机制:画像 | λ=0 | 0.1 | 0.2 | 0.3 | 结论 |
|---|---|---|---|---|---|
| boost:battery_research | 55.0% | 55.0% | 50.0% | 50.0% | 无害或略有害 |
| boost:off_domain_control | 55.0% | 55.0% | 55.0% | 55.0% | P@1 50%→75% |
| expand:battery_research | 55.0% | 35.0% | 35.0% | 35.0% | **显著有害** |
| expand:off_domain_control | 55.0% | 35.0% | 35.0% | 35.0% | 同幅变差 |

**归因**：评测语料 23 篇**全为锂电池**，用户画像在"领域"这一维对全部文档近似等权，
没有区分度；且对照组（离域画像）变化幅度与同域画像几乎一致，
说明当前评测集**无法分辨**个性化信号。这是统计功效问题，不是机制问题。

**结论：在评测集扩标（≥50 query）且**跨域**之前，不得宣称个性化带来的检索增益。**

## 4. 未做 / 待办

- [ ] 评测集扩标：`data/eval/retrieval_eval.json` 4 条 → ≥50 条，且跨域（M9）
- [ ] 跨域语料上的偏好回流消融（当前语料同域，测不出）
- [ ] 端到端延迟 P95 与任务完成率（需 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`）
- [ ] LoRA 写作指标的复现脚本（当前 ROUGE-L 0.47 / 术语 89% 无脚本可复现）

## 5. 对既有测试的影响

`tests/test_memory.py::test_invalid_preference_category_raises` 的断言已随契约变更而更新：
未知类别默认**告警并自动登记**（`PreferenceSchema(strict=True)` 可恢复旧的抛错行为）。
该测试被替换为两个测试（默认行为 + 严格模式）。

`74 passed`。
