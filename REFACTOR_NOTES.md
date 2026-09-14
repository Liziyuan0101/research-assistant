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

## 4. 第二阶段：把新架构接到**产品路径**（本次）

**动机**：上一阶段（§2–§5）只落地了组件与测试，产品入口 `ResearchAssistant` **仍然走
`ResearchAgentGraph`**，而且"偏好回流检索打分先验"**只存在于评测脚本** ——
`HybridRetriever` 完全没有接收偏好的入口。也就是说：组件是真的，接线是假的。

### 6.1 本次改动

| 位置 | 内容 |
|---|---|
| `agents/single_agent.py`（新增） | 单 Agent 运行时：路由一跳 → 按需加载 skill → 执行器分发 → 记忆落库；无 LLM 时如实标注 `llm_used=False` |
| `retrieval/hybrid_retriever.py` | `search()` 新增 `preference_profile / prior_lambda / memory / user_id`，并在 RRF 之后增加**个性化步骤**（偏好先验重算打分）；`last_search_meta` 如实上报 `backend / reranker_skipped / personalized` |
| `_encode_dense()`（新增） | 稠密编码统一入口：FlagEmbedding 优先，缺失时回退 sentence-transformers（**同一份** bge-m3 权重 → 与记忆层同向量空间） |
| `assistant.py` | 新增 `backend='single'\|'graph'`、`prior_lambda`、`data_dir`；`run_agent()` 按后端分发；`hybrid_search(personalize=...)`；`add_preference / get_preferences / memory_report`；`health_report` 增加 skills 与 dense_available |
| `cli.py` | `--user-id / --backend / --lambda / --pref CAT=VAL（! 前缀=负面）/ --search / --agent / --chain / --skills / --memory` |
| `api.py` | 新增 `/agent`、`/skills`、`/local-search`、`/memory/stats`、`/memory/traces` |
| `config/config.yaml` | 新增 `personalization:` 段（lambda / 偏好类别 / 半衰期 / 权重上限） |
| `scripts/demo_e2e.py`（新增） | 端到端演示，**不需要 API key** 即可跑通全链路 |

### 6.2 修掉的两个**真 bug**（都是本次实测才暴露的）

1. **链式编排顺序错误**：按**相关度**顺序执行，导致 `experiment-design` 先于
   `paper-retrieval` 跑，下游 skill 拿不到检索结果。
   改为按**依赖顺序**（`_SKILL_DEPENDENCY_ORDER`：检索 → 实验 → 写作）。
   —— 相关度决定"要不要执行"，依赖决定"按什么次序执行"，两者必须分开。
2. **Reranker 守卫写错层级**：原守卫是"FlagEmbedding 能不能 import"。但装了
   FlagEmbedding 而本地**没有** `BAAI/bge-reranker-v2-m3` 权重时，`FlagReranker(...)`
   会抛 `OSError`（离线无网可下），直接把整个检索打挂。
   改为"**权重能不能加载**"（try/except + 失败后不再重试），跳过时记
   `reranker_skipped: True` 并保持 RRF 顺序。

### 6.3 环境与外部依赖（已解决）

| 事实 | 状态 |
|---|---|
| `FlagEmbedding` | **1.3.5 已安装**（`pip show` 确认）→ 此前"FlagEmbedding 缺失"的记录已过时 |
| `BAAI/bge-m3`（稠密编码） | 已缓存 8.5G ✓ |
| `BAAI/bge-reranker-v2-m3`（精排） | **已从 hf-mirror 下载补齐（2.14G）** ✓ `reranker_skipped: False` / `reranked: True` 实测通过 |

**下载要点**（本机 `huggingface.co` 被墙：`WinError 10054`；`hf-mirror.com` 可达）：
```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 <python> -c "
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-reranker-v2-m3', allow_patterns=[
    'config.json','model.safetensors','sentencepiece.bpe.model',
    'special_tokens_map.json','tokenizer.json','tokenizer_config.json'])"
```
用 `allow_patterns` 排除 `onnx/` 与 `assets/`：只需 6 个文件、2.14G（全量会多出数百 MB 无用体积）。
实测耗时 725s。注意 Windows 无开发者模式时 HF 缓存退回**复制**而非符号链接（会多占空间）。

### 6.3b 顺带修的第三个坑：CPU 上的 fp16

`FlagReranker` / `BGEM3FlagModel` 原本硬编码 `use_fp16=True`。本机 **CPU-only**
（`torch.cuda.is_available() == False`，`resolve_device('auto') → 'cpu'`），
fp16 在 CPU 上算子缺失会直接报错。已改为**随设备决定**：
`use_fp16 = str(self.device).startswith('cuda')`。与 §6.2 的第 2 个 bug 是同一类
——"只在真跑时才暴露"。

### 6.4 复现

```bash
PYTHONPATH= HF_HUB_OFFLINE=1 <conda research_assistant>/python.exe scripts/demo_e2e.py
PYTHONPATH= HF_HUB_OFFLINE=1 <...>/python.exe -m research_assistant.cli --health
PYTHONPATH= HF_HUB_OFFLINE=1 <...>/python.exe -m research_assistant.cli \
    --user-id lzy --lambda 0.25 --pref "method=Bayesian deep learning" \
    --pref "style=!purely empirical curve fitting" \
    --search "remaining useful life prediction of lithium-ion batteries"
PYTHONPATH= HF_HUB_OFFLINE=1 <...>/python.exe -m pytest -q     # 103 passed
```

实测（demo 输出）：
- 检索后端 `bm25+dense`（sentence-transformers/FlagEmbedding），`personalized: True`
- 个性化**确实改变排序**（如偏好画像把 `Two-stage Early Prediction` 从 #2 顶到 #1）
- 链式编排按依赖序：`['paper-retrieval', 'experiment-design', 'academic-writing']`，
  实验方案拿到 **5 篇**上游检索结果
- 路由跳数 1、**LLM 调用 0**；常驻 290 字符 / 加载 1016–1178 字符
- `test_*` 共 **103 passed**

---

## 5. 未做 / 待办

- [ ] 评测集扩标：`data/eval/retrieval_eval.json` 4 条 → ≥50 条，且跨域（M9）
- [ ] 跨域语料上的偏好回流消融（当前语料同域，测不出 —— 见 §3.5）
- [ ] 端到端延迟 P95 与任务完成率（需 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`；
      注意首调用含模型加载：实测检索类任务 34.5s 里大部分是 BGE-M3 首次加载，
      精排首次调用还要再加 reranker 的 2.14G 权重加载）
- [ ] LoRA 写作指标的复现脚本（当前 ROUGE-L 0.47 / 术语 89% 无脚本可复现）


## 6. 对既有测试的影响

`tests/test_memory.py::test_invalid_preference_category_raises` 的断言已随契约变更而更新：
未知类别默认**告警并自动登记**（`PreferenceSchema(strict=True)` 可恢复旧的抛错行为）。
该测试被替换为两个测试（默认行为 + 严格模式）。

`74 passed`。
